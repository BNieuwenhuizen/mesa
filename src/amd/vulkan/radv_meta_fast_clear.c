/*
 * Copyright © 2016 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <assert.h>
#include <stdbool.h>

#include "radv_meta.h"
#include "radv_private.h"
#include "nir/nir_builder.h"
#include "sid.h"
/**
 * Vertex attributes used by all pipelines.
 */
struct vertex_attrs {
	float position[2]; /**< 3DPRIM_RECTLIST */
	float tex_position[2];
};

static void
meta_fast_clear_flush_save(struct radv_meta_saved_state *saved_state,
                  struct radv_cmd_buffer *cmd_buffer)
{
	radv_meta_save(saved_state, cmd_buffer, 0);
}

static void
meta_fast_clear_flush_restore(struct radv_meta_saved_state *saved_state,
                     struct radv_cmd_buffer *cmd_buffer)
{
	radv_meta_restore(saved_state, cmd_buffer);
}

/* passthrough vertex shader */
static nir_shader *
build_nir_vs(void)
{
	const struct glsl_type *vec4 = glsl_vec4_type();

	nir_builder b;
	nir_variable *a_position;
	nir_variable *v_position;
	nir_variable *a_tex_position;
	nir_variable *v_tex_position;

	nir_builder_init_simple_shader(&b, NULL, MESA_SHADER_VERTEX, NULL);
	b.shader->info.name = ralloc_strdup(b.shader, "meta_fast_clear_vs");

	a_position = nir_variable_create(b.shader, nir_var_shader_in, vec4,
					 "a_position");
	a_position->data.location = VERT_ATTRIB_GENERIC0;

	v_position = nir_variable_create(b.shader, nir_var_shader_out, vec4,
					 "gl_Position");
	v_position->data.location = VARYING_SLOT_POS;

	a_tex_position = nir_variable_create(b.shader, nir_var_shader_in, vec4,
					     "a_tex_position");
	a_tex_position->data.location = VERT_ATTRIB_GENERIC1;

	v_tex_position = nir_variable_create(b.shader, nir_var_shader_out, vec4,
					     "v_tex_position");
	v_tex_position->data.location = VARYING_SLOT_VAR0;

	nir_copy_var(&b, v_position, a_position);
	nir_copy_var(&b, v_tex_position, a_tex_position);

	return b.shader;
}

/* simple passthrough shader */
static nir_shader *
build_nir_fs(void)
{
	const struct glsl_type *vec4 = glsl_vec4_type();
	nir_builder b;
	nir_variable *v_tex_position; /* vec4, varying texture coordinate */
	nir_variable *f_color; /* vec4, fragment output color */

	nir_builder_init_simple_shader(&b, NULL, MESA_SHADER_FRAGMENT, NULL);
	b.shader->info.name = ralloc_asprintf(b.shader,
					      "meta_fast_clear_fs");

	v_tex_position = nir_variable_create(b.shader, nir_var_shader_in, vec4,
					     "v_tex_position");
	v_tex_position->data.location = VARYING_SLOT_VAR0;

	f_color = nir_variable_create(b.shader, nir_var_shader_out, vec4,
				      "f_color");
	f_color->data.location = FRAG_RESULT_DATA0;

	nir_copy_var(&b, f_color, v_tex_position);

	return b.shader;
}

static VkResult
create_pass(struct radv_device *device)
{
	VkResult result;
	VkDevice device_h = radv_device_to_handle(device);
	const VkAllocationCallbacks *alloc = &device->meta_state.alloc;
	VkAttachmentDescription attachment;

	attachment.format = VK_FORMAT_UNDEFINED;
	attachment.samples = 1;
	attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
	attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;

	result = radv_CreateRenderPass(device_h,
				       &(VkRenderPassCreateInfo) {
					       .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
						       .attachmentCount = 1,
						       .pAttachments = &attachment,
						       .subpassCount = 1,
						       .pSubpasses = &(VkSubpassDescription) {
						       .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
						       .inputAttachmentCount = 0,
						       .colorAttachmentCount = 1,
						       .pColorAttachments = (VkAttachmentReference[]) {
							       {
								       .attachment = 0,
								       .layout = VK_IMAGE_LAYOUT_GENERAL,
							       },
						       },
						       .pResolveAttachments = NULL,
						       .pDepthStencilAttachment = &(VkAttachmentReference) {
							       .attachment = VK_ATTACHMENT_UNUSED,
						       },
						       .preserveAttachmentCount = 0,
						       .pPreserveAttachments = NULL,
					       },
								.dependencyCount = 0,
				       },
				       alloc,
				       &device->meta_state.fast_clear_flush.pass);

	return result;
}

static VkResult
create_pipeline(struct radv_device *device,
                VkShaderModule vs_module_h)
{
	VkResult result;
	VkDevice device_h = radv_device_to_handle(device);

	struct radv_shader_module fs_module = {
		.nir = build_nir_fs(),
	};

	if (!fs_module.nir) {
		/* XXX: Need more accurate error */
		result = VK_ERROR_OUT_OF_HOST_MEMORY;
		goto cleanup;
	}

	result = radv_graphics_pipeline_create(device_h,
					       radv_pipeline_cache_to_handle(&device->meta_state.cache),
					       &(VkGraphicsPipelineCreateInfo) {
						       .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
						       .stageCount = 2,
						       .pStages = (VkPipelineShaderStageCreateInfo[]) {
						       {
							       .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
							       .stage = VK_SHADER_STAGE_VERTEX_BIT,
							       .module = vs_module_h,
							       .pName = "main",
						       },
						       {
							       .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
							       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
							       .module = radv_shader_module_to_handle(&fs_module),
							       .pName = "main",
						       },
					       },
					       .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo) {
						       .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
						       .vertexBindingDescriptionCount = 1,
						       .pVertexBindingDescriptions = (VkVertexInputBindingDescription[]) {
							       {
								       .binding = 0,
								       .stride = sizeof(struct vertex_attrs),
								       .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
							       },
						       },
						       .vertexAttributeDescriptionCount = 2,
						       .pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]) {
							       {
								       /* Position */
								       .location = 0,
								       .binding = 0,
								       .format = VK_FORMAT_R32G32_SFLOAT,
								       .offset = offsetof(struct vertex_attrs, position),
							       },
							       {
								       /* Texture Coordinate */
								       .location = 1,
								       .binding = 0,
								       .format = VK_FORMAT_R32G32_SFLOAT,
								       .offset = offsetof(struct vertex_attrs, tex_position),
							       },
						       },
					       },
					       .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo) {
						       .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
						       .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
						       .primitiveRestartEnable = false,
					       },
					       .pViewportState = &(VkPipelineViewportStateCreateInfo) {
						       .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
						       .viewportCount = 0,
						       .scissorCount = 0,
					       },
					       .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo) {
						       .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
						       .depthClampEnable = false,
						       .rasterizerDiscardEnable = false,
						       .polygonMode = VK_POLYGON_MODE_FILL,
						       .cullMode = VK_CULL_MODE_NONE,
						       .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
					       },
					       .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo) {
						       .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
						       .rasterizationSamples = 1,
						       .sampleShadingEnable = false,
						       .pSampleMask = NULL,
						       .alphaToCoverageEnable = false,
						       .alphaToOneEnable = false,
					       },
					       .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo) {
						       .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
						       .logicOpEnable = false,
						       .attachmentCount = 1,
						       .pAttachments = (VkPipelineColorBlendAttachmentState []) {
							       {
							       .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
									       VK_COLOR_COMPONENT_G_BIT |
									       VK_COLOR_COMPONENT_B_BIT |
									       VK_COLOR_COMPONENT_A_BIT,
							       },
						       },
						},
						.pDynamicState = NULL,
						.renderPass = device->meta_state.fast_clear_flush.pass,
						.subpass = 0,
					       },
					       &(struct radv_graphics_pipeline_create_info) {
						       .use_rectlist = true,
						       .custom_blend_mode = V_028808_CB_ELIMINATE_FAST_CLEAR,
					       },
					       &device->meta_state.alloc,
					       &device->meta_state.fast_clear_flush.pipeline);
	if (result != VK_SUCCESS)
		goto cleanup;

	goto cleanup;

cleanup:
	ralloc_free(fs_module.nir);
	return result;
}

void
radv_device_finish_meta_fast_clear_flush_state(struct radv_device *device)
{
	struct radv_meta_state *state = &device->meta_state;
	VkDevice device_h = radv_device_to_handle(device);
	VkRenderPass pass_h = device->meta_state.fast_clear_flush.pass;
	const VkAllocationCallbacks *alloc = &device->meta_state.alloc;

	if (pass_h)
		RADV_CALL(DestroyRenderPass)(device_h, pass_h,
					     &device->meta_state.alloc);

	VkPipeline pipeline_h = state->fast_clear_flush.pipeline;
	if (pipeline_h) {
		RADV_CALL(DestroyPipeline)(device_h, pipeline_h, alloc);
	}
}

VkResult
radv_device_init_meta_fast_clear_flush_state(struct radv_device *device)
{
	VkResult res = VK_SUCCESS;

	zero(device->meta_state.fast_clear_flush);

	struct radv_shader_module vs_module = { .nir = build_nir_vs() };
	if (!vs_module.nir) {
		/* XXX: Need more accurate error */
		res = VK_ERROR_OUT_OF_HOST_MEMORY;
		goto fail;
	}

	res = create_pass(device);
	if (res != VK_SUCCESS)
		goto fail;

	VkShaderModule vs_module_h = radv_shader_module_to_handle(&vs_module);
	res = create_pipeline(device, vs_module_h);
	if (res != VK_SUCCESS)
		goto fail;

	goto cleanup;

fail:
	radv_device_finish_meta_fast_clear_flush_state(device);

cleanup:
	ralloc_free(vs_module.nir);

	return res;
}

static void
emit_fast_clear_flush(struct radv_cmd_buffer *cmd_buffer,
		      const VkExtent2D *resolve_extent)
{
	struct radv_device *device = cmd_buffer->device;
	VkCommandBuffer cmd_buffer_h = radv_cmd_buffer_to_handle(cmd_buffer);
	uint32_t offset;
	const struct vertex_attrs vertex_data[3] = {
		{
			.position = {
				0,
				0,
			},
			.tex_position = {
				0,
				0,
			},
		},
		{
			.position = {
				0,
				resolve_extent->height,
			},
			.tex_position = {
				0,
				resolve_extent->height,
			},
		},
		{
			.position = {
				resolve_extent->width,
				0,
			},
			.tex_position = {
				resolve_extent->width,
				0,
			},
		},
	};

	cmd_buffer->state.flush_bits |= (RADV_CMD_FLAG_FLUSH_AND_INV_CB |
					 RADV_CMD_FLAG_FLUSH_AND_INV_CB_META);
	radv_cmd_buffer_upload_data(cmd_buffer, sizeof(vertex_data), 16, vertex_data, &offset);
	struct radv_buffer vertex_buffer = {
		.device = device,
		.size = sizeof(vertex_data),
		.bo = &cmd_buffer->upload.upload_bo,
		.offset = offset,
	};

	VkBuffer vertex_buffer_h = radv_buffer_to_handle(&vertex_buffer);

	radv_CmdBindVertexBuffers(cmd_buffer_h,
				  /*firstBinding*/ 0,
				  /*bindingCount*/ 1,
				  (VkBuffer[]) { vertex_buffer_h },
				  (VkDeviceSize[]) { 0 });

	VkPipeline pipeline_h = device->meta_state.fast_clear_flush.pipeline;
	RADV_FROM_HANDLE(radv_pipeline, pipeline, pipeline_h);

	if (cmd_buffer->state.pipeline != pipeline) {
		radv_CmdBindPipeline(cmd_buffer_h, VK_PIPELINE_BIND_POINT_GRAPHICS,
				     pipeline_h);
	}

	RADV_CALL(CmdDraw)(cmd_buffer_h, 3, 1, 0, 0);
	cmd_buffer->state.flush_bits |= (RADV_CMD_FLAG_FLUSH_AND_INV_CB |
					 RADV_CMD_FLAG_FLUSH_AND_INV_CB_META);
	si_emit_cache_flush(cmd_buffer);
}

/**
 */
void
radv_fast_clear_flush_image_inplace(struct radv_cmd_buffer *cmd_buffer,
				    struct radv_image *image)
{
	struct radv_meta_saved_state saved_state;
	struct radv_meta_saved_pass_state saved_pass_state;
	VkDevice device_h = radv_device_to_handle(cmd_buffer->device);
	VkCommandBuffer cmd_buffer_h = radv_cmd_buffer_to_handle(cmd_buffer);

	if (!image->cmask.size)
		return;

	if (!cmd_buffer->device->allow_fast_clears)
		return;

	radv_meta_save_pass(&saved_pass_state, cmd_buffer);
	meta_fast_clear_flush_save(&saved_state, cmd_buffer);

	struct radv_image_view iview;
	radv_image_view_init(&iview, cmd_buffer->device,
			     &(VkImageViewCreateInfo) {
				     .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
					     .image = radv_image_to_handle(image),
					     .format = image->vk_format,
					     .subresourceRange = {
						     .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						     .baseMipLevel = 0,
						     .levelCount = 1,
						     .baseArrayLayer = 0,
						     .layerCount = 1,
					     },
				     },
				     cmd_buffer, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

	VkFramebuffer fb_h;
	radv_CreateFramebuffer(device_h,
			       &(VkFramebufferCreateInfo) {
				       .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				       .attachmentCount = 1,
				       .pAttachments = (VkImageView[]) {
					       radv_image_view_to_handle(&iview)
				       },
				       .width = image->extent.width,
				       .height = image->extent.height,
				       .layers = 1
			      },
			      &cmd_buffer->pool->alloc,
			      &fb_h);

	RADV_CALL(CmdBeginRenderPass)(cmd_buffer_h,
				      &(VkRenderPassBeginInfo) {
					      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
						      .renderPass = cmd_buffer->device->meta_state.fast_clear_flush.pass,
						      .framebuffer = fb_h,
						      .renderArea = {
						      .offset = {
							      0,
							      0,
						      },
						      .extent = {
							      image->extent.width,
							      image->extent.height,
						      }
					      },
					      .clearValueCount = 0,
					      .pClearValues = NULL,
				     },
				     VK_SUBPASS_CONTENTS_INLINE);

	emit_fast_clear_flush(cmd_buffer,
			      &(VkExtent2D) { image->extent.width, image->extent.height });
	RADV_CALL(CmdEndRenderPass)(cmd_buffer_h);

	radv_DestroyFramebuffer(device_h, fb_h,
				&cmd_buffer->pool->alloc);

	meta_fast_clear_flush_restore(&saved_state, cmd_buffer);
	radv_meta_restore_pass(&saved_pass_state, cmd_buffer);
}
