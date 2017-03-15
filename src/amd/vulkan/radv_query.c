/*
 * Copyrigh 2016 Red Hat Inc.
 * Based on anv:
 * Copyright © 2015 Intel Corporation
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
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "nir/nir_builder.h"
#include "radv_meta.h"
#include "radv_private.h"
#include "radv_cs.h"
#include "sid.h"

static unsigned get_max_db(struct radv_device *device)
{
	unsigned num_db = device->physical_device->rad_info.num_render_backends;
	MAYBE_UNUSED unsigned rb_mask = device->physical_device->rad_info.enabled_rb_mask;

	if (device->physical_device->rad_info.chip_class == SI)
		num_db = 8;
	else
		num_db = MAX2(8, num_db);

	/* Otherwise we need to change the query reset procedure */
	assert(rb_mask == ((1ull << num_db) - 1));

	return num_db;
}

static void break_on_count(nir_builder *b, nir_variable *var, int count)
{
	nir_ssa_def *counter = nir_load_var(b, var);

	nir_if *if_stmt = nir_if_create(b->shader);
	if_stmt->condition = nir_src_for_ssa(nir_uge(b, counter, nir_imm_int(b, count)));
	nir_cf_node_insert(b->cursor, &if_stmt->cf_node);

	b->cursor = nir_after_cf_list(&if_stmt->then_list);

	nir_jump_instr *instr = nir_jump_instr_create(b->shader, nir_jump_break);
	nir_builder_instr_insert(b, &instr->instr);

	b->cursor = nir_after_cf_node(&if_stmt->cf_node);
	counter = nir_iadd(b, counter, nir_imm_int(b, 1));
	nir_store_var(b, var, counter, 0x1);
}

static nir_shader *
build_occlusion_query_shader(struct radv_device *device) {
	nir_builder b;
	nir_builder_init_simple_shader(&b, NULL, MESA_SHADER_COMPUTE, NULL);
	b.shader->info->name = ralloc_strdup(b.shader, "occlusion_query");
	b.shader->info->cs.local_size[0] = 1;
	b.shader->info->cs.local_size[1] = 1;
	b.shader->info->cs.local_size[2] = 1;

	nir_variable *result = nir_local_variable_create(b.impl, glsl_uint64_t_type(), "result");
	nir_variable *inner_counter = nir_local_variable_create(b.impl, glsl_int_type(), "inner_counter");
	nir_variable *outer_counter = nir_local_variable_create(b.impl, glsl_int_type(), "outer_counter");
	nir_variable *start = nir_local_variable_create(b.impl, glsl_uint64_t_type(), "start");
	nir_variable *end = nir_local_variable_create(b.impl, glsl_uint64_t_type(), "end");
	nir_variable *available = nir_local_variable_create(b.impl, glsl_int_type(), "available");
	unsigned db_count = get_max_db(device);

	nir_intrinsic_instr *flags = nir_intrinsic_instr_create(b.shader, nir_intrinsic_load_push_constant);
	flags->src[0] = nir_src_for_ssa(nir_imm_int(&b, 0));
	flags->num_components = 1;
	nir_ssa_dest_init(&flags->instr, &flags->dest, 1, 32, "flags");
	nir_builder_instr_insert(&b, &flags->instr);

	nir_intrinsic_instr *dst_buf = nir_intrinsic_instr_create(b.shader,
	                                                          nir_intrinsic_vulkan_resource_index);
	dst_buf->src[0] = nir_src_for_ssa(nir_imm_int(&b, 0));
	nir_intrinsic_set_desc_set(dst_buf, 0);
	nir_intrinsic_set_binding(dst_buf, 0);
	nir_ssa_dest_init(&dst_buf->instr, &dst_buf->dest, 1, 32, NULL);
	nir_builder_instr_insert(&b, &dst_buf->instr);

	nir_intrinsic_instr *src_buf = nir_intrinsic_instr_create(b.shader,
	                                                          nir_intrinsic_vulkan_resource_index);
	src_buf->src[0] = nir_src_for_ssa(nir_imm_int(&b, 0));
	nir_intrinsic_set_desc_set(src_buf, 0);
	nir_intrinsic_set_binding(src_buf, 1);
	nir_ssa_dest_init(&src_buf->instr, &src_buf->dest, 1, 32, NULL);
	nir_builder_instr_insert(&b, &src_buf->instr);

	nir_store_var(&b, result, nir_imm_int64(&b, 0), 0x1);
	nir_store_var(&b, outer_counter, nir_imm_int(&b, 0), 0x1);
	nir_store_var(&b, available, nir_imm_int(&b, 1), 0x1);

	nir_loop *outer_loop = nir_loop_create(b.shader);
	nir_builder_cf_insert(&b, &outer_loop->cf_node);
	b.cursor = nir_after_cf_list(&outer_loop->body);

	nir_ssa_def *current_outer_count = nir_load_var(&b, outer_counter);
	break_on_count(&b, outer_counter, db_count);

	nir_store_var(&b, inner_counter, nir_imm_int(&b, 0), 0x1);

	nir_loop *inner_loop = nir_loop_create(b.shader);
	nir_builder_cf_insert(&b, &inner_loop->cf_node);
	b.cursor = nir_after_cf_list(&inner_loop->body);

	/* to prevent an infinite loop. count = 1, because values are cached for now */
	break_on_count(&b, inner_counter, 1);

	nir_ssa_def *load_offset = nir_imul(&b, current_outer_count, nir_imm_int(&b, 16));

	nir_intrinsic_instr *load = nir_intrinsic_instr_create(b.shader, nir_intrinsic_load_ssbo);
	load->src[0] = nir_src_for_ssa(&src_buf->dest.ssa);
	load->src[1] = nir_src_for_ssa(load_offset);
	nir_ssa_dest_init(&load->instr, &load->dest, 2, 64, NULL);
	load->num_components = 2;
	nir_builder_instr_insert(&b, &load->instr);

	const unsigned swizzle0[] = {0,0,0,0};
	const unsigned swizzle1[] = {1,1,1,1};
	nir_store_var(&b, start, nir_swizzle(&b, &load->dest.ssa, swizzle0, 1, false), 0x1);
	nir_store_var(&b, end, nir_swizzle(&b, &load->dest.ssa, swizzle1, 1, false), 0x1);

	nir_ssa_def *start_done = nir_ilt(&b, nir_load_var(&b, start), nir_imm_int64(&b, 0));
	nir_ssa_def *end_done = nir_ilt(&b, nir_load_var(&b, end), nir_imm_int64(&b, 0));
	nir_ssa_def *skip_wait = nir_iand(&b, nir_inot(&b, &flags->dest.ssa), nir_imm_int(&b, VK_QUERY_RESULT_WAIT_BIT));

	nir_if *done_if = nir_if_create(b.shader);
	done_if->condition = nir_src_for_ssa(nir_ior(&b, nir_iand(&b, start_done, end_done), skip_wait));
	nir_cf_node_insert(b.cursor, &done_if->cf_node);

	b.cursor = nir_after_cf_list(&done_if->then_list);

	nir_jump_instr *break_instr = nir_jump_instr_create(b.shader, nir_jump_break);
	nir_builder_instr_insert(&b, &break_instr->instr);

	b.cursor = nir_after_cf_list(&done_if->else_list);

	b.cursor = nir_after_cf_node(&inner_loop->cf_node);

	start_done = nir_ilt(&b, nir_load_var(&b, start), nir_imm_int64(&b, 0));
	end_done = nir_ilt(&b, nir_load_var(&b, end), nir_imm_int64(&b, 0));

	nir_if *update_if = nir_if_create(b.shader);
	update_if->condition = nir_src_for_ssa(nir_iand(&b, start_done, end_done));
	nir_cf_node_insert(b.cursor, &update_if->cf_node);

	b.cursor = nir_after_cf_list(&update_if->then_list);

	nir_store_var(&b, result, nir_iadd(&b, nir_load_var(&b, result), nir_isub(&b, nir_load_var(&b, end), nir_load_var(&b, start))), 0x1);

	b.cursor = nir_after_cf_list(&update_if->else_list);

	nir_store_var(&b, available, nir_imm_int(&b, 0), 0x1);

	b.cursor = nir_after_cf_node(&outer_loop->cf_node);

	/* Store the result if complete or if partial results have been requested. */

	nir_ssa_def *result_is_64bit = nir_iand(&b, &flags->dest.ssa,
	                                        nir_imm_int(&b, VK_QUERY_RESULT_64_BIT));
	nir_ssa_def *result_size = nir_bcsel(&b, result_is_64bit, nir_imm_int(&b, 8), nir_imm_int(&b, 4));

	nir_if *store_if = nir_if_create(b.shader);
	store_if->condition = nir_src_for_ssa(nir_ior(&b, nir_iand(&b, &flags->dest.ssa, nir_imm_int(&b, VK_QUERY_RESULT_PARTIAL_BIT)), nir_load_var(&b, available)));
	nir_cf_node_insert(b.cursor, &store_if->cf_node);

	b.cursor = nir_after_cf_list(&store_if->then_list);

	nir_if *store_64bit_if = nir_if_create(b.shader);
	store_64bit_if->condition = nir_src_for_ssa(result_is_64bit);
	nir_cf_node_insert(b.cursor, &store_64bit_if->cf_node);

	b.cursor = nir_after_cf_list(&store_64bit_if->then_list);

	nir_intrinsic_instr *store = nir_intrinsic_instr_create(b.shader, nir_intrinsic_store_ssbo);
	store->src[0] = nir_src_for_ssa(nir_load_var(&b, result));
	store->src[1] = nir_src_for_ssa(&dst_buf->dest.ssa);
	store->src[2] = nir_src_for_ssa(nir_imm_int(&b, 0));
	nir_intrinsic_set_write_mask(store, 0x1);
	store->num_components = 1;
	nir_builder_instr_insert(&b, &store->instr);

	b.cursor = nir_after_cf_list(&store_64bit_if->else_list);

	store = nir_intrinsic_instr_create(b.shader, nir_intrinsic_store_ssbo);
	store->src[0] = nir_src_for_ssa(nir_u2u32(&b, nir_load_var(&b, result)));
	store->src[1] = nir_src_for_ssa(&dst_buf->dest.ssa);
	store->src[2] = nir_src_for_ssa(nir_imm_int(&b, 0));
	nir_intrinsic_set_write_mask(store, 0x1);
	store->num_components = 1;
	nir_builder_instr_insert(&b, &store->instr);

	b.cursor = nir_after_cf_node(&store_if->cf_node);

	/* Store the availability bit if requested. */

	nir_if *availability_if = nir_if_create(b.shader);
	availability_if->condition = nir_src_for_ssa(nir_iand(&b, &flags->dest.ssa, nir_imm_int(&b, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)));
	nir_cf_node_insert(b.cursor, &availability_if->cf_node);

	b.cursor = nir_after_cf_list(&availability_if->then_list);

	store = nir_intrinsic_instr_create(b.shader, nir_intrinsic_store_ssbo);
	store->src[0] = nir_src_for_ssa(nir_load_var(&b, available));
	store->src[1] = nir_src_for_ssa(&dst_buf->dest.ssa);
	store->src[2] = nir_src_for_ssa(result_size);
	nir_intrinsic_set_write_mask(store, 0x1);
	store->num_components = 1;
	nir_builder_instr_insert(&b, &store->instr);

	return b.shader;
}

VkResult radv_device_init_meta_query_state(struct radv_device *device)
{
	VkResult result;
	struct radv_shader_module occlusion_cs = { .nir = NULL };

	zero(device->meta_state.query);

	occlusion_cs.nir = build_occlusion_query_shader(device);

	VkDescriptorSetLayoutCreateInfo occlusion_ds_create_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 2,
		.pBindings = (VkDescriptorSetLayoutBinding[]) {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = NULL
			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = NULL
			},
		}
	};

	result = radv_CreateDescriptorSetLayout(radv_device_to_handle(device),
						&occlusion_ds_create_info,
						&device->meta_state.alloc,
						&device->meta_state.query.occlusion_query_ds_layout);
	if (result != VK_SUCCESS)
		goto fail;

	VkPipelineLayoutCreateInfo occlusion_pl_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &device->meta_state.query.occlusion_query_ds_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_COMPUTE_BIT, 0, 8},
	};

	result = radv_CreatePipelineLayout(radv_device_to_handle(device),
					  &occlusion_pl_create_info,
					  &device->meta_state.alloc,
					  &device->meta_state.query.occlusion_query_p_layout);
	if (result != VK_SUCCESS)
		goto fail;

	VkPipelineShaderStageCreateInfo occlusion_pipeline_shader_stage = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_COMPUTE_BIT,
		.module = radv_shader_module_to_handle(&occlusion_cs),
		.pName = "main",
		.pSpecializationInfo = NULL,
	};

	VkComputePipelineCreateInfo occlusion_vk_pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = occlusion_pipeline_shader_stage,
		.flags = 0,
		.layout = device->meta_state.query.occlusion_query_p_layout,
	};

	result = radv_CreateComputePipelines(radv_device_to_handle(device),
					     radv_pipeline_cache_to_handle(&device->meta_state.cache),
					     1, &occlusion_vk_pipeline_info, NULL,
					     &device->meta_state.query.occlusion_query_pipeline);
	if (result != VK_SUCCESS)
		goto fail;

	return VK_SUCCESS;
fail:
	radv_device_finish_meta_query_state(device);
	ralloc_free(occlusion_cs.nir);
	return result;
}

void radv_device_finish_meta_query_state(struct radv_device *device)
{
	if (device->meta_state.query.occlusion_query_pipeline)
		radv_DestroyPipeline(radv_device_to_handle(device),
				     device->meta_state.query.occlusion_query_pipeline,
				     &device->meta_state.alloc);

	if (device->meta_state.query.occlusion_query_p_layout)
		radv_DestroyPipelineLayout(radv_device_to_handle(device),
					   device->meta_state.query.occlusion_query_p_layout,
					   &device->meta_state.alloc);

	if (device->meta_state.query.occlusion_query_ds_layout)
		radv_DestroyDescriptorSetLayout(radv_device_to_handle(device),
						device->meta_state.query.occlusion_query_ds_layout,
						&device->meta_state.alloc);
}

static void occlusion_query_shader(struct radv_cmd_buffer *cmd_buffer,
                                   struct radeon_winsys_bo *src_bo,
                                   struct radeon_winsys_bo *dst_bo,
                                   uint64_t src_offset, uint64_t dst_offset,
                                   uint32_t count, uint32_t flags)
{
	struct radv_device *device = cmd_buffer->device;
	struct radv_meta_saved_compute_state saved_state;
	unsigned stride = get_max_db(device) * 16;
	VkDescriptorSet ds;

	radv_meta_save_compute(&saved_state, cmd_buffer, 4);

	radv_temp_descriptor_set_create(device, cmd_buffer,
					device->meta_state.query.occlusion_query_ds_layout,
					&ds);

	struct radv_buffer dst_buffer = {
		.bo = dst_bo,
		.offset = dst_offset,
		.size = 16 * count
	};

	struct radv_buffer src_buffer = {
		.bo = src_bo,
		.offset = src_offset,
		.size = stride * count
	};

	radv_UpdateDescriptorSets(radv_device_to_handle(device),
				  2, /* writeCount */
				  (VkWriteDescriptorSet[]) {
					  {
						  .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						  .dstSet = ds,
						  .dstBinding = 0,
						  .dstArrayElement = 0,
						  .descriptorCount = 1,
						  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						  .pBufferInfo = &(VkDescriptorBufferInfo) {
							.buffer = radv_buffer_to_handle(&dst_buffer),
							.offset = 0,
							.range = 16 * count
						  }
					  },
					  {
						  .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						  .dstSet = ds,
						  .dstBinding = 1,
						  .dstArrayElement = 0,
						  .descriptorCount = 1,
						  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						  .pBufferInfo = &(VkDescriptorBufferInfo) {
							.buffer = radv_buffer_to_handle(&src_buffer),
							.offset = 0,
							.range = stride * count
						  }
					  }
				  }, 0, NULL);

	radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd_buffer),
			     VK_PIPELINE_BIND_POINT_COMPUTE,
			     device->meta_state.query.occlusion_query_pipeline);

	radv_CmdBindDescriptorSets(radv_cmd_buffer_to_handle(cmd_buffer),
				   VK_PIPELINE_BIND_POINT_COMPUTE,
				   device->meta_state.query.occlusion_query_p_layout, 0, 1,
				   &ds, 0, NULL);


	radv_CmdPushConstants(radv_cmd_buffer_to_handle(cmd_buffer),
				      device->meta_state.query.occlusion_query_p_layout,
				      VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
				      &flags);

	cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_INV_GLOBAL_L2 |
	                                RADV_CMD_FLAG_INV_VMEM_L1;

	if (flags & VK_QUERY_RESULT_WAIT_BIT)
		cmd_buffer->state.flush_bits |= RADV_CMD_FLUSH_AND_INV_FRAMEBUFFER;

	radv_unaligned_dispatch(cmd_buffer, count, 1, 1);

	cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_INV_GLOBAL_L2 |
	                                RADV_CMD_FLAG_INV_VMEM_L1 |
	                                RADV_CMD_FLAG_CS_PARTIAL_FLUSH;

	radv_temp_descriptor_set_destroy(device, ds);

	radv_meta_restore_compute(&saved_state, cmd_buffer, 4);
}

VkResult radv_CreateQueryPool(
	VkDevice                                    _device,
	const VkQueryPoolCreateInfo*                pCreateInfo,
	const VkAllocationCallbacks*                pAllocator,
	VkQueryPool*                                pQueryPool)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	uint64_t size;
	struct radv_query_pool *pool = vk_alloc2(&device->alloc, pAllocator,
					       sizeof(*pool), 8,
					       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

	if (!pool)
		return VK_ERROR_OUT_OF_HOST_MEMORY;


	switch(pCreateInfo->queryType) {
	case VK_QUERY_TYPE_OCCLUSION:
		pool->stride = 16 * get_max_db(device);
		break;
	case VK_QUERY_TYPE_PIPELINE_STATISTICS:
		pool->stride = 16 * 11;
		break;
	case VK_QUERY_TYPE_TIMESTAMP:
		pool->stride = 8;
		break;
	default:
		unreachable("creating unhandled query type");
	}

	pool->type = pCreateInfo->queryType;
	pool->availability_offset = pool->stride * pCreateInfo->queryCount;
	size = pool->availability_offset + 4 * pCreateInfo->queryCount;

	pool->bo = device->ws->buffer_create(device->ws, size,
					     64, RADEON_DOMAIN_GTT, 0);

	if (!pool->bo) {
		vk_free2(&device->alloc, pAllocator, pool);
		return VK_ERROR_OUT_OF_DEVICE_MEMORY;
	}

	pool->ptr = device->ws->buffer_map(pool->bo);

	if (!pool->ptr) {
		device->ws->buffer_destroy(pool->bo);
		vk_free2(&device->alloc, pAllocator, pool);
		return VK_ERROR_OUT_OF_DEVICE_MEMORY;
	}
	memset(pool->ptr, 0, size);

	*pQueryPool = radv_query_pool_to_handle(pool);
	return VK_SUCCESS;
}

void radv_DestroyQueryPool(
	VkDevice                                    _device,
	VkQueryPool                                 _pool,
	const VkAllocationCallbacks*                pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_query_pool, pool, _pool);

	if (!pool)
		return;

	device->ws->buffer_destroy(pool->bo);
	vk_free2(&device->alloc, pAllocator, pool);
}

VkResult radv_GetQueryPoolResults(
	VkDevice                                    _device,
	VkQueryPool                                 queryPool,
	uint32_t                                    firstQuery,
	uint32_t                                    queryCount,
	size_t                                      dataSize,
	void*                                       pData,
	VkDeviceSize                                stride,
	VkQueryResultFlags                          flags)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_query_pool, pool, queryPool);
	char *data = pData;
	VkResult result = VK_SUCCESS;

	for(unsigned i = 0; i < queryCount; ++i, data += stride) {
		char *dest = data;
		unsigned query = firstQuery + i;
		char *src = pool->ptr + query * pool->stride;
		uint32_t available;

		switch (pool->type) {
		case VK_QUERY_TYPE_TIMESTAMP: {
			if (flags & VK_QUERY_RESULT_WAIT_BIT) {
				while(!*(volatile uint32_t*)(pool->ptr + pool->availability_offset + 4 * query))
					;
			}

			available = *(uint32_t*)(pool->ptr + pool->availability_offset + 4 * query);
			if (!available && !(flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
				result = VK_NOT_READY;
				break;

			}

			if (flags & VK_QUERY_RESULT_64_BIT) {
				*(uint64_t*)dest = *(uint64_t*)src;
				dest += 8;
			} else {
				*(uint32_t*)dest = *(uint32_t*)src;
				dest += 4;
			}
			break;
		}
		case VK_QUERY_TYPE_OCCLUSION: {
			volatile uint64_t const *src64 = (volatile uint64_t const *)src;
			uint64_t result = 0;
			int db_count = get_max_db(device);
			available = 1;

			for (int i = 0; i < db_count; ++i) {
				uint64_t start, end;
				do {
					start = src64[2 * i];
					end = src64[2 * i + 1];
				} while ((!(start & (1ull << 63)) || !(end & (1ull << 63))) && (flags & VK_QUERY_RESULT_WAIT_BIT));

				if (!(start & (1ull << 63)) || !(end & (1ull << 63)))
					available = 0;
				else {
					result += end - start;
				}
			}

			if (!available && !(flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
				result = VK_NOT_READY;
				break;

			}

			if (flags & VK_QUERY_RESULT_64_BIT) {
				*(uint64_t*)dest = result;
				dest += 8;
			} else {
				*(uint32_t*)dest = result;
				dest += 4;
			}
			break;
		default:
			unreachable("trying to get results of unhandled query type");
		}
		}

		if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
			if (flags & VK_QUERY_RESULT_64_BIT) {
				*(uint64_t*)dest = available;
			} else {
				*(uint32_t*)dest = available;
			}
		}
	}

	return result;
}

void radv_CmdCopyQueryPoolResults(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                stride,
    VkQueryResultFlags                          flags)
{
	RADV_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
	RADV_FROM_HANDLE(radv_query_pool, pool, queryPool);
	RADV_FROM_HANDLE(radv_buffer, dst_buffer, dstBuffer);
	struct radeon_winsys_cs *cs = cmd_buffer->cs;
	uint64_t va = cmd_buffer->device->ws->buffer_get_va(pool->bo);
	uint64_t dest_va = cmd_buffer->device->ws->buffer_get_va(dst_buffer->bo);
	dest_va += dst_buffer->offset + dstOffset;

	cmd_buffer->device->ws->cs_add_buffer(cmd_buffer->cs, pool->bo, 8);
	cmd_buffer->device->ws->cs_add_buffer(cmd_buffer->cs, dst_buffer->bo, 8);

	for(unsigned i = 0; i < queryCount; ++i, dest_va += stride) {
		unsigned query = firstQuery + i;
		uint64_t local_src_va = va  + query * pool->stride;
		unsigned elem_size = (flags & VK_QUERY_RESULT_64_BIT) ? 8 : 4;

		MAYBE_UNUSED unsigned cdw_max = radeon_check_space(cmd_buffer->device->ws, cs, 26);

		if (flags & VK_QUERY_RESULT_WAIT_BIT) {
			/* TODO, not sure if there is any case where we won't always be ready yet */
			uint64_t avail_va = va + pool->availability_offset + 4 * query;


			/* This waits on the ME. All copies below are done on the ME */
			radeon_emit(cs, PKT3(PKT3_WAIT_REG_MEM, 5, 0));
			radeon_emit(cs, WAIT_REG_MEM_EQUAL | WAIT_REG_MEM_MEM_SPACE(1));
			radeon_emit(cs, avail_va);
			radeon_emit(cs, avail_va >> 32);
			radeon_emit(cs, 1); /* reference value */
			radeon_emit(cs, 0xffffffff); /* mask */
			radeon_emit(cs, 4); /* poll interval */
		}

		switch (pool->type) {
		case VK_QUERY_TYPE_OCCLUSION:
			local_src_va += pool->stride - 16;

		case VK_QUERY_TYPE_TIMESTAMP:
			radeon_emit(cs, PKT3(PKT3_COPY_DATA, 4, 0));
			radeon_emit(cs, COPY_DATA_SRC_SEL(COPY_DATA_MEM) |
					COPY_DATA_DST_SEL(COPY_DATA_MEM) |
					((flags & VK_QUERY_RESULT_64_BIT) ? COPY_DATA_COUNT_SEL : 0));
			radeon_emit(cs, local_src_va);
			radeon_emit(cs, local_src_va >> 32);
			radeon_emit(cs, dest_va);
			radeon_emit(cs, dest_va >> 32);
			break;
		default:
			unreachable("trying to get results of unhandled query type");
		}

		/* The flag could be still changed while the data copy is busy and we
		 * then might have invalid data, but a ready flag. However, the availability
		 * writes happen on the ME too, so they should be synchronized. Might need to
		 * revisit this with multiple queues.
		 */
		if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
			uint64_t avail_va = va + pool->availability_offset + 4 * query;
			uint64_t avail_dest_va = dest_va;
			if (pool->type != VK_QUERY_TYPE_PIPELINE_STATISTICS)
				avail_dest_va += elem_size;
			else
				abort();

			radeon_emit(cs, PKT3(PKT3_COPY_DATA, 4, 0));
			radeon_emit(cs, COPY_DATA_SRC_SEL(COPY_DATA_MEM) |
					COPY_DATA_DST_SEL(COPY_DATA_MEM));
			radeon_emit(cs, avail_va);
			radeon_emit(cs, avail_va >> 32);
			radeon_emit(cs, avail_dest_va);
			radeon_emit(cs, avail_dest_va >> 32);
		}

		assert(cs->cdw <= cdw_max);
	}

}

void radv_CmdResetQueryPool(
	VkCommandBuffer                             commandBuffer,
	VkQueryPool                                 queryPool,
	uint32_t                                    firstQuery,
	uint32_t                                    queryCount)
{
	RADV_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
	RADV_FROM_HANDLE(radv_query_pool, pool, queryPool);
	uint64_t va = cmd_buffer->device->ws->buffer_get_va(pool->bo);

	cmd_buffer->device->ws->cs_add_buffer(cmd_buffer->cs, pool->bo, 8);

	si_cp_dma_clear_buffer(cmd_buffer, va + firstQuery * pool->stride,
			       queryCount * pool->stride, 0);
	si_cp_dma_clear_buffer(cmd_buffer, va + pool->availability_offset + firstQuery * 4,
			       queryCount * 4, 0);
}

void radv_CmdBeginQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    VkQueryControlFlags                         flags)
{
	RADV_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
	RADV_FROM_HANDLE(radv_query_pool, pool, queryPool);
	struct radeon_winsys_cs *cs = cmd_buffer->cs;
	uint64_t va = cmd_buffer->device->ws->buffer_get_va(pool->bo);
	va += pool->stride * query;

	cmd_buffer->device->ws->cs_add_buffer(cs, pool->bo, 8);

	switch (pool->type) {
	case VK_QUERY_TYPE_OCCLUSION:
		radeon_check_space(cmd_buffer->device->ws, cs, 7);

		++cmd_buffer->state.active_occlusion_queries;
		if (cmd_buffer->state.active_occlusion_queries == 1)
			radv_set_db_count_control(cmd_buffer);

		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 2, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_ZPASS_DONE) | EVENT_INDEX(1));
		radeon_emit(cs, va);
		radeon_emit(cs, va >> 32);
		break;
	default:
		unreachable("beginning unhandled query type");
	}
}


void radv_CmdEndQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
	RADV_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
	RADV_FROM_HANDLE(radv_query_pool, pool, queryPool);
	struct radeon_winsys_cs *cs = cmd_buffer->cs;
	uint64_t va = cmd_buffer->device->ws->buffer_get_va(pool->bo);
	va += pool->stride * query;

	cmd_buffer->device->ws->cs_add_buffer(cs, pool->bo, 8);

	switch (pool->type) {
	case VK_QUERY_TYPE_OCCLUSION:
		radeon_check_space(cmd_buffer->device->ws, cs, 14);

		cmd_buffer->state.active_occlusion_queries--;
		if (cmd_buffer->state.active_occlusion_queries == 0)
			radv_set_db_count_control(cmd_buffer);

		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 2, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_ZPASS_DONE) | EVENT_INDEX(1));
		radeon_emit(cs, va + 8);
		radeon_emit(cs, (va + 8) >> 32);

		/* hangs for VK_COMMAND_BUFFER_LEVEL_SECONDARY. */
		if (cmd_buffer->level == VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
			radeon_emit(cs, PKT3(PKT3_OCCLUSION_QUERY, 3, 0));
			radeon_emit(cs, va);
			radeon_emit(cs, va >> 32);
			radeon_emit(cs, va + pool->stride - 16);
			radeon_emit(cs, (va + pool->stride - 16) >> 32);
		}

		break;
	default:
		unreachable("ending unhandled query type");
	}
}

void radv_CmdWriteTimestamp(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlagBits                     pipelineStage,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
	RADV_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
	RADV_FROM_HANDLE(radv_query_pool, pool, queryPool);
	bool mec = radv_cmd_buffer_uses_mec(cmd_buffer);
	struct radeon_winsys_cs *cs = cmd_buffer->cs;
	uint64_t va = cmd_buffer->device->ws->buffer_get_va(pool->bo);
	uint64_t avail_va = va + pool->availability_offset + 4 * query;
	uint64_t query_va = va + pool->stride * query;

	cmd_buffer->device->ws->cs_add_buffer(cs, pool->bo, 5);

	MAYBE_UNUSED unsigned cdw_max = radeon_check_space(cmd_buffer->device->ws, cs, 12);

	if (mec) {
		radeon_emit(cs, PKT3(PKT3_RELEASE_MEM, 5, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_BOTTOM_OF_PIPE_TS) | EVENT_INDEX(5));
		radeon_emit(cs, 3 << 29);
		radeon_emit(cs, query_va);
		radeon_emit(cs, query_va >> 32);
		radeon_emit(cs, 0);
		radeon_emit(cs, 0);
	} else {
		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE_EOP, 4, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_BOTTOM_OF_PIPE_TS) | EVENT_INDEX(5));
		radeon_emit(cs, query_va);
		radeon_emit(cs, (3 << 29) | ((query_va >> 32) & 0xFFFF));
		radeon_emit(cs, 0);
		radeon_emit(cs, 0);
	}

	radeon_emit(cs, PKT3(PKT3_WRITE_DATA, 3, 0));
	radeon_emit(cs, S_370_DST_SEL(mec ? V_370_MEM_ASYNC : V_370_MEMORY_SYNC) |
		    S_370_WR_CONFIRM(1) |
		    S_370_ENGINE_SEL(V_370_ME));
	radeon_emit(cs, avail_va);
	radeon_emit(cs, avail_va >> 32);
	radeon_emit(cs, 1);

	assert(cmd_buffer->cs->cdw <= cdw_max);
}
