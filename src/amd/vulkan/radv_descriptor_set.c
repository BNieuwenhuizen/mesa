/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
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

#include "radv_private.h"
#include "sid.h"

VkResult radv_CreateDescriptorSetLayout(
	VkDevice                                    _device,
	const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
	const VkAllocationCallbacks*                pAllocator,
	VkDescriptorSetLayout*                      pSetLayout)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	struct radv_descriptor_set_layout *set_layout;

	assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);

	uint32_t max_binding = 0;
	uint32_t immutable_sampler_count = 0;
	for (uint32_t j = 0; j < pCreateInfo->bindingCount; j++) {
		max_binding = MAX(max_binding, pCreateInfo->pBindings[j].binding);
		if (pCreateInfo->pBindings[j].pImmutableSamplers)
			immutable_sampler_count += pCreateInfo->pBindings[j].descriptorCount;
	}

	size_t size = sizeof(struct radv_descriptor_set_layout) +
		(max_binding + 1) * sizeof(set_layout->binding[0]) +
		immutable_sampler_count * sizeof(struct radv_sampler *);

	set_layout = radv_alloc2(&device->alloc, pAllocator, size, 8,
				 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (!set_layout)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	/* We just allocate all the samplers at the end of the struct */
	struct radv_sampler **samplers =
		(struct radv_sampler **)&set_layout->binding[max_binding + 1];

	set_layout->binding_count = max_binding + 1;
	set_layout->shader_stages = 0;
	set_layout->size = 0;

	memset(set_layout->binding, 0, size - sizeof(struct radv_descriptor_set_layout));

	uint32_t buffer_count = 0;
	uint32_t dynamic_offset_count = 0;

	for (uint32_t j = 0; j < pCreateInfo->bindingCount; j++) {
		const VkDescriptorSetLayoutBinding *binding = &pCreateInfo->pBindings[j];
		uint32_t b = binding->binding;
		uint32_t alignment;

		switch (binding->descriptorType) {
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
			set_layout->binding[b].dynamic_offset_count = 1;
			/* fall through */
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
			set_layout->binding[b].size = 16;
			set_layout->binding[b].buffer_count = 1;
			alignment = 16;
			break;
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
			/* main descriptor + fmask descriptor */
			set_layout->binding[b].size = 64;
			set_layout->binding[b].buffer_count = 1;
			alignment = 32;
			break;
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			/* main descriptor + gap + sampler */
			set_layout->binding[b].size = 64;
			set_layout->binding[b].buffer_count = 1;
			alignment = 32;
			break;
		default:
			break;
		}

		set_layout->size = align(set_layout->size, alignment);
		assert(binding->descriptorCount > 0);
		set_layout->binding[b].type = binding->descriptorType;
		set_layout->binding[b].array_size = binding->descriptorCount;
		set_layout->binding[b].offset = set_layout->size;
		set_layout->binding[b].buffer_offset = buffer_count;
		set_layout->binding[b].dynamic_offset_offset = dynamic_offset_count;

		set_layout->size += binding->descriptorCount * set_layout->binding[b].size;
		buffer_count += binding->descriptorCount * set_layout->binding[b].buffer_count;
		dynamic_offset_count += binding->descriptorCount *
			set_layout->binding[b].dynamic_offset_count;


		if (binding->pImmutableSamplers) {
			set_layout->binding[b].immutable_samplers = samplers;
			samplers += binding->descriptorCount;

			for (uint32_t i = 0; i < binding->descriptorCount; i++)
				set_layout->binding[b].immutable_samplers[i] =
					radv_sampler_from_handle(binding->pImmutableSamplers[i]);
		} else {
			set_layout->binding[b].immutable_samplers = NULL;
		}

		set_layout->shader_stages |= binding->stageFlags;
	}

	set_layout->buffer_count = buffer_count;
	set_layout->dynamic_offset_count = dynamic_offset_count;

	*pSetLayout = radv_descriptor_set_layout_to_handle(set_layout);

	return VK_SUCCESS;
}

void radv_DestroyDescriptorSetLayout(
	VkDevice                                    _device,
	VkDescriptorSetLayout                       _set_layout,
	const VkAllocationCallbacks*                pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_descriptor_set_layout, set_layout, _set_layout);

	radv_free2(&device->alloc, pAllocator, set_layout);
}

/*
 * Pipeline layouts.  These have nothing to do with the pipeline.  They are
 * just muttiple descriptor set layouts pasted together
 */

VkResult radv_CreatePipelineLayout(
	VkDevice                                    _device,
	const VkPipelineLayoutCreateInfo*           pCreateInfo,
	const VkAllocationCallbacks*                pAllocator,
	VkPipelineLayout*                           pPipelineLayout)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	struct radv_pipeline_layout *layout;

	assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);

	layout = radv_alloc2(&device->alloc, pAllocator, sizeof(*layout), 8,
			     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (layout == NULL)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	layout->num_sets = pCreateInfo->setLayoutCount;

	unsigned dynamic_offset_count = 0;

	memset(layout->stage, 0, sizeof(layout->stage));
	for (uint32_t set = 0; set < pCreateInfo->setLayoutCount; set++) {
		RADV_FROM_HANDLE(radv_descriptor_set_layout, set_layout,
				 pCreateInfo->pSetLayouts[set]);
		layout->set[set].layout = set_layout;

		layout->set[set].dynamic_offset_start = dynamic_offset_count;
		for (uint32_t b = 0; b < set_layout->binding_count; b++) {
			dynamic_offset_count += set_layout->binding[b].array_size;
			for (gl_shader_stage s = 0; s < MESA_SHADER_STAGES; s++) {
			}
		}
	}

	*pPipelineLayout = radv_pipeline_layout_to_handle(layout);

	return VK_SUCCESS;
}

void radv_DestroyPipelineLayout(
	VkDevice                                    _device,
	VkPipelineLayout                            _pipelineLayout,
	const VkAllocationCallbacks*                pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_pipeline_layout, pipeline_layout, _pipelineLayout);

	radv_free2(&device->alloc, pAllocator, pipeline_layout);
}

#define EMPTY 1

VkResult radv_CreateDescriptorPool(
	VkDevice                                    _device,
	const VkDescriptorPoolCreateInfo*           pCreateInfo,
	const VkAllocationCallbacks*                pAllocator,
	VkDescriptorPool*                           pDescriptorPool)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	struct radv_descriptor_pool *pool;
	int size = 0;
	pool = radv_alloc2(&device->alloc, pAllocator, size, 8,
			   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (!pool)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	*pDescriptorPool = radv_descriptor_pool_to_handle(pool);
	return VK_SUCCESS;
}

void radv_DestroyDescriptorPool(
	VkDevice                                    _device,
	VkDescriptorPool                            _pool,
	const VkAllocationCallbacks*                pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_descriptor_pool, pool, _pool);

	radv_free2(&device->alloc, pAllocator, pool);
}

VkResult radv_ResetDescriptorPool(
	VkDevice                                    _device,
	VkDescriptorPool                            descriptorPool,
	VkDescriptorPoolResetFlags                  flags)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_descriptor_pool, pool, descriptorPool);

	return VK_SUCCESS;
}

static VkResult
radv_descriptor_set_create(struct radv_device *device,
			   struct radv_descriptor_pool *pool,
			   const struct radv_descriptor_set_layout *layout,
			   struct radv_descriptor_set **out_set)
{
	struct radv_descriptor_set *set;
	unsigned mem_size = sizeof(struct radv_descriptor_set) +
		sizeof(struct radv_bo *) * layout->buffer_count;
	set = radv_alloc2(&device->alloc, NULL, mem_size, 8,
			  VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

	if (!set)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	memset(set, 0, mem_size);

	set->layout = layout;
	set->bo.bo = device->ws->buffer_create(device->ws, layout->size,
					       16, RADEON_DOMAIN_VRAM, 0);

	set->mapped_ptr = (uint32_t*)device->ws->buffer_map(set->bo.bo);
	*out_set = set;
	return VK_SUCCESS;
}

static void
radv_descriptor_set_destroy(struct radv_device *device,
			    struct radv_descriptor_pool *pool,
			    struct radv_descriptor_set *set)
{
	device->ws->buffer_destroy(set->bo.bo);
	radv_free2(&device->alloc, NULL, set);
}

VkResult radv_AllocateDescriptorSets(
	VkDevice                                    _device,
	const VkDescriptorSetAllocateInfo*          pAllocateInfo,
	VkDescriptorSet*                            pDescriptorSets)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_descriptor_pool, pool, pAllocateInfo->descriptorPool);

	VkResult result = VK_SUCCESS;
	uint32_t i;
	struct radv_descriptor_set *set;

	/* allocate a set of buffers for each shader to contain descriptors */
	for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
		RADV_FROM_HANDLE(radv_descriptor_set_layout, layout,
				 pAllocateInfo->pSetLayouts[i]);

		result = radv_descriptor_set_create(device, pool, layout, &set);
		if (result != VK_SUCCESS)
			break;

		pDescriptorSets[i] = radv_descriptor_set_to_handle(set);
	}

	if (result != VK_SUCCESS)
		radv_FreeDescriptorSets(_device, pAllocateInfo->descriptorPool,
					i, pDescriptorSets);
	return result;
}

VkResult radv_FreeDescriptorSets(
	VkDevice                                    _device,
	VkDescriptorPool                            descriptorPool,
	uint32_t                                    count,
	const VkDescriptorSet*                      pDescriptorSets)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_descriptor_pool, pool, descriptorPool);

	for (uint32_t i = 0; i < count; i++) {
		RADV_FROM_HANDLE(radv_descriptor_set, set, pDescriptorSets[i]);

		radv_descriptor_set_destroy(device, pool, set);
	}
	return VK_SUCCESS;
}


static void write_buffer_descriptor(struct radv_device *device,
                                    unsigned *dst,
                                    struct radv_bo **buffer_list,
                                    const VkDescriptorBufferInfo *buffer_info)
{
	RADV_FROM_HANDLE(radv_buffer, buffer, buffer_info->buffer);
	uint64_t va = device->ws->buffer_get_va(buffer->bo->bo);
	uint32_t range = buffer_info->range;

	if (buffer_info->range == VK_WHOLE_SIZE)
		range = buffer->size - buffer_info->offset;

	va += buffer_info->offset + buffer->offset;
	dst[0] = va;
	dst[1] = S_008F04_BASE_ADDRESS_HI(va >> 32);
	dst[2] = range;
	dst[3] = S_008F0C_DST_SEL_X(V_008F0C_SQ_SEL_X) |
		S_008F0C_DST_SEL_Y(V_008F0C_SQ_SEL_Y) |
		S_008F0C_DST_SEL_Z(V_008F0C_SQ_SEL_Z) |
		S_008F0C_DST_SEL_W(V_008F0C_SQ_SEL_W) |
		S_008F0C_NUM_FORMAT(V_008F0C_BUF_NUM_FORMAT_FLOAT) |
		S_008F0C_DATA_FORMAT(V_008F0C_BUF_DATA_FORMAT_32);

	*buffer_list = buffer->bo;
}

static void
write_image_descriptor(struct radv_device *device,
		       unsigned *dst,
		       struct radv_bo **buffer_list,
		       const VkDescriptorImageInfo *image_info)
{
	RADV_FROM_HANDLE(radv_image_view, iview, image_info->imageView);
	memcpy(dst, iview->descriptor, 8 * 4);
	memcpy(dst + 8, iview->fmask_descriptor, 8 * 4);
	*buffer_list = iview->bo;
}
static void
write_combined_image_sampler_descriptor(struct radv_device *device,
					unsigned *dst,
					struct radv_bo **buffer_list,
					const VkDescriptorImageInfo *image_info)
{
	RADV_FROM_HANDLE(radv_sampler, sampler, image_info->sampler);
	RADV_FROM_HANDLE(radv_image_view, iview, image_info->imageView);
	memcpy(dst, iview->descriptor, 8 * 4);
	/* copy over sampler state */
	memset(dst + 8, 0, 4 * 4);
	dst[11] = S_008F1C_DST_SEL_W(V_008F1C_SQ_SEL_1) |
	  S_008F1C_TYPE(V_008F1C_SQ_RSRC_IMG_1D);

	memcpy(dst + 12, sampler->state, 4 * 4);
	*buffer_list = iview->bo;
}

void radv_UpdateDescriptorSets(
	VkDevice                                    _device,
	uint32_t                                    descriptorWriteCount,
	const VkWriteDescriptorSet*                 pDescriptorWrites,
	uint32_t                                    descriptorCopyCount,
	const VkCopyDescriptorSet*                  pDescriptorCopies)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	uint32_t i, j;
	for (i = 0; i < descriptorWriteCount; i++) {
		const VkWriteDescriptorSet *writeset = &pDescriptorWrites[i];
		RADV_FROM_HANDLE(radv_descriptor_set, set, writeset->dstSet);
		const struct radv_descriptor_set_binding_layout *binding_layout =
			set->layout->binding + writeset->dstBinding;
		uint32_t *ptr = set->mapped_ptr;
		struct radv_bo **buffer_list =  set->descriptors;

		ptr += binding_layout->offset / 4;
		ptr += binding_layout->size * writeset->dstArrayElement / 4;
		buffer_list += binding_layout->buffer_offset;
		buffer_list += binding_layout->buffer_count * writeset->dstArrayElement;
		for (j = 0; j < writeset->descriptorCount; ++j) {
			switch(writeset->descriptorType) {
			case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
			case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
			case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
				write_buffer_descriptor(device, ptr, buffer_list,
							writeset->pBufferInfo + j);
				break;
			case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
				write_image_descriptor(device, ptr, buffer_list,
						       writeset->pImageInfo + j);
				break;
			case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
				write_combined_image_sampler_descriptor(device, ptr, buffer_list,
									writeset->pImageInfo + j);
				break;
			default:
				unreachable("unimplemented descriptor type");
				break;
			}
			ptr += binding_layout->size / 4;
			buffer_list += binding_layout->buffer_count;
		}

	}
	for (i = 0; i < descriptorCopyCount; i++) {
		const VkCopyDescriptorSet *copyset = &pDescriptorCopies[i];
	}
}
