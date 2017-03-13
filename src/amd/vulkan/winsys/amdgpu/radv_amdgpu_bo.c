/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based on amdgpu winsys.
 * Copyright © 2011 Marek Olšák <maraeo@gmail.com>
 * Copyright © 2015 Advanced Micro Devices, Inc.
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

#include <stdio.h>

#include "radv_amdgpu_bo.h"

#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <inttypes.h>

static struct radeon_winsys_bo *
radv_amdgpu_winsys_bo_create(struct radeon_winsys *_ws,
			     uint64_t size,
			     unsigned alignment,
			     enum radeon_bo_heap heap);

static void *
radv_amdgpu_winsys_bo_map(struct radeon_winsys_bo *_bo);

static void
radv_amdgpu_winsys_bo_drm_deinit(struct radv_amdgpu_winsys_bo_drm *bo)
{
	if (bo->base.ws->debug_all_bos) {
		pthread_mutex_lock(&bo->base.ws->global_bo_list_lock);
		LIST_DEL(&bo->global_list_item);
		bo->base.ws->num_buffers--;
		pthread_mutex_unlock(&bo->base.ws->global_bo_list_lock);
	}
	amdgpu_bo_va_op(bo->base.bo, 0, bo->base.size, bo->base.va, 0, AMDGPU_VA_OP_UNMAP);
	amdgpu_va_range_free(bo->va_handle);
	amdgpu_bo_free(bo->base.bo);
}


static void
radv_amdgpu_winsys_bo_drm_destroy(struct radv_amdgpu_winsys_bo_drm *bo)
{
	radv_amdgpu_winsys_bo_drm_deinit(bo);
	FREE(bo);
}

static void
radv_amdgpu_winsys_bo_slab_entry_destroy(struct radv_amdgpu_winsys_bo_slab_entry *bo)
{
	mtx_lock(&bo->base.ws->slab_mtx);
	list_add(&bo->slab_entry_list, &bo->base.ws->slab_entries[bo->base.slab->heap][bo->base.slab->size_shift]);
	mtx_unlock(&bo->base.ws->slab_mtx);
}

static void radv_amdgpu_winsys_bo_destroy(struct radeon_winsys_bo *_bo)
{
	struct radv_amdgpu_winsys_bo *bo = radv_amdgpu_winsys_bo(_bo);

	if (bo->slab) {
		radv_amdgpu_winsys_bo_slab_entry_destroy((struct radv_amdgpu_winsys_bo_slab_entry*)bo);
	} else {
		radv_amdgpu_winsys_bo_drm_destroy((struct radv_amdgpu_winsys_bo_drm*)bo);
	}
}

static void radv_amdgpu_winsys_slab_destroy(struct radv_amdgpu_winsys_slab *slab)
{
	radv_amdgpu_winsys_bo_destroy((struct radeon_winsys_bo*)slab->base);
	free(slab);
}


static void radv_amdgpu_add_buffer_to_global_list(struct radv_amdgpu_winsys_bo_drm *bo)
{
	struct radv_amdgpu_winsys *ws = bo->base.ws;

	if (ws->debug_all_bos) {
		pthread_mutex_lock(&ws->global_bo_list_lock);
		LIST_ADDTAIL(&bo->global_list_item, &ws->global_bo_list);
		ws->num_buffers++;
		pthread_mutex_unlock(&ws->global_bo_list_lock);
	}
}

static int
radv_amdgpu_winsys_bo_drm_init(struct radv_amdgpu_winsys *ws,
                               uint64_t size,
                               unsigned alignment,
                               enum radeon_bo_heap heap,
                               struct radv_amdgpu_winsys_bo_drm *bo)
{
	struct amdgpu_bo_alloc_request request = {0};
	amdgpu_bo_handle buf_handle;
	uint64_t va = 0;
	amdgpu_va_handle va_handle;
	int r;

	request.alloc_size = size;
	request.phys_alignment = alignment;

	if (heap == RADEON_HEAP_VRAM || heap == RADEON_HEAP_VRAM)
		request.preferred_heap |= AMDGPU_GEM_DOMAIN_VRAM;
	else
		request.preferred_heap |= AMDGPU_GEM_DOMAIN_GTT;

	if (heap != RADEON_HEAP_VRAM)
		request.flags |= AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;
	else
		request.flags |= AMDGPU_GEM_CREATE_NO_CPU_ACCESS;
	if (heap == RADEON_HEAP_GTT_WC)
		request.flags |= AMDGPU_GEM_CREATE_CPU_GTT_USWC;

	r = amdgpu_bo_alloc(ws->dev, &request, &buf_handle);
	if (r) {
		fprintf(stderr, "amdgpu: Failed to allocate a buffer:\n");
		fprintf(stderr, "amdgpu:    size      : %"PRIu64" bytes\n", size);
		fprintf(stderr, "amdgpu:    alignment : %u bytes\n", alignment);
		fprintf(stderr, "amdgpu:    domains   : %u\n", request.preferred_heap);
		goto error_bo_alloc;
	}

	r = amdgpu_va_range_alloc(ws->dev, amdgpu_gpu_va_range_general,
				  size, alignment, 0, &va, &va_handle, 0);
	if (r)
		goto error_va_alloc;

	r = amdgpu_bo_va_op(buf_handle, 0, size, va, 0, AMDGPU_VA_OP_MAP);
	if (r)
		goto error_va_map;

	bo->base.bo = buf_handle;
	bo->base.va = va;
	bo->va_handle = va_handle;
	bo->base.size = size;
	bo->is_shared = false;
	bo->base.ws = ws;
	bo->base.slab = NULL;
	radv_amdgpu_add_buffer_to_global_list(bo);
	return 0;
error_va_map:
	amdgpu_va_range_free(va_handle);

error_va_alloc:
	amdgpu_bo_free(buf_handle);

error_bo_alloc:
	FREE(bo);
	return r;
}

static struct radeon_winsys_bo *
radv_amdgpu_winsys_bo_drm_create(struct radv_amdgpu_winsys *ws,
			     uint64_t size,
			     unsigned alignment,
			     enum radeon_bo_heap heap)
{
	struct radv_amdgpu_winsys_bo_drm *bo;
	int r;
	bo = CALLOC_STRUCT(radv_amdgpu_winsys_bo_drm);
	if (!bo) {
		return NULL;
	}

	r = radv_amdgpu_winsys_bo_drm_init(ws, size, alignment, heap, bo);
	if (r) {
		FREE(bo);
		return NULL;
	}
	return (struct radeon_winsys_bo *)bo;
}

static struct radv_amdgpu_winsys_slab *
radv_amdgpu_winsys_slab_create(struct radv_amdgpu_winsys *ws,
			     enum radeon_bo_heap heap,
			     int size_shift)
{
	struct radv_amdgpu_winsys_slab *slab;
	uint64_t elem_size = 1u << size_shift;
	int32_t elem_count;
	uint64_t size, alignment;

	if (elem_size <= 16384)
		size = 65536;
	else if (elem_size <= 256 * 1024)
		size = 1024 * 1024;
	else
		size = 4 * 1024 * 1024;

	elem_count = size / elem_size;
	alignment = MIN2(elem_size, 256 * 1024);

	slab = calloc(1, sizeof(struct radv_amdgpu_winsys_slab) +
	              sizeof(struct radv_amdgpu_winsys_bo_slab_entry) * elem_count);
	if (!slab) {
		return NULL;
	}

	slab->base = (struct radv_amdgpu_winsys_bo*)radv_amdgpu_winsys_bo_create(
	                                        (struct radeon_winsys*)ws, size,
	                                        alignment, heap);
	if (!slab->base) {
		free(slab);
		return NULL;
	}

	slab->heap = heap;
	slab->size_shift = size_shift;
	if (heap != RADEON_HEAP_VRAM) {
		slab->mapped_ptr = (char*)radv_amdgpu_winsys_bo_map((struct radeon_winsys_bo*)slab->base);
		if (!slab->mapped_ptr) {
			radv_amdgpu_winsys_bo_destroy((struct radeon_winsys_bo*)slab);
			free(slab);
			return NULL;
		}
	}

	mtx_lock(&ws->slab_mtx);
	list_add(&slab->slabs, &ws->slabs);
	for(int32_t i = 0; i < elem_count; ++i) {
		slab->entries[i].base.bo = slab->base->bo;
		slab->entries[i].base.va = slab->base->va + ((uint64_t)i << size_shift);
		slab->entries[i].base.size = 1ull << size_shift;
		slab->entries[i].base.ws = ws;
		slab->entries[i].base.slab = slab;
		list_add(&slab->entries[i].slab_entry_list, &ws->slab_entries[heap][size_shift]);
	}
	mtx_unlock(&ws->slab_mtx);

	return slab;
}

static struct radeon_winsys_bo *
radv_amdgpu_winsys_bo_slab_entry_create(struct radv_amdgpu_winsys *ws,
			     uint64_t size,
			     unsigned alignment,
			     enum radeon_bo_heap heap)
{
	int size_shift;

	size = MAX2(size, alignment);
	size_shift = 64 - __builtin_clzll(size);

	mtx_lock(&ws->slab_mtx);
	for (;;) {
		if (!list_empty(&ws->slab_entries[heap][size_shift])) {
			struct radv_amdgpu_winsys_bo_slab_entry *bo = list_first_entry((&ws->slab_entries[heap][size_shift]), struct radv_amdgpu_winsys_bo_slab_entry, slab_entry_list);
			list_del(&bo->slab_entry_list);
			mtx_unlock(&ws->slab_mtx);
			assert(bo->base.size >= size);
			return (struct radeon_winsys_bo *)bo;
		}
		mtx_unlock(&ws->slab_mtx);

		if(!radv_amdgpu_winsys_slab_create(ws, heap, size_shift)) {
			return NULL;
		}
		mtx_lock(&ws->slab_mtx);
	}
}

static struct radeon_winsys_bo *
radv_amdgpu_winsys_bo_create(struct radeon_winsys *_ws,
			     uint64_t size,
			     unsigned alignment,
			     enum radeon_bo_heap heap)
{
	struct radv_amdgpu_winsys *ws = radv_amdgpu_winsys(_ws);

	if (MAX2(size, alignment) <= 1024 * 1024) {
		return radv_amdgpu_winsys_bo_slab_entry_create(ws, size, alignment, heap);
	} else {
		return radv_amdgpu_winsys_bo_drm_create(ws, size, alignment, heap);
	}
}

static uint64_t radv_amdgpu_winsys_bo_get_va(struct radeon_winsys_bo *_bo)
{
	struct radv_amdgpu_winsys_bo *bo = radv_amdgpu_winsys_bo(_bo);
	return bo->va;
}

static void *
radv_amdgpu_winsys_bo_map(struct radeon_winsys_bo *_bo)
{
	struct radv_amdgpu_winsys_bo *bo = radv_amdgpu_winsys_bo(_bo);
	int ret;
	void *data;

	if (bo->slab)
		return bo->slab->mapped_ptr + (bo->va - bo->slab->base->va);

	ret = amdgpu_bo_cpu_map(bo->bo, &data);
	if (ret)
		return NULL;
	return data;
}

static void
radv_amdgpu_winsys_bo_unmap(struct radeon_winsys_bo *_bo)
{
	struct radv_amdgpu_winsys_bo *bo = radv_amdgpu_winsys_bo(_bo);
	if (!bo->slab)
		amdgpu_bo_cpu_unmap(bo->bo);
}

static struct radeon_winsys_bo *
radv_amdgpu_winsys_bo_from_fd(struct radeon_winsys *_ws,
			      int fd, unsigned *stride,
			      unsigned *offset)
{
	struct radv_amdgpu_winsys *ws = radv_amdgpu_winsys(_ws);
	struct radv_amdgpu_winsys_bo_drm *bo;
	uint64_t va;
	amdgpu_va_handle va_handle;
	enum amdgpu_bo_handle_type type = amdgpu_bo_handle_type_dma_buf_fd;
	struct amdgpu_bo_import_result result = {0};
	struct amdgpu_bo_info info = {0};
	enum radeon_bo_domain initial = 0;
	int r;
	bo = CALLOC_STRUCT(radv_amdgpu_winsys_bo_drm);
	if (!bo)
		return NULL;

	r = amdgpu_bo_import(ws->dev, type, fd, &result);
	if (r)
		goto error;

	r = amdgpu_bo_query_info(result.buf_handle, &info);
	if (r)
		goto error_query;

	r = amdgpu_va_range_alloc(ws->dev, amdgpu_gpu_va_range_general,
				  result.alloc_size, 1 << 20, 0, &va, &va_handle, 0);
	if (r)
		goto error_query;

	r = amdgpu_bo_va_op(result.buf_handle, 0, result.alloc_size, va, 0, AMDGPU_VA_OP_MAP);
	if (r)
		goto error_va_map;

	if (info.preferred_heap & AMDGPU_GEM_DOMAIN_VRAM)
		initial |= RADEON_DOMAIN_VRAM;
	if (info.preferred_heap & AMDGPU_GEM_DOMAIN_GTT)
		initial |= RADEON_DOMAIN_GTT;

	bo->base.bo = result.buf_handle;
	bo->base.va = va;
	bo->va_handle = va_handle;
	bo->base.size = result.alloc_size;
	bo->is_shared = true;
	bo->base.ws = ws;
	bo->base.slab = NULL;
	radv_amdgpu_add_buffer_to_global_list(bo);
	return (struct radeon_winsys_bo *)bo;
error_va_map:
	amdgpu_va_range_free(va_handle);

error_query:
	amdgpu_bo_free(result.buf_handle);

error:
	FREE(bo);
	return NULL;
}

static bool
radv_amdgpu_winsys_get_fd(struct radeon_winsys *_ws,
			  struct radeon_winsys_bo *_bo,
			  int *fd)
{
	struct radv_amdgpu_winsys_bo *bo = radv_amdgpu_winsys_bo(_bo);
	enum amdgpu_bo_handle_type type = amdgpu_bo_handle_type_dma_buf_fd;
	int r;
	unsigned handle;
	assert(!bo->slab);

	r = amdgpu_bo_export(bo->bo, type, &handle);
	if (r)
		return false;

	*fd = (int)handle;
	((struct radv_amdgpu_winsys_bo_drm *)bo)->is_shared = true;
	return true;
}

static unsigned radv_eg_tile_split_rev(unsigned eg_tile_split)
{
	switch (eg_tile_split) {
	case 64:    return 0;
	case 128:   return 1;
	case 256:   return 2;
	case 512:   return 3;
	default:
	case 1024:  return 4;
	case 2048:  return 5;
	case 4096:  return 6;
	}
}

static void
radv_amdgpu_winsys_bo_set_metadata(struct radeon_winsys_bo *_bo,
				   struct radeon_bo_metadata *md)
{
	struct radv_amdgpu_winsys_bo *bo = radv_amdgpu_winsys_bo(_bo);
	struct amdgpu_bo_metadata metadata = {0};
	uint32_t tiling_flags = 0;

	if (md->macrotile == RADEON_LAYOUT_TILED)
		tiling_flags |= AMDGPU_TILING_SET(ARRAY_MODE, 4); /* 2D_TILED_THIN1 */
	else if (md->microtile == RADEON_LAYOUT_TILED)
		tiling_flags |= AMDGPU_TILING_SET(ARRAY_MODE, 2); /* 1D_TILED_THIN1 */
	else
		tiling_flags |= AMDGPU_TILING_SET(ARRAY_MODE, 1); /* LINEAR_ALIGNED */

	tiling_flags |= AMDGPU_TILING_SET(PIPE_CONFIG, md->pipe_config);
	tiling_flags |= AMDGPU_TILING_SET(BANK_WIDTH, util_logbase2(md->bankw));
	tiling_flags |= AMDGPU_TILING_SET(BANK_HEIGHT, util_logbase2(md->bankh));
	if (md->tile_split)
		tiling_flags |= AMDGPU_TILING_SET(TILE_SPLIT, radv_eg_tile_split_rev(md->tile_split));
	tiling_flags |= AMDGPU_TILING_SET(MACRO_TILE_ASPECT, util_logbase2(md->mtilea));
	tiling_flags |= AMDGPU_TILING_SET(NUM_BANKS, util_logbase2(md->num_banks)-1);

	if (md->scanout)
		tiling_flags |= AMDGPU_TILING_SET(MICRO_TILE_MODE, 0); /* DISPLAY_MICRO_TILING */
	else
		tiling_flags |= AMDGPU_TILING_SET(MICRO_TILE_MODE, 1); /* THIN_MICRO_TILING */

	metadata.tiling_info = tiling_flags;
	metadata.size_metadata = md->size_metadata;
	memcpy(metadata.umd_metadata, md->metadata, sizeof(md->metadata));

	assert(!bo->slab);
	amdgpu_bo_set_metadata(bo->bo, &metadata);
}

void radv_amdgpu_winsys_free_slabs(struct radv_amdgpu_winsys *ws) {
	list_for_each_entry_safe(struct radv_amdgpu_winsys_slab, slab, &ws->slabs, slabs) {
		radv_amdgpu_winsys_slab_destroy(slab);
	}
}

void radv_amdgpu_bo_init_functions(struct radv_amdgpu_winsys *ws)
{
	ws->base.buffer_create = radv_amdgpu_winsys_bo_create;
	ws->base.buffer_destroy = radv_amdgpu_winsys_bo_destroy;
	ws->base.buffer_get_va = radv_amdgpu_winsys_bo_get_va;
	ws->base.buffer_map = radv_amdgpu_winsys_bo_map;
	ws->base.buffer_unmap = radv_amdgpu_winsys_bo_unmap;
	ws->base.buffer_from_fd = radv_amdgpu_winsys_bo_from_fd;
	ws->base.buffer_get_fd = radv_amdgpu_winsys_get_fd;
	ws->base.buffer_set_metadata = radv_amdgpu_winsys_bo_set_metadata;
}
