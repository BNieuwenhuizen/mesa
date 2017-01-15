/*
 * Copyright © 2017 Google

 * based on amdgpu & radeon winsys.
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
#include "radv_radeon_winsys.h"

struct radv_radeon_ctx {
	struct radv_radeon_winsys *ws;
};

struct radv_radeon_cs {
	struct radeon_winsys_cs     base;
	struct radv_radeon_winsys   *ws;

	unsigned                    max_num_buffers;
	unsigned                    num_buffers;
	uint32_t                    *handles;
	uint8_t                     *priorities;

	bool                        failed;

	int                         buffer_hash_table[1024];
};

static struct radeon_winsys_ctx *
radv_radeon_ctx_create(struct radeon_winsys *ws)
{
	struct radv_radeon_ctx *ctx = malloc(sizeof(*ctx));

	return (struct radeon_winsys_ctx *)ctx;
}

static void
radv_radeon_ctx_destroy(struct radeon_winsys_ctx *ctx)
{
	free(ctx);
}

static bool
radv_radeon_ctx_wait_idle(struct radeon_winsys_ctx *ctx,
                          enum ring_type ring_type, int ring_index)
{
	return true;
}

static struct radeon_winsys_cs *
radv_radeon_winsys_cs_create(struct radeon_winsys *ws,
                             enum ring_type ring_type)
{
	struct radv_radeon_cs *cs = calloc(1, sizeof(struct radv_radeon_cs));

	if (!cs)
		return NULL;

	for (int i = 0; i < ARRAY_SIZE(cs->buffer_hash_table); ++i)
		cs->buffer_hash_table[i] = -1;

	cs->base.buf = malloc(16384);
	cs->base.max_dw = 4096;
	return (struct radeon_winsys_cs*)cs;
}

static void
radv_radeon_winsys_cs_destroy(struct radeon_winsys_cs *rcs)
{
	struct radv_radeon_cs *cs = (struct radv_radeon_cs *)rcs;

	free(cs->handles);
	free(cs->priorities);
	free(cs->base.buf);
	free(cs);
}

static void
radv_radeon_winsys_cs_reset(struct radeon_winsys_cs *rcs)
{
	struct radv_radeon_cs *cs = (struct radv_radeon_cs *)rcs;
	for (int i = 0; i < ARRAY_SIZE(cs->buffer_hash_table); ++i)
		cs->buffer_hash_table[i] = -1;

	cs->num_buffers = 0;
	cs->base.cdw = 0;
	cs->failed = false;
}

static bool
radv_radeon_winsys_cs_finalize(struct radeon_winsys_cs *rcs)
{
	struct radv_radeon_cs *cs = (struct radv_radeon_cs *)rcs;
	return !cs->failed;
}

static void
radv_radeon_winsys_cs_grow(struct radeon_winsys_cs * rcs, size_t min_size)
{
	struct radv_radeon_cs *cs = (struct radv_radeon_cs *)rcs;

	if (cs->failed) {
		cs->base.cdw = 0;
		return;
	}
}

static int
radv_radeon_winsys_cs_find_buffer(struct radv_radeon_cs *cs,
                                  uint32_t bo)
{
	unsigned hash = ((uintptr_t)bo >> 6) & (ARRAY_SIZE(cs->buffer_hash_table) - 1);
	int index = cs->buffer_hash_table[hash];

	if (index == -1)
		return -1;

	if (cs->handles[index] == bo)
		return index;

	for (unsigned i = 0; i < cs->num_buffers; ++i) {
		if (cs->handles[i] == bo) {
			cs->buffer_hash_table[hash] = i;
			return i;
		}
	}

	return -1;
}

static void
radv_radeon_winsys_cs_add_buffer_internal(struct radv_radeon_cs *cs,
                                          uint32_t bo,
                                          uint8_t priority)
{
	unsigned hash;
	int index = radv_radeon_winsys_cs_find_buffer(cs, bo);

	if (index != -1) {
		cs->priorities[index] = MAX2(cs->priorities[index], priority);
		return;
	}

	if (cs->num_buffers == cs->max_num_buffers) {
		unsigned new_count = MAX2(1, cs->max_num_buffers * 2);
		cs->handles = realloc(cs->handles, new_count * sizeof(uint32_t));
		cs->priorities = realloc(cs->priorities, new_count * sizeof(uint8_t));
		cs->max_num_buffers = new_count;
	}

	cs->handles[cs->num_buffers] = bo;
	cs->priorities[cs->num_buffers] = priority;

	hash = ((uintptr_t)bo >> 6) & (ARRAY_SIZE(cs->buffer_hash_table) - 1);
	cs->buffer_hash_table[hash] = cs->num_buffers;

	++cs->num_buffers;
}

static int
radv_radeon_winsys_cs_submit(struct radeon_winsys_ctx *ctx,
                             int queue_index,
                             struct radeon_winsys_cs **cs_array,
                             unsigned cs_count,
                             struct radeon_winsys_sem **wait_sem,
                             unsigned wait_sem_count,
                             struct radeon_winsys_sem **signal_sem,
                             unsigned signal_sem_count,
                             bool can_patch,
                             struct radeon_winsys_fence *fence)
{
	struct radv_radeon_cs *cs0 = (struct radv_radeon_cs *)cs_array[0];
	struct radv_radeon_winsys *ws = cs0->ws;
	uint32_t pad_word = 0xffff1000U;

	if (ws->info.chip_class == SI)
		pad_word = 0x80000000;

	for (unsigned i = 0; i < cs_count;) {
		unsigned cnt = 0, size = 0;
		struct drm_radeon_cs cs_args = {0};
		struct drm_radeon_cs_chunk ib_chunk, reloc_chunk, flag_chunk;
		struct drm_radeon_cs_chunk *chunk_array[3] = {&ibchunk, &reloc_chunk, &flag_chunk};
		uint32_t cs_flags[2] = {0};
		struct radeon_winsys_bo *bo = NULL;
		char *ptr;

		while (i + cnt < cs_count && 0xffff8 - size >= cs_array[i + cnt]->cdw) {
			size += cs_array[i + cnt]->cdw;
			++cnt;
		}

		assert(cnt);

		bo = ws->base.buffer_create(&ws->base, 4 * size, 4096, RADEON_DOMAIN_GTT, RADEON_FLAG_CPU_ACCESS);
		ptr = ws->base.buffer_map(bo);

		for (unsigned j = 0; j < cnt; ++j) {
			struct radv_radeon_cs *cs = (struct radv_radeon_cs *)cs_array[i + j];
			memcpy(ptr, cs->base.buf, 4 * cs->base.cdw);
			ptr += cs->base.cdw;
		}

		while(!size || (size & 7)) {
			*ptr++ = pad_word;
			++size;
		}

		cs.chunks = (uint64_t)(uintptr_t)chunk_array;
		cs.num_chunks = 3;



		ws->base.buffer_destroy(bo);
	}
	return 0;
}

static void
radv_radeon_winsys_cs_add_buffer(struct radeon_winsys_cs *rcs,
                                 struct radeon_winsys_bo *rbo,
                                 uint8_t priority)
{
	struct radv_radeon_cs *cs = (struct radv_radeon_cs *)rcs;
	struct radv_radeon_bo *bo = (struct radv_radeon_bo *)rbo;

	radv_radeon_winsys_cs_add_buffer_internal(cs, bo->handle, priority);
}

static void
radv_radeon_winsys_cs_execute_secondary(struct radeon_winsys_cs *rparent,
                                        struct radeon_winsys_cs *rchild)
{
	struct radv_radeon_cs *parent = (struct radv_radeon_cs *)rparent;
	struct radv_radeon_cs *child = (struct radv_radeon_cs *)rchild;

	for (unsigned i = 0; i < child->num_buffers; ++i) {
		radv_radeon_winsys_cs_add_buffer_internal(parent,
		                                          child->handles[i],
		                                          child->priorities[i]);
	}
}

static void
radv_radeon_winsys_cs_dump(struct radeon_winsys_cs *cs, FILE* file, uint32_t trace_id)
{
}


static struct radeon_winsys_fence *
radv_radeon_winsys_create_fence(struct radeon_winsys *rws)
{
	return rws->buffer_create(rws, 1, 1, RADEON_DOMAIN_GTT, 0);
}


static bool radv_radeon_winsys_fence_wait(struct radeon_winsys *_ws,
                                          struct radeon_winsys_fence *_fence,
                                          bool absolute,
                                          uint64_t timeout)
{
#if 0
	struct _fence *fence = (struct amdgpu_cs_fence *)_fence;
	unsigned flags = absolute ? AMDGPU_QUERY_FENCE_TIMEOUT_IS_ABSOLUTE : 0;
	int r;
	uint32_t expired = 0;

	/* Now use the libdrm query. */
	r = amdgpu_cs_query_fence_status(fence,
					 timeout,
					 flags,
					 &expired);

	if (r) {
		fprintf(stderr, "amdgpu: radv_amdgpu_cs_query_fence_status failed.\n");
		return false;
	}

	if (expired)
		return true;

#endif
	return true;
}

void
radv_radeon_cs_init_functions(struct radv_radeon_winsys *ws)
{
	ws->base.ctx_create = radv_radeon_ctx_create;
	ws->base.ctx_destroy = radv_radeon_ctx_destroy;
	ws->base.ctx_wait_idle = radv_radeon_ctx_wait_idle;
	ws->base.cs_create = radv_radeon_winsys_cs_create;
	ws->base.cs_destroy = radv_radeon_winsys_cs_destroy;
	ws->base.cs_reset = radv_radeon_winsys_cs_reset;
	ws->base.cs_finalize = radv_radeon_winsys_cs_finalize;
	ws->base.cs_grow = radv_radeon_winsys_cs_grow;
	ws->base.cs_submit = radv_radeon_winsys_cs_submit;
	ws->base.cs_add_buffer = radv_radeon_winsys_cs_add_buffer;
	ws->base.cs_execute_secondary = radv_radeon_winsys_cs_execute_secondary;
	ws->base.cs_dump = radv_radeon_winsys_cs_dump;
	ws->base.create_fence = radv_radeon_winsys_create_fence;
	ws->base.destroy_fence = ws->base.buffer_destroy;
	ws->base.fence_wait = radv_radeon_winsys_fence_wait;
}
