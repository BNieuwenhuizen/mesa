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

#ifndef RADV_RADEON_WINSYS_H
#define RADV_RADEON_WINSYS_H

#include "radv_winsys.h"
#include "addrlib/addrinterface.h"
#include "util/list.h"

struct radv_radeon_winsys {
	struct radeon_winsys base;
	int fd;

	struct radeon_info info;

	bool debug_all_bos;
	pthread_mutex_t global_bo_list_lock;
	struct list_head global_bo_list;
	unsigned num_buffers;

	bool use_ib_bos;

	uint32_t va_offset;
	mtx_t bo_va_mutex;
	struct list_head va_holes;

	ADDR_HANDLE addrlib;
	uint32_t rev_id;
	unsigned family;
	uint32_t gb_addr_config;
};

struct radv_radeon_bo {
	uint64_t address;
	uint64_t size;
	struct radv_radeon_winsys *ws;
	uint32_t handle;
	void *map_ptr;
	uint32_t domains;
};

static inline struct radv_radeon_winsys *
radv_radeon_winsys(struct radeon_winsys *base)
{
	return (struct radv_radeon_winsys*)base;
}

void
radv_radeon_bo_init_functions(struct radv_radeon_winsys *ws);
void
radv_radeon_cs_init_functions(struct radv_radeon_winsys *ws);
void
radv_radeon_surface_init_functions(struct radv_radeon_winsys *ws);

ADDR_HANDLE
radv_radeon_addr_create(struct radv_radeon_winsys *ws, int family, int rev_id,
                        enum chip_class chip_class);

#endif /* RADV_RADEON_WINSYS_H */
