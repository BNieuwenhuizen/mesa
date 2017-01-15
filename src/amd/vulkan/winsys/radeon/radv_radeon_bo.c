#include "radv_radeon_winsys.h"

#include <radeon_drm.h>
#include <sys/mman.h>
#include <xf86drm.h>

struct radeon_bo_va_hole {
    struct list_head list;
    uint64_t         offset;
    uint64_t         size;
};

static uint64_t radeon_bomgr_find_va(struct radv_radeon_winsys *ws,
                                     uint64_t size, uint64_t alignment)
{
    struct radeon_bo_va_hole *hole, *n;
    uint64_t offset = 0, waste = 0;

    /* All VM address space holes will implicitly start aligned to the
     * size alignment, so we don't need to sanitize the alignment here
     */
    size = align(size, ws->info.gart_page_size);

    mtx_lock(&ws->bo_va_mutex);
    /* first look for a hole */
    LIST_FOR_EACH_ENTRY_SAFE(hole, n, &ws->va_holes, list) {
        offset = hole->offset;
        waste = offset % alignment;
        waste = waste ? alignment - waste : 0;
        offset += waste;
        if (offset >= (hole->offset + hole->size)) {
            continue;
        }
        if (!waste && hole->size == size) {
            offset = hole->offset;
            list_del(&hole->list);
            FREE(hole);
            mtx_unlock(&ws->bo_va_mutex);
            return offset;
        }
        if ((hole->size - waste) > size) {
            if (waste) {
                n = CALLOC_STRUCT(radeon_bo_va_hole);
                n->size = waste;
                n->offset = hole->offset;
                list_add(&n->list, &hole->list);
            }
            hole->size -= (size + waste);
            hole->offset += size + waste;
            mtx_unlock(&ws->bo_va_mutex);
            return offset;
        }
        if ((hole->size - waste) == size) {
            hole->size = waste;
            mtx_unlock(&ws->bo_va_mutex);
            return offset;
        }
    }

    offset = ws->va_offset;
    waste = offset % alignment;
    waste = waste ? alignment - waste : 0;
    if (waste) {
        n = CALLOC_STRUCT(radeon_bo_va_hole);
        n->size = waste;
        n->offset = offset;
        list_add(&n->list, &ws->va_holes);
    }
    offset += waste;
    ws->va_offset += size + waste;
    mtx_unlock(&ws->bo_va_mutex);
    return offset;
}

static void radeon_bomgr_free_va(struct radv_radeon_winsys *ws,
                                 uint64_t va, uint64_t size)
{
    struct radeon_bo_va_hole *hole;

    size = align(size, ws->info.gart_page_size);

    mtx_lock(&ws->bo_va_mutex);
    if ((va + size) == ws->va_offset) {
        ws->va_offset = va;
        /* Delete uppermost hole if it reaches the new top */
        if (!LIST_IS_EMPTY(&ws->va_holes)) {
            hole = container_of(ws->va_holes.next, hole, list);
            if ((hole->offset + hole->size) == va) {
                ws->va_offset = hole->offset;
                list_del(&hole->list);
                FREE(hole);
            }
        }
    } else {
        struct radeon_bo_va_hole *next;

        hole = container_of(&ws->va_holes, hole, list);
        LIST_FOR_EACH_ENTRY(next, &ws->va_holes, list) {
	    if (next->offset < va)
	        break;
            hole = next;
        }

        if (&hole->list != &ws->va_holes) {
            /* Grow upper hole if it's adjacent */
            if (hole->offset == (va + size)) {
                hole->offset = va;
                hole->size += size;
                /* Merge lower hole if it's adjacent */
                if (next != hole && &next->list != &ws->va_holes &&
                    (next->offset + next->size) == va) {
                    next->size += hole->size;
                    list_del(&hole->list);
                    FREE(hole);
                }
                goto out;
            }
        }

        /* Grow lower hole if it's adjacent */
        if (next != hole && &next->list != &ws->va_holes &&
            (next->offset + next->size) == va) {
            next->size += size;
            goto out;
        }

        /* FIXME on allocation failure we just lose virtual address space
         * maybe print a warning
         */
        next = CALLOC_STRUCT(radeon_bo_va_hole);
        if (next) {
            next->size = size;
            next->offset = va;
            list_add(&next->list, &hole->list);
        }
    }
out:
    mtx_unlock(&ws->bo_va_mutex);
}

static void
radv_radeon_winsys_buffer_destroy(struct radeon_winsys_bo *rbo)
{
	struct radv_radeon_bo *bo = (struct radv_radeon_bo *)rbo;
	struct drm_gem_close close_args;
	struct drm_radeon_gem_va va_args;
	int r;

	va_args.handle = bo->handle;
	va_args.vm_id = 0;
	va_args.operation = RADEON_VA_UNMAP;
	va_args.flags = RADEON_VM_PAGE_READABLE |
	                RADEON_VM_PAGE_WRITEABLE |
	                RADEON_VM_PAGE_SNOOPED;
	va_args.offset = bo->address;

	r = drmCommandWriteRead(bo->ws->fd, DRM_RADEON_GEM_VA, &va_args, sizeof(va_args));
	if (r && va_args.operation == RADEON_VA_RESULT_ERROR) {
		fprintf(stderr, "radeon: failed to deallocate buffer virtual memory.\n");
	}

	radeon_bomgr_free_va(bo->ws, bo->address, bo->size);

	close_args.handle = bo->handle;
	drmIoctl(bo->ws->fd, DRM_IOCTL_GEM_CLOSE, &close_args);

	free(bo);
}

static struct radeon_winsys_bo *
radv_radeon_winsys_buffer_create(struct radeon_winsys *rws,
                                 uint64_t size,
                                 unsigned alignment,
                                 enum radeon_bo_domain domain,
                                 enum radeon_bo_flag flags)
{
	struct radv_radeon_winsys *ws = radv_radeon_winsys(rws);
	struct radv_radeon_bo *bo = calloc(1, sizeof(struct radv_radeon_bo));
	struct drm_radeon_gem_create create_args;
	struct drm_radeon_gem_va va_args;
	int r;

	if (!bo)
		goto fail;

	size = (size + 65535) & ~65535ull;
	alignment = (alignment + 65535) & ~65535ull;
	create_args.size = size;
	create_args.alignment = alignment;
	create_args.initial_domain = domain;
	create_args.flags = 0;

	if (flags & RADEON_FLAG_GTT_WC)
		create_args.flags |= RADEON_GEM_GTT_WC;
	if (flags & RADEON_FLAG_CPU_ACCESS)
		create_args.flags |= RADEON_GEM_CPU_ACCESS;
	if (flags & RADEON_FLAG_NO_CPU_ACCESS)
		create_args.flags |= RADEON_GEM_NO_CPU_ACCESS;

	if (r = drmCommandWriteRead(ws->fd, DRM_RADEON_GEM_CREATE,
	                        &create_args, sizeof(create_args))) {
		fprintf(stderr, "radeon: Failed to allocate a buffer with error %d:\n", r);
		fprintf(stderr, "radeon:    size      : %" PRIu64 " bytes\n", size);
		fprintf(stderr, "radeon:    alignment : %u bytes\n", alignment);
		fprintf(stderr, "radeon:    domains   : %u\n", create_args.initial_domain);
		fprintf(stderr, "radeon:    flags     : %u\n", create_args.flags);
		goto fail;
	}

	bo->handle = create_args.handle;
	bo->size = size;
	bo->address =  radeon_bomgr_find_va(ws, size, alignment);
	bo->ws = ws;
	bo->domains = domain;

	va_args.handle = bo->handle;
	va_args.vm_id = 0;
	va_args.operation = RADEON_VA_MAP;
	va_args.flags = RADEON_VM_PAGE_READABLE |
	                RADEON_VM_PAGE_WRITEABLE |
	                RADEON_VM_PAGE_SNOOPED;
	va_args.offset = bo->address;

	r = drmCommandWriteRead(ws->fd, DRM_RADEON_GEM_VA, &va_args, sizeof(va_args));
	if (r && va_args.operation == RADEON_VA_RESULT_ERROR) {
		fprintf(stderr, "radeon: Failed to allocate virtual address for buffer:\n");
		fprintf(stderr, "radeon:    size      : %" PRIu64 " bytes\n", size);
		fprintf(stderr, "radeon:    alignment : %d bytes\n", alignment);
		fprintf(stderr, "radeon:    domains   : %d\n", create_args.initial_domain);
		fprintf(stderr, "radeon:    va        : 0x%016llx\n", (unsigned long long)bo->address);
		//radeon_bo_destroy(&bo->base);
		goto fail;
	}

	return (struct radeon_winsys_bo *)bo;
fail:
	fprintf(stderr, "bo alloc failed\n");
	free(bo);
	return NULL;
}

static void *
radv_radeon_buffer_map(struct radeon_winsys_bo *rbo)
{
	struct radv_radeon_bo *bo = (struct radv_radeon_bo *)rbo;
	struct drm_radeon_gem_mmap mmap_args = {0};
	void *ptr;

	mmap_args.handle = bo->handle;
	mmap_args.offset = 0;
	mmap_args.size = (uint64_t)bo->size;
	if (drmCommandWriteRead(bo->ws->fd, DRM_RADEON_GEM_MMAP, &mmap_args,
                                sizeof(mmap_args))) {
		fprintf(stderr, "radeon: gem_mmap failed: %p 0x%08X\n",
		        bo, bo->handle);
		return NULL;
	}

	ptr = mmap(0, mmap_args.size, PROT_READ|PROT_WRITE, MAP_SHARED,
	           bo->ws->fd, mmap_args.addr_ptr);
	if (ptr == MAP_FAILED)
		return NULL;

	bo->map_ptr = ptr;
	return ptr;
}

static void
radv_radeon_buffer_unmap(struct radeon_winsys_bo *rbo)
{
	struct radv_radeon_bo *bo = (struct radv_radeon_bo *)rbo;
	munmap(bo->map_ptr, bo->size);
	bo->map_ptr = NULL;
}

static uint64_t radv_radeon_winsys_bo_get_va(struct radeon_winsys_bo *rbo)
{
	struct radv_radeon_bo *bo = (struct radv_radeon_bo *)rbo;
	return bo->address;
}

void
radv_radeon_bo_init_functions(struct radv_radeon_winsys *ws)
{
	ws->base.buffer_create = radv_radeon_winsys_buffer_create;
	ws->base.buffer_destroy = radv_radeon_winsys_buffer_destroy;
	ws->base.buffer_map = radv_radeon_buffer_map;
	ws->base.buffer_unmap = radv_radeon_buffer_unmap;
	ws->base.buffer_get_va = radv_radeon_winsys_bo_get_va;
}
