/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
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

#include "radv_private.h"
#include "vk_format.h"
#include "radv_radeon_winsys.h"
#include "sid.h"
static unsigned
radv_choose_tiling(struct radv_device *Device,
		   const struct radv_image_create_info *create_info)
{
	const VkImageCreateInfo *pCreateInfo = create_info->vk_info;
	const struct vk_format_description *desc = vk_format_description(pCreateInfo->format);
	//   bool force_tiling = templ->flags & R600_RESOURCE_FLAG_FORCE_TILING;

	if (pCreateInfo->tiling == VK_IMAGE_TILING_LINEAR) {
		assert(pCreateInfo->samples <= 1);
		return RADEON_SURF_MODE_LINEAR_ALIGNED;
	}

	/* MSAA resources must be 2D tiled. */
	if (pCreateInfo->samples > 1)
		return RADEON_SURF_MODE_2D;

	return RADEON_SURF_MODE_2D;
}
static int
radv_init_surface(struct radv_device *device,
		  struct radeon_surf *surface,
		  const struct radv_image_create_info *create_info)
{
	const VkImageCreateInfo *pCreateInfo = create_info->vk_info;
	unsigned array_mode = radv_choose_tiling(device, create_info);
	const struct vk_format_description *desc =
		vk_format_description(pCreateInfo->format);
	bool is_depth, is_stencil;

	is_depth = vk_format_has_depth(desc);
	is_stencil = vk_format_has_stencil(desc);
	surface->npix_x = pCreateInfo->extent.width;
	surface->npix_y = pCreateInfo->extent.height;
	surface->npix_z = pCreateInfo->extent.depth;

	surface->blk_w = vk_format_get_blockwidth(pCreateInfo->format);
	surface->blk_h = vk_format_get_blockheight(pCreateInfo->format);
	surface->blk_d = 1;
	surface->array_size = pCreateInfo->arrayLayers;
	surface->last_level = pCreateInfo->mipLevels - 1;

	surface->bpe = vk_format_get_blocksize(pCreateInfo->format);
	/* align byte per element on dword */
	if (surface->bpe == 3) {
		surface->bpe = 4;
	}
	surface->nsamples = pCreateInfo->samples ? pCreateInfo->samples : 1;
	surface->flags = RADEON_SURF_SET(array_mode, MODE);

	switch (pCreateInfo->imageType){
	case VK_IMAGE_TYPE_1D:
		if (pCreateInfo->arrayLayers > 1)
			surface->flags |= RADEON_SURF_SET(RADEON_SURF_TYPE_1D_ARRAY, TYPE);
		else
			surface->flags |= RADEON_SURF_SET(RADEON_SURF_TYPE_1D, TYPE);
		break;
	case VK_IMAGE_TYPE_2D:
		if (pCreateInfo->arrayLayers > 1)
			surface->flags |= RADEON_SURF_SET(RADEON_SURF_TYPE_2D_ARRAY, TYPE);
		else
			surface->flags |= RADEON_SURF_SET(RADEON_SURF_TYPE_2D, TYPE);
		break;
	case VK_IMAGE_TYPE_3D:
		surface->flags |= RADEON_SURF_SET(RADEON_SURF_TYPE_3D, TYPE);
		break;
	}

	if (is_depth) {
		surface->flags |= RADEON_SURF_ZBUFFER;
		if (is_stencil)
			surface->flags |= RADEON_SURF_SBUFFER |
				RADEON_SURF_HAS_SBUFFER_MIPTREE;
	}

	surface->flags |= RADEON_SURF_HAS_TILE_MODE_INDEX;

	surface->flags |= RADEON_SURF_DISABLE_DCC;
	if (create_info->scanout)
		surface->flags |= RADEON_SURF_SCANOUT;
	return 0;
}
#define ATI_VENDOR_ID 0x1002
static uint32_t si_get_bo_metadata_word1(struct radv_device *device)
{
	return (ATI_VENDOR_ID << 16) | device->instance->physicalDevice.rad_info.pci_id;
}

static inline unsigned
si_tile_mode_index(const struct radv_image *image, unsigned level, bool stencil)
{
	if (stencil)
		return image->surface.stencil_tiling_index[level];
	else
		return image->surface.tiling_index[level];
}

static unsigned radv_map_swizzle(unsigned swizzle)
{
	switch (swizzle) {
	case VK_SWIZZLE_Y:
		return V_008F0C_SQ_SEL_Y;
	case VK_SWIZZLE_Z:
		return V_008F0C_SQ_SEL_Z;
	case VK_SWIZZLE_W:
		return V_008F0C_SQ_SEL_W;
	case VK_SWIZZLE_0:
		return V_008F0C_SQ_SEL_0;
	case VK_SWIZZLE_1:
		return V_008F0C_SQ_SEL_1;
	default: /* VK_SWIZZLE_X */
		return V_008F0C_SQ_SEL_X;
	}
}

static void
radv_make_buffer_descriptor(struct radv_device *device,
			    struct radv_buffer *buffer,
			    VkFormat vk_format,
			    unsigned offset,
			    unsigned range,
			    uint32_t *state)
{
	const struct vk_format_description *desc;
	unsigned stride;
	uint64_t gpu_address = device->ws->buffer_get_va(buffer->bo->bo);
	uint64_t va = gpu_address + buffer->offset;
	unsigned num_format, data_format;
	unsigned num_records;
	int first_non_void;
	desc = vk_format_description(vk_format);
	first_non_void = vk_format_get_first_non_void_channel(vk_format);
	stride = desc->block.bits / 8;

	num_format = radv_translate_buffer_numformat(desc, first_non_void);
	data_format = radv_translate_buffer_dataformat(desc, first_non_void);

	va += offset;
	state[0] = va;
	state[1] = S_008F04_BASE_ADDRESS_HI(va >> 32) |
		S_008F04_STRIDE(stride);
	state[2] = range;
	state[3] = S_008F0C_DST_SEL_X(radv_map_swizzle(desc->swizzle[0])) |
		   S_008F0C_DST_SEL_Y(radv_map_swizzle(desc->swizzle[1])) |
		   S_008F0C_DST_SEL_Z(radv_map_swizzle(desc->swizzle[2])) |
		   S_008F0C_DST_SEL_W(radv_map_swizzle(desc->swizzle[3])) |
		   S_008F0C_NUM_FORMAT(num_format) |
		   S_008F0C_DATA_FORMAT(data_format);
}

static void
si_set_mutable_tex_desc_fields(struct radv_device *device,
			       struct radv_image *image,
			       const struct radeon_surf_level *base_level_info,
			       unsigned base_level, unsigned first_level,
			       unsigned block_width, bool is_stencil,
			       uint32_t *state)
{
	uint64_t gpu_address = device->ws->buffer_get_va(image->bo->bo);
	uint64_t va = gpu_address + base_level_info->offset + image->offset;
	unsigned pitch = base_level_info->nblk_x * block_width;

	state[1] &= C_008F14_BASE_ADDRESS_HI;
	state[3] &= C_008F1C_TILING_INDEX;
	state[4] &= C_008F20_PITCH;
	state[6] &= C_008F28_COMPRESSION_EN;

	assert(!(va & 255));

	state[0] = va >> 8;
	state[1] |= S_008F14_BASE_ADDRESS_HI(va >> 40);
	state[3] |= S_008F1C_TILING_INDEX(si_tile_mode_index(image, base_level,
							     is_stencil));
	state[4] |= S_008F20_PITCH(pitch - 1);

	if (image->dcc_offset && image->surface.level[first_level].dcc_enabled) {
		state[6] |= S_008F28_COMPRESSION_EN(1);
		state[7] = (gpu_address + 
			    image->dcc_offset +
			    base_level_info->dcc_offset) >> 8;
	}
}

static unsigned radv_tex_dim(VkImageType image_type, VkImageViewType view_type,
			     unsigned nr_samples)
{
	switch (view_type) {
	case VK_IMAGE_VIEW_TYPE_1D:
		return V_008F1C_SQ_RSRC_IMG_1D;
	case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
		return V_008F1C_SQ_RSRC_IMG_1D_ARRAY;
	case VK_IMAGE_VIEW_TYPE_2D:
		return nr_samples > 1 ? V_008F1C_SQ_RSRC_IMG_2D_MSAA :
		V_008F1C_SQ_RSRC_IMG_2D;
	case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
		return nr_samples > 1 ? V_008F1C_SQ_RSRC_IMG_2D_MSAA_ARRAY :
		V_008F1C_SQ_RSRC_IMG_2D_ARRAY;
	case VK_IMAGE_VIEW_TYPE_3D:
		return V_008F1C_SQ_RSRC_IMG_3D;
	case VK_IMAGE_VIEW_TYPE_CUBE:
	case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
		return V_008F1C_SQ_RSRC_IMG_CUBE;
	}
}
/**
 * Build the sampler view descriptor for a texture.
 */
static void
si_make_texture_descriptor(struct radv_device *device,
			   struct radv_image *image,
			   bool sampler,
			   VkImageViewType view_type,
			   VkFormat vk_format,
			   const unsigned char state_swizzle[4],
			   unsigned first_level, unsigned last_level,
			   unsigned first_layer, unsigned last_layer,
			   unsigned width, unsigned height, unsigned depth,
			   uint32_t *state,
			   uint32_t *fmask_state)
{
	const struct vk_format_description *desc;
	unsigned char swizzle[4];
	int first_non_void;
	unsigned num_format, data_format, type;
	uint64_t va;
	

	desc = vk_format_description(vk_format);

	if (desc->colorspace == VK_FORMAT_COLORSPACE_ZS) {
		const unsigned char swizzle_xxxx[4] = {0, 0, 0, 0};
		const unsigned char swizzle_yyyy[4] = {1, 1, 1, 1};

		switch (vk_format) {
		case VK_FORMAT_X8_D24_UNORM_PACK32:
		case VK_FORMAT_D24_UNORM_S8_UINT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			vk_format_compose_swizzles(swizzle_yyyy, state_swizzle, swizzle);
			break;
		default:
			vk_format_compose_swizzles(swizzle_xxxx, state_swizzle, swizzle);
		}
	} else {
		vk_format_compose_swizzles(desc->swizzle, state_swizzle, swizzle);
	}

	first_non_void = vk_format_get_first_non_void_channel(vk_format);

	num_format = radv_translate_tex_numformat(vk_format, desc, first_non_void);
	if (num_format == ~0) {
		num_format = 0;
	}

	data_format = radv_translate_tex_dataformat(vk_format, desc, first_non_void);
	if (data_format == ~0) {
		data_format = 0;
	}

#if 0
	if (!sampler &&
	    (res->target == PIPE_TEXTURE_CUBE ||
	     res->target == PIPE_TEXTURE_CUBE_ARRAY ||
	     res->target == PIPE_TEXTURE_3D)) {
		/* For the purpose of shader images, treat cube maps and 3D
		 * textures as 2D arrays. For 3D textures, the address
		 * calculations for mipmaps are different, so we rely on the
		 * caller to effectively disable mipmaps.
		 */
		type = V_008F1C_SQ_RSRC_IMG_2D_ARRAY;

		assert(res->target != PIPE_TEXTURE_3D || (first_level == 0 && last_level == 0));
	} else {
		type = radv_tex_dim(res->target, target, res->nr_samples);
	}
#endif
	type = radv_tex_dim(image->type, view_type, image->samples);
	if (type == V_008F1C_SQ_RSRC_IMG_1D_ARRAY) {
	        height = 1;
		depth = image->array_size;
	} else if (type == V_008F1C_SQ_RSRC_IMG_2D_ARRAY ||
		   type == V_008F1C_SQ_RSRC_IMG_2D_MSAA_ARRAY) {
		depth = image->array_size;
	} else if (type == V_008F1C_SQ_RSRC_IMG_CUBE)
		depth = image->array_size / 6;

	state[0] = 0;
	state[1] = (S_008F14_DATA_FORMAT(data_format) |
		    S_008F14_NUM_FORMAT(num_format));
	state[2] = (S_008F18_WIDTH(width - 1) |
		    S_008F18_HEIGHT(height - 1));
	state[3] = (S_008F1C_DST_SEL_X(radv_map_swizzle(swizzle[0])) |
		    S_008F1C_DST_SEL_Y(radv_map_swizzle(swizzle[1])) |
		    S_008F1C_DST_SEL_Z(radv_map_swizzle(swizzle[2])) |
		    S_008F1C_DST_SEL_W(radv_map_swizzle(swizzle[3])) |
		    S_008F1C_BASE_LEVEL(image->samples > 1 ?
					0 : first_level) |
		    S_008F1C_LAST_LEVEL(image->samples > 1 ?
					util_logbase2(image->samples) :
					last_level) |
		    S_008F1C_POW2_PAD(image->levels > 1) |
		    S_008F1C_TYPE(type));
	state[4] = S_008F20_DEPTH(depth - 1);
	state[5] = (S_008F24_BASE_ARRAY(first_layer) |
		    S_008F24_LAST_ARRAY(last_layer));
	state[6] = 0;
	state[7] = 0;

	if (image->dcc_offset) {
		unsigned swap = radv_translate_colorswap(vk_format, FALSE);

		state[6] = S_008F28_ALPHA_IS_ON_MSB(swap <= 1);
	} else {
		/* The last dword is unused by hw. The shader uses it to clear
		 * bits in the first dword of sampler state.
		 */
		if (device->instance->physicalDevice.rad_info.chip_class <= CIK && image->samples <= 1) {
			if (first_level == last_level)
				state[7] = C_008F30_MAX_ANISO_RATIO;
			else
				state[7] = 0xffffffff;
		}
	}

	if (fmask_state)
		memset(fmask_state, 0, (8 * 4));
#if 0
	/* Initialize the sampler view for FMASK. */
	if (tex->fmask.size) {
		uint32_t fmask_format;

		va = tex->resource.gpu_address + tex->fmask.offset;

		switch (res->nr_samples) {
		case 2:
			fmask_format = V_008F14_IMG_DATA_FORMAT_FMASK8_S2_F2;
			break;
		case 4:
			fmask_format = V_008F14_IMG_DATA_FORMAT_FMASK8_S4_F4;
			break;
		case 8:
			fmask_format = V_008F14_IMG_DATA_FORMAT_FMASK32_S8_F8;
			break;
		default:
			assert(0);
			fmask_format = V_008F14_IMG_DATA_FORMAT_INVALID;
		}

		fmask_state[0] = va >> 8;
		fmask_state[1] = S_008F14_BASE_ADDRESS_HI(va >> 40) |
			S_008F14_DATA_FORMAT(fmask_format) |
			S_008F14_NUM_FORMAT(V_008F14_IMG_NUM_FORMAT_UINT);
		fmask_state[2] = S_008F18_WIDTH(width - 1) |
			S_008F18_HEIGHT(height - 1);
		fmask_state[3] = S_008F1C_DST_SEL_X(V_008F1C_SQ_SEL_X) |
			S_008F1C_DST_SEL_Y(V_008F1C_SQ_SEL_X) |
			S_008F1C_DST_SEL_Z(V_008F1C_SQ_SEL_X) |
			S_008F1C_DST_SEL_W(V_008F1C_SQ_SEL_X) |
			S_008F1C_TILING_INDEX(tex->fmask.tile_mode_index) |
			S_008F1C_TYPE(radv_tex_dim(image->type, 0, 0));
		fmask_state[4] = S_008F20_DEPTH(depth - 1) |
			S_008F20_PITCH(tex->fmask.pitch_in_pixels - 1);
		fmask_state[5] = S_008F24_BASE_ARRAY(first_layer) |
			S_008F24_LAST_ARRAY(last_layer);
		fmask_state[6] = 0;
		fmask_state[7] = 0;
	}
#endif
}

static void
radv_query_opaque_metadata(struct radv_device *device,
			   struct radv_image *image,
			   struct radeon_bo_metadata *md)
{
	static const unsigned char swizzle[] = {
		VK_SWIZZLE_X,
		VK_SWIZZLE_Y,
		VK_SWIZZLE_Z,
		VK_SWIZZLE_W
	};
	uint32_t desc[8], i;

	/* Metadata image format format version 1:
	 * [0] = 1 (metadata format identifier)
	 * [1] = (VENDOR_ID << 16) | PCI_ID
	 * [2:9] = image descriptor for the whole resource
	 *         [2] is always 0, because the base address is cleared
	 *         [9] is the DCC offset bits [39:8] from the beginning of
	 *             the buffer
	 * [10:10+LAST_LEVEL] = mipmap level offset bits [39:8] for each level
	 */
	md->metadata[0] = 1; /* metadata image format version 1 */

	/* TILE_MODE_INDEX is ambiguous without a PCI ID. */
	md->metadata[1] = si_get_bo_metadata_word1(device);


	si_make_texture_descriptor(device, image, true,
				   (VkImageViewType)image->type, image->vk_format,
				   swizzle, 0, image->levels - 1, 0,
				   0, //is_array ? image->array_size - 1 : 0,
				   image->extent.width, image->extent.height,
				   image->extent.depth,
				   desc, NULL);

	si_set_mutable_tex_desc_fields(device, image, &image->surface.level[0], 0, 0,
				       image->surface.blk_w, false, desc);

	/* Clear the base address and set the relative DCC offset. */
	desc[0] = 0;
	desc[1] &= C_008F14_BASE_ADDRESS_HI;
	desc[7] = image->dcc_offset >> 8;

	/* Dwords [2:9] contain the image descriptor. */
	memcpy(&md->metadata[2], desc, sizeof(desc));

	/* Dwords [10:..] contain the mipmap level offsets. */
	for (i = 0; i <= image->levels - 1; i++)
		md->metadata[10+i] = image->surface.level[i].offset >> 8;

	md->size_metadata = (11 + image->levels - 1) * 4;
}

void
radv_init_metadata(struct radv_device *device,
		   struct radv_image *image,
		   struct radeon_bo_metadata *metadata)
{
	struct radeon_surf *surface = &image->surface;

	memset(metadata, 0, sizeof(*metadata));
	metadata->microtile = surface->level[0].mode >= RADEON_SURF_MODE_1D ?
		RADEON_LAYOUT_TILED : RADEON_LAYOUT_LINEAR;
	metadata->macrotile = surface->level[0].mode >= RADEON_SURF_MODE_2D ?
		RADEON_LAYOUT_TILED : RADEON_LAYOUT_LINEAR;
	metadata->pipe_config = surface->pipe_config;
	metadata->bankw = surface->bankw;
	metadata->bankh = surface->bankh;
	metadata->tile_split = surface->tile_split;
	metadata->mtilea = surface->mtilea;
	metadata->num_banks = surface->num_banks;
	metadata->stride = surface->level[0].pitch_bytes;
	metadata->scanout = (surface->flags & RADEON_SURF_SCANOUT) != 0;

	radv_query_opaque_metadata(device, image, metadata);
}

VkResult
radv_image_create(VkDevice _device,
		  const struct radv_image_create_info *create_info,
		  const VkAllocationCallbacks* alloc,
		  VkImage *pImage)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	const VkImageCreateInfo *pCreateInfo = create_info->vk_info;
	struct radv_image *image = NULL;
	VkResult r;

	assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);

	radv_assert(pCreateInfo->mipLevels > 0);
	radv_assert(pCreateInfo->arrayLayers > 0);
	radv_assert(pCreateInfo->samples > 0);
	radv_assert(pCreateInfo->extent.width > 0);
	radv_assert(pCreateInfo->extent.height > 0);
	radv_assert(pCreateInfo->extent.depth > 0);

	image = radv_alloc2(&device->alloc, alloc, sizeof(*image), 8,
			    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (!image)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	memset(image, 0, sizeof(*image));
	image->type = pCreateInfo->imageType;
	image->extent = pCreateInfo->extent;
	image->vk_format = pCreateInfo->format;
	image->levels = pCreateInfo->mipLevels;
	image->array_size = pCreateInfo->arrayLayers;
	image->samples = pCreateInfo->samples;
	image->tiling = pCreateInfo->tiling;

	radv_init_surface(device, &image->surface, create_info);

	device->ws->surface_init(device->ws, &image->surface);
	image->size = image->surface.bo_size;
	image->alignment = image->surface.bo_alignment;

	if (create_info->stride && create_info->stride != image->surface.level[0].pitch_bytes) {
		image->surface.level[0].nblk_x = create_info->stride / image->surface.bpe;
		image->surface.level[0].pitch_bytes = create_info->stride;
		image->surface.level[0].slice_size = create_info->stride * image->surface.level[0].nblk_y;
	}
	*pImage = radv_image_to_handle(image);

	return VK_SUCCESS;

fail:
	if (image)
		radv_free2(&device->alloc, alloc, image);

	return r;
}

void
radv_image_view_init(struct radv_image_view *iview,
		     struct radv_device *device,
		     const VkImageViewCreateInfo* pCreateInfo,
		     struct radv_cmd_buffer *cmd_buffer,
		     VkImageUsageFlags usage_mask)
{
	RADV_FROM_HANDLE(radv_image, image, pCreateInfo->image);
	const VkImageSubresourceRange *range = &pCreateInfo->subresourceRange;
	switch (image->type) {
	default:
		unreachable("bad VkImageType");
	case VK_IMAGE_TYPE_1D:
	case VK_IMAGE_TYPE_2D:
		assert(range->baseArrayLayer + radv_get_layerCount(image, range) - 1 <= image->array_size);
		break;
	case VK_IMAGE_TYPE_3D:
		assert(range->baseArrayLayer + radv_get_layerCount(image, range) - 1
		       <= radv_minify(image->extent.depth, range->baseMipLevel));
		break;
	}
	iview->image = image;
	iview->bo = image->bo;
	iview->offset = image->offset;
	iview->type = pCreateInfo->viewType;
	iview->vk_format = pCreateInfo->format;
	iview->aspect_mask = pCreateInfo->subresourceRange.aspectMask;

	iview->extent = (VkExtent3D) {
		.width  = radv_minify(image->extent.width , range->baseMipLevel),
		.height = radv_minify(image->extent.height, range->baseMipLevel),
		.depth  = radv_minify(image->extent.depth , range->baseMipLevel),
	};
	iview->base_layer = range->baseArrayLayer;
	iview->base_mip = range->baseMipLevel;

	static const unsigned char swizzle[] = {
		VK_SWIZZLE_X,
		VK_SWIZZLE_Y,
		VK_SWIZZLE_Z,
		VK_SWIZZLE_W
	};
	si_make_texture_descriptor(device, image, false,
				   iview->type,
				   pCreateInfo->format,
				   swizzle,
				   range->baseMipLevel,
				   range->baseMipLevel + range->levelCount - 1,
				   range->baseArrayLayer,
				   range->baseArrayLayer + range->layerCount - 1,
				   iview->extent.width,
				   iview->extent.height,
				   iview->extent.depth,
				   iview->descriptor,
				   iview->fmask_descriptor);
	si_set_mutable_tex_desc_fields(device, image, &image->surface.level[0], 0, 0,
				       image->surface.blk_w, false, iview->descriptor);
}

VkResult
radv_CreateImage(VkDevice device,
		 const VkImageCreateInfo *pCreateInfo,
		 const VkAllocationCallbacks *pAllocator,
		 VkImage *pImage)
{
	return radv_image_create(device,
				 &(struct radv_image_create_info) {
					 .vk_info = pCreateInfo,
						 .scanout = false,
						 },
				 pAllocator,
				 pImage);
}

void
radv_DestroyImage(VkDevice _device, VkImage _image,
		  const VkAllocationCallbacks *pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);

	radv_free2(&device->alloc, pAllocator, radv_image_from_handle(_image));
}

void radv_GetImageSubresourceLayout(
	VkDevice                                    device,
	VkImage                                     _image,
	const VkImageSubresource*                   pSubresource,
	VkSubresourceLayout*                        pLayout)
{
	RADV_FROM_HANDLE(radv_image, image, _image);

	pLayout->rowPitch = image->surface.level[0].pitch_bytes;
	pLayout->size = image->surface.bo_size;
}


VkResult
radv_CreateImageView(VkDevice _device,
		     const VkImageViewCreateInfo *pCreateInfo,
		     const VkAllocationCallbacks *pAllocator,
		     VkImageView *pView)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	struct radv_image_view *view;

	view = radv_alloc2(&device->alloc, pAllocator, sizeof(*view), 8,
			   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (view == NULL)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	radv_image_view_init(view, device, pCreateInfo, NULL, ~0);

	*pView = radv_image_view_to_handle(view);

	return VK_SUCCESS;
}

void
radv_DestroyImageView(VkDevice _device, VkImageView _iview,
		      const VkAllocationCallbacks *pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_image_view, iview, _iview);


	radv_free2(&device->alloc, pAllocator, iview);
}

void radv_buffer_view_init(struct radv_buffer_view *view,
			   struct radv_device *device,
			   const VkBufferViewCreateInfo* pCreateInfo,
			   struct radv_cmd_buffer *cmd_buffer)
{
	RADV_FROM_HANDLE(radv_buffer, buffer, pCreateInfo->buffer);

	view->bo = buffer->bo;
	view->offset = buffer->offset + pCreateInfo->offset;
	view->range = pCreateInfo->range == VK_WHOLE_SIZE ?
		buffer->size - pCreateInfo->offset : pCreateInfo->range;
	view->vk_format = pCreateInfo->format;

	radv_make_buffer_descriptor(device, buffer, view->vk_format,
				    pCreateInfo->offset, view->range, view->state);
}

VkResult
radv_CreateBufferView(VkDevice _device,
		      const VkBufferViewCreateInfo *pCreateInfo,
		      const VkAllocationCallbacks *pAllocator,
		      VkBufferView *pView)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	struct radv_buffer_view *view;

	view = radv_alloc2(&device->alloc, pAllocator, sizeof(*view), 8,
			   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (!view)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	radv_buffer_view_init(view, device, pCreateInfo, NULL);

	*pView = radv_buffer_view_to_handle(view);

	return VK_SUCCESS;
}

void
radv_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
		       const VkAllocationCallbacks *pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_buffer_view, view, bufferView);

	radv_free2(&device->alloc, pAllocator, view);
}
