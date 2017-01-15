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
#include "radv_radeon_winsys_public.h"
#include <radeon_drm.h>
#include <radeon_surface.h>
#include <xf86drm.h>
#include "amdgpu_id.h"
#include "sid.h"

static void radv_radeon_winsys_query_info(struct radeon_winsys *rws,
                                     struct radeon_info *info)
{
	*info = radv_radeon_winsys(rws)->info;
}

static void radv_radeon_winsys_destroy(struct radeon_winsys *rws)
{
	struct radv_radeon_winsys *ws = radv_radeon_winsys(rws);

	//AddrDestroy(ws->addrlib);
	//amdgpu_device_deinitialize(ws->dev);
	close(ws->fd);
	FREE(ws);
}

static bool
radv_radeon_get_drm_value(int fd, unsigned request,
                          const char *errname, uint32_t *out)
{
	struct drm_radeon_info info;
	int retval;

	memset(&info, 0, sizeof(info));

	info.value = (unsigned long)out;
	info.request = request;

	retval = drmCommandWriteRead(fd, DRM_RADEON_INFO, &info, sizeof(info));
	if (retval) {
		if (errname) {
		fprintf(stderr, "radv winsys: Failed to get %s, error number %d\n",
			errname, retval);
		}
		return false;
	}
	return true;
}

static const char *
get_chip_name(enum radeon_family family)
{
	switch (family) {
	case CHIP_TAHITI: return "AMD RADV TAHITI (radeon)";
	case CHIP_PITCAIRN: return "AMD RADV PITCAIRN (radeon)";
	case CHIP_VERDE: return "AMD RADV CAPE VERDE (radeon)";
	case CHIP_OLAND: return "AMD RADV OLAND (radeon)";
	case CHIP_HAINAN: return "AMD RADV HAINAN (radeon)";
	case CHIP_BONAIRE: return "AMD RADV BONAIRE (radeon)";
	case CHIP_KAVERI: return "AMD RADV KAVERI (radeon)";
	case CHIP_KABINI: return "AMD RADV KABINI (radeon)";
	case CHIP_HAWAII: return "AMD RADV HAWAII (radeon)";
	case CHIP_MULLINS: return "AMD RADV MULLINS (radeon)";
	default: return "AMD RADV unknown (radeon)";
	}
}


static bool
do_winsys_init(struct radv_radeon_winsys *ws, int fd)
{
	struct drm_radeon_gem_info gem_info;
	drmVersionPtr version;
	drmDevicePtr devinfo;
	uint32_t tiling_config;
	int r;
	int i, j;

	memset(&gem_info, 0, sizeof(gem_info));

	version = drmGetVersion(ws->fd);
	if (version->version_major != 2 || version->version_minor < 12) {
		fprintf(stderr, "%s: DRM version is %d.%d.%d but this driver is "
                                "only compatible with 2.12.0 (kernel 3.2) or later.\n",
		        __FUNCTION__, version->version_major,
		        version->version_minor, version->version_patchlevel);
		drmFreeVersion(version);
		return false;
	}

	ws->info.drm_major = version->version_major;
	ws->info.drm_minor = version->version_minor;
	ws->info.drm_patchlevel = version->version_patchlevel;
	drmFreeVersion(version);

	if (!radv_radeon_get_drm_value(ws->fd, RADEON_INFO_DEVICE_ID, "PCI ID",
	                               &ws->info.pci_id))
		goto fail;

	/* Get PCI info. */
	r = drmGetDevice(fd, &devinfo);
	if (r) {
		fprintf(stderr, "radeon: drmGetDevice failed.\n");
		goto fail;
	}
	ws->info.pci_domain = devinfo->businfo.pci->domain;
	ws->info.pci_bus = devinfo->businfo.pci->bus;
	ws->info.pci_dev = devinfo->businfo.pci->dev;
	ws->info.pci_func = devinfo->businfo.pci->func;
	drmFreeDevice(&devinfo);

#if 0
	ws->info.vce_harvest_config = ws->amdinfo.vce_harvest_config;
#endif

	switch (ws->info.pci_id) {
#define CHIPSET(pci_id, name, cfamily) case pci_id: ws->info.family = CHIP_##cfamily; break;
#include "pci_ids/radeonsi_pci_ids.h"
#undef CHIPSET
	default:
		fprintf(stderr, "radeon: Invalid PCI ID.\n");
		goto fail;
	}

	if (ws->info.family >= CHIP_TONGA)
		goto fail;
	else if (ws->info.family >= CHIP_BONAIRE)
		ws->info.chip_class = CIK;
	else if (ws->info.family >= CHIP_TAHITI)
		ws->info.chip_class = SI;
	else
		goto fail;


	r = drmCommandWriteRead(ws->fd, DRM_RADEON_GEM_INFO, &gem_info, sizeof(gem_info));
	if (r) {
		fprintf(stderr, "radeon: Failed to get MM info, error number %d\n",
                        r);
		goto fail;
	}

	ws->info.name = get_chip_name(ws->info.family);
	ws->info.gart_size = gem_info.gart_size;
	ws->info.vram_size = gem_info.vram_size;
	ws->info.visible_vram_size = MIN2(ws->info.vram_size, 256 * 1024 * 1024);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_MAX_SCLK, NULL,
	                          &ws->info.max_shader_clock);
	ws->info.max_shader_clock /= 1000;

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_SI_BACKEND_ENABLED_MASK, NULL,
	                          &ws->info.enabled_rb_mask);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_NUM_BACKENDS, "num backends",
	                          &ws->info.num_render_backends);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_CLOCK_CRYSTAL_FREQ, NULL,
	                          &ws->info.clock_crystal_freq);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_NUM_TILE_PIPES, NULL,
	                          &ws->info.num_tile_pipes);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_ACTIVE_CU_COUNT, NULL,
	                          &ws->info.num_good_compute_units);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_MAX_SE, NULL,
	                          &ws->info.max_se);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_MAX_SH_PER_SE, NULL,
	                          &ws->info.max_sh_per_se);


	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_SI_TILE_MODE_ARRAY, NULL,
	                          ws->info.si_tile_mode_array);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_TILING_CONFIG, NULL,
	                          &tiling_config);

	radv_radeon_get_drm_value(ws->fd, RADEON_INFO_VA_START, NULL,
	                          &ws->va_offset);

	if (ws->info.chip_class == CIK) {
		radv_radeon_get_drm_value(ws->fd, RADEON_INFO_CIK_MACROTILE_MODE_ARRAY, NULL,
		                     ws->info.cik_macrotile_mode_array);
	}

	ws->info.pipe_interleave_bytes =  256 << ((tiling_config & 0xf00) >> 8);

        if (!ws->info.pipe_interleave_bytes)
            ws->info.pipe_interleave_bytes = 512;

	ws->gb_addr_config = S_0098F8_NUM_PIPES(ws->info.num_tile_pipes) |
	                     S_0098F8_PIPE_INTERLEAVE_SIZE((tiling_config & 0xf00) >> 8) |
	                     S_0098F8_ROW_SIZE((tiling_config >> 28) & 3);

	ws->info.has_uvd = 0;
	ws->info.vce_fw_version = 0;
	ws->info.has_userptr = TRUE;
	ws->info.has_virtual_memory = TRUE;
	ws->info.gart_page_size = sysconf(_SC_PAGESIZE);

	/* family and rev_id are for addrlib */
	switch (ws->info.family) {
	case CHIP_TAHITI:
		ws->family = FAMILY_SI;
		ws->rev_id = SI_TAHITI_P_A0;
		break;
	case CHIP_PITCAIRN:
		ws->family = FAMILY_SI;
		ws->rev_id = SI_PITCAIRN_PM_A0;
	  break;
	case CHIP_VERDE:
		ws->family = FAMILY_SI;
		ws->rev_id = SI_CAPEVERDE_M_A0;
		break;
	case CHIP_OLAND:
		ws->family = FAMILY_SI;
		ws->rev_id = SI_OLAND_M_A0;
		break;
	case CHIP_HAINAN:
		ws->family = FAMILY_SI;
		ws->rev_id = SI_HAINAN_V_A0;
		break;
	case CHIP_BONAIRE:
		ws->family = FAMILY_CI;
		ws->rev_id = CI_BONAIRE_M_A0;
		break;
	case CHIP_KAVERI:
		ws->family = FAMILY_KV;
		ws->rev_id = KV_SPECTRE_A0;
		break;
	case CHIP_KABINI:
		ws->family = FAMILY_KV;
		ws->rev_id = KB_KALINDI_A0;
		break;
	case CHIP_HAWAII:
		ws->family = FAMILY_CI;
		ws->rev_id = CI_HAWAII_P_A0;
		break;
	case CHIP_MULLINS:
		ws->family = FAMILY_KV;
		ws->rev_id = ML_GODAVARI_A0;
		break;
	case CHIP_TONGA:
		ws->family = FAMILY_VI;
		ws->rev_id = VI_TONGA_P_A0;
		break;
	case CHIP_ICELAND:
		ws->family = FAMILY_VI;
		ws->rev_id = VI_ICELAND_M_A0;
		break;
	case CHIP_CARRIZO:
		ws->family = FAMILY_CZ;
		ws->rev_id = CARRIZO_A0;
		break;
	case CHIP_STONEY:
		ws->family = FAMILY_CZ;
		ws->rev_id = STONEY_A0;
		break;
	case CHIP_FIJI:
		ws->family = FAMILY_VI;
		ws->rev_id = VI_FIJI_P_A0;
		break;
	case CHIP_POLARIS10:
		ws->family = FAMILY_VI;
		ws->rev_id = VI_POLARIS10_P_A0;
		break;
	case CHIP_POLARIS11:
		ws->family = FAMILY_VI;
		ws->rev_id = VI_POLARIS11_M_A0;
		break;
	default:
		fprintf(stderr, "amdgpu: Unknown family.\n");
		goto fail;
	}

	ws->addrlib = radv_radeon_addr_create(ws, ws->family, ws->rev_id, ws->info.chip_class);

	list_inithead(&ws->va_holes);

#if 0
	if (ws->info.chip_class == SI)
		ws->info.gfx_ib_pad_with_type2 = TRUE;

#endif
	return true;
fail:
	return false;
}

struct radeon_winsys *
radv_radeon_winsys_create(int fd)
{
	uint32_t r;
	struct radv_radeon_winsys *ws;

	ws = calloc(1, sizeof(struct radv_radeon_winsys));
	if (!ws)
		goto fail;

	ws->fd = dup(fd);
	if (!do_winsys_init(ws, fd))
		goto winsys_fail;

	ws->debug_all_bos = getenv("RADV_DEBUG_ALL_BOS") ? true : false;
	LIST_INITHEAD(&ws->global_bo_list);
	pthread_mutex_init(&ws->global_bo_list_lock, NULL);
	ws->base.query_info = radv_radeon_winsys_query_info;
	ws->base.destroy = radv_radeon_winsys_destroy;
	radv_radeon_bo_init_functions(ws);
	radv_radeon_cs_init_functions(ws);
	radv_radeon_surface_init_functions(ws);

	return &ws->base;

winsys_fail:
	free(ws);
fail:
	return NULL;
}
