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

#include "util/mesa-sha1.h"
#include "radv_private.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "spirv/nir_spirv.h"

#include <llvm-c/Core.h>
#include <llvm-c/TargetMachine.h>

#include "sid.h"
#include "ac_binary.h"
#include "ac_llvm_util.h"
#include "ac_nir_to_llvm.h"
#include "vk_format.h"

static void radv_shader_variant_destroy(struct radv_device *device,
                                        struct radv_shader_variant *variant);

static const struct nir_shader_compiler_options nir_options = {
	.vertex_id_zero_based = true,
	.lower_scmp = true,
	.lower_flrp32 = true,
	.lower_fsat = true,
	.lower_pack_snorm_2x16 = true,
	.lower_pack_snorm_4x8 = true,
	.lower_pack_unorm_2x16 = true,
	.lower_pack_unorm_4x8 = true,
	.lower_unpack_snorm_2x16 = true,
	.lower_unpack_snorm_4x8 = true,
	.lower_unpack_unorm_2x16 = true,
	.lower_unpack_unorm_4x8 = true,
	.lower_extract_byte = true,
	.lower_extract_word = true,
};

VkResult radv_CreateShaderModule(
	VkDevice                                    _device,
	const VkShaderModuleCreateInfo*             pCreateInfo,
	const VkAllocationCallbacks*                pAllocator,
	VkShaderModule*                             pShaderModule)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	struct radv_shader_module *module;

	assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
	assert(pCreateInfo->flags == 0);

	module = radv_alloc2(&device->alloc, pAllocator,
			     sizeof(*module) + pCreateInfo->codeSize, 8,
			     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (module == NULL)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	module->nir = NULL;
	module->size = pCreateInfo->codeSize;
	memcpy(module->data, pCreateInfo->pCode, module->size);

	_mesa_sha1_compute(module->data, module->size, module->sha1);

	*pShaderModule = radv_shader_module_to_handle(module);

	return VK_SUCCESS;
}

void radv_DestroyShaderModule(
	VkDevice                                    _device,
	VkShaderModule                              _module,
	const VkAllocationCallbacks*                pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_shader_module, module, _module);

	radv_free2(&device->alloc, pAllocator, module);
}

void radv_DestroyPipeline(
	VkDevice                                    _device,
	VkPipeline                                  _pipeline,
	const VkAllocationCallbacks*                pAllocator)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_pipeline, pipeline, _pipeline);

	if (!_pipeline)
		return;

	for (unsigned i = 0; i < MESA_SHADER_STAGES; ++i)
		if (pipeline->shaders[i])
			radv_shader_variant_destroy(device, pipeline->shaders[i]);

	radv_free2(&device->alloc, pAllocator, pipeline);
}


static void
radv_optimize_nir(struct nir_shader *shader)
{
        bool progress;

        do {
                progress = false;

                NIR_PASS_V(shader, nir_lower_vars_to_ssa);
                NIR_PASS_V(shader, nir_lower_alu_to_scalar);
                NIR_PASS_V(shader, nir_lower_phis_to_scalar);

                NIR_PASS(progress, shader, nir_copy_prop);
                NIR_PASS(progress, shader, nir_opt_remove_phis);
                NIR_PASS(progress, shader, nir_opt_dce);
                NIR_PASS(progress, shader, nir_opt_dead_cf);
                NIR_PASS(progress, shader, nir_opt_cse);
                NIR_PASS(progress, shader, nir_opt_peephole_select);
                NIR_PASS(progress, shader, nir_opt_algebraic);
                NIR_PASS(progress, shader, nir_opt_constant_folding);
                NIR_PASS(progress, shader, nir_opt_undef);
        } while (progress);
}

static nir_shader *
radv_shader_compile_to_nir(struct radv_device *device,
			   struct radv_shader_module *module,
			   const char *entrypoint_name,
			   gl_shader_stage stage,
			   const VkSpecializationInfo *spec_info,
			   bool dump)
{
	if (strcmp(entrypoint_name, "main") != 0) {
		radv_finishme("Multiple shaders per module not really supported");
	}

	nir_shader *nir;
	nir_function *entry_point;
	if (module->nir) {
		/* Some things such as our meta clear/blit code will give us a NIR
		 * shader directly.  In that case, we just ignore the SPIR-V entirely
		 * and just use the NIR shader */
		nir = module->nir;
		nir->options = &nir_options;
		nir_validate_shader(nir);

		assert(exec_list_length(&nir->functions) == 1);
		struct exec_node *node = exec_list_get_head(&nir->functions);
		entry_point = exec_node_data(nir_function, node, node);
	} else {
		uint32_t *spirv = (uint32_t *) module->data;
		assert(module->size % 4 == 0);

		uint32_t num_spec_entries = 0;
		struct nir_spirv_specialization *spec_entries = NULL;
		if (spec_info && spec_info->mapEntryCount > 0) {
			num_spec_entries = spec_info->mapEntryCount;
			spec_entries = malloc(num_spec_entries * sizeof(*spec_entries));
			for (uint32_t i = 0; i < num_spec_entries; i++) {
				VkSpecializationMapEntry entry = spec_info->pMapEntries[i];
				const void *data = spec_info->pData + entry.offset;
				assert(data + entry.size <= spec_info->pData + spec_info->dataSize);

				spec_entries[i].id = spec_info->pMapEntries[i].constantID;
				spec_entries[i].data = *(const uint32_t *)data;
			}
		}

		entry_point = spirv_to_nir(spirv, module->size / 4,
					   spec_entries, num_spec_entries,
					   stage, entrypoint_name, &nir_options);
		nir = entry_point->shader;
		assert(nir->stage == stage);
		nir_validate_shader(nir);

		free(spec_entries);


		if (stage == MESA_SHADER_FRAGMENT) {
			nir_lower_wpos_center(nir);
			nir_validate_shader(nir);
		}

		nir_lower_returns(nir);
		nir_validate_shader(nir);

		nir_inline_functions(nir);
		nir_validate_shader(nir);

		/* Pick off the single entrypoint that we want */
		foreach_list_typed_safe(nir_function, func, node, &nir->functions) {
			if (func != entry_point)
				exec_node_remove(&func->node);
		}
		assert(exec_list_length(&nir->functions) == 1);
		entry_point->name = ralloc_strdup(entry_point, "main");

		nir_remove_dead_variables(nir, nir_var_shader_in);
		nir_remove_dead_variables(nir, nir_var_shader_out);
		nir_remove_dead_variables(nir, nir_var_system_value);
		nir_validate_shader(nir);

		nir_lower_system_values(nir);
		nir_validate_shader(nir);
	}

	/* Vulkan uses the separate-shader linking model */
	nir->info.separate_shader = true;

	//   nir = brw_preprocess_nir(compiler, nir);

	nir_shader_gather_info(nir, entry_point->impl);

	nir_variable_mode indirect_mask = 0;
	//   if (compiler->glsl_compiler_options[stage].EmitNoIndirectInput)
	indirect_mask |= nir_var_shader_in;
	//   if (compiler->glsl_compiler_options[stage].EmitNoIndirectTemp)
	indirect_mask |= nir_var_local;

	nir_lower_indirect_derefs(nir, indirect_mask);
	nir_lower_vars_to_ssa(nir);
	nir_lower_var_copies(nir);
	nir_lower_global_vars_to_local(nir);
	nir_remove_dead_variables(nir, nir_var_local);
	radv_optimize_nir(nir);

	if (dump)
		nir_print_shader(nir, stderr);
   
	return nir;
}

static void radv_shader_variant_destroy(struct radv_device *device,
                                        struct radv_shader_variant *variant)
{
	device->ws->buffer_destroy(variant->bo);
	free(variant);
}

static
struct radv_shader_variant *radv_shader_variant_create(struct radv_device *device,
                                                       struct nir_shader *shader,
                                                       struct radv_pipeline_layout *layout,
                                                       union ac_shader_variant_key *key,
						       bool dump)
{
	struct radv_shader_variant *variant = calloc(1, sizeof(struct radv_shader_variant));
	enum radeon_family chip_family = device->instance->physicalDevice.rad_info.family;
	LLVMTargetMachineRef tm;
	if (!variant)
		return NULL;

	struct ac_nir_compiler_options options = {0};
	options.layout = layout;
	if (key)
		options.key = *key;

	struct ac_shader_binary binary;

	tm = ac_create_target_machine(chip_family);
	ac_compile_nir_shader(tm, &binary, &variant->config,
			      &variant->info, shader, &options, dump);
	LLVMDisposeTargetMachine(tm);

	bool scratch_enabled = variant->config.scratch_bytes_per_wave > 0;
	unsigned vgpr_comp_cnt = 0;
	switch (shader->stage) {
	case MESA_SHADER_VERTEX:
		variant->rsrc2 = S_00B12C_USER_SGPR(variant->info.num_user_sgprs) |
			S_00B12C_SCRATCH_EN(scratch_enabled);
		vgpr_comp_cnt = variant->info.vs.vgpr_comp_cnt;
		break;
	case MESA_SHADER_FRAGMENT:
		variant->rsrc2 = S_00B12C_USER_SGPR(variant->info.num_user_sgprs) |
			S_00B12C_SCRATCH_EN(scratch_enabled);
		break;
	case MESA_SHADER_COMPUTE:
		variant->rsrc2 = S_00B84C_USER_SGPR(variant->info.num_user_sgprs) |
			S_00B84C_SCRATCH_EN(scratch_enabled) |
			S_00B84C_TGID_X_EN(1) | S_00B84C_TGID_Y_EN(1) |
			S_00B84C_TGID_Z_EN(1) | S_00B84C_TIDIG_COMP_CNT(2) |
			S_00B84C_LDS_SIZE(variant->config.lds_size);
		break;
	}

	variant->rsrc1 =  S_00B848_VGPRS((variant->config.num_vgprs - 1) / 4) |
		S_00B848_SGPRS((variant->config.num_sgprs - 1) / 8) |
		S_00B128_VGPR_COMP_CNT(vgpr_comp_cnt) |
		S_00B848_DX10_CLAMP(1) |
		S_00B848_FLOAT_MODE(variant->config.float_mode);

	variant->bo = device->ws->buffer_create(device->ws, binary.code_size, 256,
						RADEON_DOMAIN_GTT, RADEON_FLAG_CPU_ACCESS);

	void *ptr = device->ws->buffer_map(variant->bo);
	memcpy(ptr, binary.code, binary.code_size);
	device->ws->buffer_unmap(variant->bo);

	free(binary.code);
	free(binary.config);
	free(binary.rodata);
	free(binary.global_symbol_offsets);
	free(binary.relocs);
	return variant;
}


static nir_shader *
radv_pipeline_compile(struct radv_pipeline *pipeline,
		      struct radv_shader_module *module,
		      const char *entrypoint,
		      gl_shader_stage stage,
		      const VkSpecializationInfo *spec_info,
		      bool dump)
{
	nir_shader *nir = radv_shader_compile_to_nir(pipeline->device,
						     module, entrypoint, stage,
						     spec_info, dump);
	if (nir == NULL)
		return NULL;

	return nir;
}

static uint32_t si_translate_blend_function(VkBlendOp op)
{
	switch (op) {
	case VK_BLEND_OP_ADD:
		return V_028780_COMB_DST_PLUS_SRC;
	case VK_BLEND_OP_SUBTRACT:
		return V_028780_COMB_SRC_MINUS_DST;
	case VK_BLEND_OP_REVERSE_SUBTRACT:
		return V_028780_COMB_DST_MINUS_SRC;
	case VK_BLEND_OP_MIN:
		return V_028780_COMB_MIN_DST_SRC;
	case VK_BLEND_OP_MAX:
		return V_028780_COMB_MAX_DST_SRC;
	default:
		return 0;
	}
}

static uint32_t si_translate_blend_factor(VkBlendFactor factor)
{
	switch (factor) {
	case VK_BLEND_FACTOR_ZERO:
		return V_028780_BLEND_ZERO;
	case VK_BLEND_FACTOR_ONE:
		return V_028780_BLEND_ONE;
	case VK_BLEND_FACTOR_SRC_COLOR:
		return V_028780_BLEND_SRC_COLOR;
	case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
		return V_028780_BLEND_ONE_MINUS_SRC_COLOR;
	case VK_BLEND_FACTOR_DST_COLOR:
		return V_028780_BLEND_DST_COLOR;
	case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR:
		return V_028780_BLEND_ONE_MINUS_DST_COLOR;
	case VK_BLEND_FACTOR_SRC_ALPHA:
		return V_028780_BLEND_SRC_ALPHA;
	case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
		return V_028780_BLEND_ONE_MINUS_SRC_ALPHA;
	case VK_BLEND_FACTOR_DST_ALPHA:
		return V_028780_BLEND_DST_ALPHA;
	case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
		return V_028780_BLEND_ONE_MINUS_DST_ALPHA;
	case VK_BLEND_FACTOR_CONSTANT_COLOR:
		return V_028780_BLEND_CONSTANT_COLOR;
	case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
		return V_028780_BLEND_ONE_MINUS_CONSTANT_COLOR;
	case VK_BLEND_FACTOR_CONSTANT_ALPHA:
		return V_028780_BLEND_CONSTANT_ALPHA;
	case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA:
		return V_028780_BLEND_ONE_MINUS_CONSTANT_ALPHA;
	case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE:
		return V_028780_BLEND_SRC_ALPHA_SATURATE;
	case VK_BLEND_FACTOR_SRC1_COLOR:
		return V_028780_BLEND_SRC1_COLOR;
	case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR:
		return V_028780_BLEND_INV_SRC1_COLOR;
	case VK_BLEND_FACTOR_SRC1_ALPHA:
		return V_028780_BLEND_SRC1_ALPHA;
	case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA:
		return V_028780_BLEND_INV_SRC1_ALPHA;
	default:
		return 0;
	}
}



static unsigned si_choose_spi_color_format(VkFormat vk_format,
					    bool blend_enable,
					    bool blend_need_alpha)
{
	const struct vk_format_description *desc = vk_format_description(vk_format);
	unsigned format, ntype, swap;

	/* Alpha is needed for alpha-to-coverage.
	 * Blending may be with or without alpha.
	 */
	unsigned normal = 0; /* most optimal, may not support blending or export alpha */
	unsigned alpha = 0; /* exports alpha, but may not support blending */
	unsigned blend = 0; /* supports blending, but may not export alpha */
	unsigned blend_alpha = 0; /* least optimal, supports blending and exports alpha */

	format = radv_translate_colorformat(vk_format);
	ntype = radv_translate_color_numformat(vk_format, desc,
					       vk_format_get_first_non_void_channel(vk_format));
	swap = radv_translate_colorswap(vk_format, false);

	/* Choose the SPI color formats. These are required values for Stoney/RB+.
	 * Other chips have multiple choices, though they are not necessarily better.
	 */
	switch (format) {
	case V_028C70_COLOR_5_6_5:
	case V_028C70_COLOR_1_5_5_5:
	case V_028C70_COLOR_5_5_5_1:
	case V_028C70_COLOR_4_4_4_4:
	case V_028C70_COLOR_10_11_11:
	case V_028C70_COLOR_11_11_10:
	case V_028C70_COLOR_8:
	case V_028C70_COLOR_8_8:
	case V_028C70_COLOR_8_8_8_8:
	case V_028C70_COLOR_10_10_10_2:
	case V_028C70_COLOR_2_10_10_10:
		if (ntype == V_028C70_NUMBER_UINT)
			alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_UINT16_ABGR;
		else if (ntype == V_028C70_NUMBER_SINT)
			alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_SINT16_ABGR;
		else
			alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_FP16_ABGR;
		break;

	case V_028C70_COLOR_16:
	case V_028C70_COLOR_16_16:
	case V_028C70_COLOR_16_16_16_16:
		if (ntype == V_028C70_NUMBER_UNORM ||
		    ntype == V_028C70_NUMBER_SNORM) {
			/* UNORM16 and SNORM16 don't support blending */
			if (ntype == V_028C70_NUMBER_UNORM)
				normal = alpha = V_028714_SPI_SHADER_UNORM16_ABGR;
			else
				normal = alpha = V_028714_SPI_SHADER_SNORM16_ABGR;

			/* Use 32 bits per channel for blending. */
			if (format == V_028C70_COLOR_16) {
				if (swap == V_028C70_SWAP_STD) { /* R */
					blend = V_028714_SPI_SHADER_32_R;
					blend_alpha = V_028714_SPI_SHADER_32_AR;
				} else if (swap == V_028C70_SWAP_ALT_REV) /* A */
					blend = blend_alpha = V_028714_SPI_SHADER_32_AR;
				else
					assert(0);
			} else if (format == V_028C70_COLOR_16_16) {
				if (swap == V_028C70_SWAP_STD) { /* RG */
					blend = V_028714_SPI_SHADER_32_GR;
					blend_alpha = V_028714_SPI_SHADER_32_ABGR;
				} else if (swap == V_028C70_SWAP_ALT) /* RA */
					blend = blend_alpha = V_028714_SPI_SHADER_32_AR;
				else
					assert(0);
			} else /* 16_16_16_16 */
				blend = blend_alpha = V_028714_SPI_SHADER_32_ABGR;
		} else if (ntype == V_028C70_NUMBER_UINT)
			alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_UINT16_ABGR;
		else if (ntype == V_028C70_NUMBER_SINT)
			alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_SINT16_ABGR;
		else if (ntype == V_028C70_NUMBER_FLOAT)
			alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_FP16_ABGR;
		else
			assert(0);
		break;

	case V_028C70_COLOR_32:
		if (swap == V_028C70_SWAP_STD) { /* R */
			blend = normal = V_028714_SPI_SHADER_32_R;
			alpha = blend_alpha = V_028714_SPI_SHADER_32_AR;
		} else if (swap == V_028C70_SWAP_ALT_REV) /* A */
			alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_32_AR;
		else
			assert(0);
		break;

	case V_028C70_COLOR_32_32:
		if (swap == V_028C70_SWAP_STD) { /* RG */
			blend = normal = V_028714_SPI_SHADER_32_GR;
			alpha = blend_alpha = V_028714_SPI_SHADER_32_ABGR;
		} else if (swap == V_028C70_SWAP_ALT) /* RA */
			alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_32_AR;
		else
			assert(0);
		break;

	case V_028C70_COLOR_32_32_32_32:
	case V_028C70_COLOR_8_24:
	case V_028C70_COLOR_24_8:
	case V_028C70_COLOR_X24_8_32_FLOAT:
		alpha = blend = blend_alpha = normal = V_028714_SPI_SHADER_32_ABGR;
		break;

	default:
		unreachable("unhandled blend format");
	}

	if (blend_enable && blend_need_alpha)
		return blend_alpha;
	else if(blend_need_alpha)
		return alpha;
	else if(blend_enable)
		return blend;
	else
		return normal;
}

static unsigned si_get_cb_shader_mask(unsigned spi_shader_col_format)
{
	unsigned i, cb_shader_mask = 0;

	for (i = 0; i < 8; i++) {
		switch ((spi_shader_col_format >> (i * 4)) & 0xf) {
		case V_028714_SPI_SHADER_ZERO:
			break;
		case V_028714_SPI_SHADER_32_R:
			cb_shader_mask |= 0x1 << (i * 4);
			break;
		case V_028714_SPI_SHADER_32_GR:
			cb_shader_mask |= 0x3 << (i * 4);
			break;
		case V_028714_SPI_SHADER_32_AR:
			cb_shader_mask |= 0x9 << (i * 4);
			break;
		case V_028714_SPI_SHADER_FP16_ABGR:
		case V_028714_SPI_SHADER_UNORM16_ABGR:
		case V_028714_SPI_SHADER_SNORM16_ABGR:
		case V_028714_SPI_SHADER_UINT16_ABGR:
		case V_028714_SPI_SHADER_SINT16_ABGR:
		case V_028714_SPI_SHADER_32_ABGR:
			cb_shader_mask |= 0xf << (i * 4);
			break;
		default:
			assert(0);
		}
	}
	return cb_shader_mask;
}

static void
radv_pipeline_compute_spi_color_formats(struct radv_pipeline *pipeline,
					const VkGraphicsPipelineCreateInfo *pCreateInfo,
					uint32_t blend_enable,
					uint32_t blend_need_alpha)
{
	RADV_FROM_HANDLE(radv_render_pass, pass, pCreateInfo->renderPass);
	struct radv_subpass *subpass = pass->subpasses + pCreateInfo->subpass;
	struct radv_blend_state *blend = &pipeline->graphics.blend;
	unsigned col_format = 0;

	for (unsigned i = 0; i < subpass->color_count; ++i) {
		struct radv_render_pass_attachment *attachment;
		unsigned cf;

		attachment = pass->attachments + subpass->color_attachments[i];

		cf = si_choose_spi_color_format(attachment->format,
						blend_enable & (1 << i),
						blend_need_alpha & (1 << i));

		col_format |= cf << (4 * i);
	}

	blend->cb_shader_mask = si_get_cb_shader_mask(col_format);

	if (!col_format)
		col_format |= V_028714_SPI_SHADER_32_R;
	blend->spi_shader_col_format = col_format;
}

static bool
format_is_int8(VkFormat format)
{
	const struct vk_format_description *desc = vk_format_description(format);
	int channel =  vk_format_get_first_non_void_channel(format);

	return channel >= 0 && desc->channel[channel].pure_integer &&
	       desc->channel[channel].size == 8;
}

unsigned radv_format_meta_fs_key(VkFormat format)
{
	unsigned col_format = si_choose_spi_color_format(format, false, false) - 1;
	bool is_int8 = format_is_int8(format);

	return col_format + (is_int8 ? 3 : 0);
}

static unsigned
radv_pipeline_compute_is_int8(const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
	RADV_FROM_HANDLE(radv_render_pass, pass, pCreateInfo->renderPass);
	struct radv_subpass *subpass = pass->subpasses + pCreateInfo->subpass;
	unsigned is_int8 = 0;

	for (unsigned i = 0; i < subpass->color_count; ++i) {
		struct radv_render_pass_attachment *attachment;

		attachment = pass->attachments + subpass->color_attachments[i];

		if (format_is_int8(attachment->format))
			is_int8 |= 1 << i;
	}

	return is_int8;
}

static void
radv_pipeline_init_blend_state(struct radv_pipeline *pipeline,
			       const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
	const VkPipelineColorBlendStateCreateInfo *vkblend = pCreateInfo->pColorBlendState;
	struct radv_blend_state *blend = &pipeline->graphics.blend;
	unsigned mode = V_028808_CB_NORMAL;
	uint32_t blend_enable = 0, blend_need_alpha = 0;
	int i;

	blend->cb_color_control = 0;
	if (vkblend->logicOpEnable)
		blend->cb_color_control |= S_028808_ROP3(vkblend->logicOp | (vkblend->logicOp << 4));
	else
		blend->cb_color_control |= S_028808_ROP3(0xcc);

	blend->cb_target_mask = 0;
	for (i = 0; i < vkblend->attachmentCount; i++) {
		const VkPipelineColorBlendAttachmentState *att = &vkblend->pAttachments[i];
		unsigned blend_cntl = 0;
		VkBlendOp eqRGB = att->colorBlendOp;
		VkBlendFactor srcRGB = att->srcColorBlendFactor;
		VkBlendFactor dstRGB = att->dstColorBlendFactor;
		VkBlendOp eqA = att->alphaBlendOp;
		VkBlendFactor srcA = att->srcAlphaBlendFactor;
		VkBlendFactor dstA = att->dstAlphaBlendFactor;

		blend->sx_mrt0_blend_opt[i] = S_028760_COLOR_COMB_FCN(V_028760_OPT_COMB_BLEND_DISABLED) | S_028760_ALPHA_COMB_FCN(V_028760_OPT_COMB_BLEND_DISABLED);

		if (!att->colorWriteMask)
			continue;

		blend->cb_target_mask |= (unsigned)att->colorWriteMask << (4 * i);
		if (!att->blendEnable) {
			blend->cb_blend_control[i] = blend_cntl;
			continue;
		}

		if (eqRGB == VK_BLEND_OP_MIN || eqRGB == VK_BLEND_OP_MAX) {
			srcRGB = VK_BLEND_FACTOR_ONE;
			dstRGB = VK_BLEND_FACTOR_ONE;
		}
		if (eqA == VK_BLEND_OP_MIN || eqA == VK_BLEND_OP_MAX) {
			srcA = VK_BLEND_FACTOR_ONE;
			dstA = VK_BLEND_FACTOR_ONE;
		}

		blend_cntl |= S_028780_ENABLE(1);

		blend_cntl |= S_028780_COLOR_COMB_FCN(si_translate_blend_function(eqRGB));
		blend_cntl |= S_028780_COLOR_SRCBLEND(si_translate_blend_factor(srcRGB));
		blend_cntl |= S_028780_COLOR_DESTBLEND(si_translate_blend_factor(dstRGB));
		if (srcA != srcRGB || dstA != dstRGB || eqA != eqRGB) {
			blend_cntl |= S_028780_SEPARATE_ALPHA_BLEND(1);
			blend_cntl |= S_028780_ALPHA_COMB_FCN(si_translate_blend_function(eqA));
			blend_cntl |= S_028780_ALPHA_SRCBLEND(si_translate_blend_factor(srcA));
			blend_cntl |= S_028780_ALPHA_DESTBLEND(si_translate_blend_factor(dstA));
		}
		blend->cb_blend_control[i] = blend_cntl;

		blend_enable |= 1 << i;

		if (srcRGB == VK_BLEND_FACTOR_SRC_ALPHA ||
		    dstRGB == VK_BLEND_FACTOR_SRC_ALPHA ||
		    srcRGB == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE ||
		    dstRGB == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE ||
		    srcRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA ||
		    dstRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
			blend_need_alpha |= 1 << i;
	}
	for (i = vkblend->attachmentCount; i < 8; i++)
		blend->cb_blend_control[i] = 0;

	if (blend->cb_target_mask)
		blend->cb_color_control |= S_028808_MODE(mode);
	else
		blend->cb_color_control |= S_028808_MODE(V_028808_CB_DISABLE);

	radv_pipeline_compute_spi_color_formats(pipeline, pCreateInfo,
						blend_enable, blend_need_alpha);
}

static uint32_t si_translate_stencil_op(enum VkStencilOp op)
{
	switch (op) {
	case VK_STENCIL_OP_KEEP:
		return V_02842C_STENCIL_KEEP;
	case VK_STENCIL_OP_ZERO:
		return V_02842C_STENCIL_ZERO;
	case VK_STENCIL_OP_REPLACE:
		return V_02842C_STENCIL_REPLACE_TEST;
	case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
		return V_02842C_STENCIL_ADD_CLAMP;
	case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
		return V_02842C_STENCIL_SUB_CLAMP;
	case VK_STENCIL_OP_INVERT:
		return V_02842C_STENCIL_INVERT;
	case VK_STENCIL_OP_INCREMENT_AND_WRAP:
		return V_02842C_STENCIL_ADD_WRAP;
	case VK_STENCIL_OP_DECREMENT_AND_WRAP:
		return V_02842C_STENCIL_SUB_WRAP;
	default:
		return 0;
	}
}
static void
radv_pipeline_init_depth_stencil_state(struct radv_pipeline *pipeline,
				       const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
	const VkPipelineDepthStencilStateCreateInfo *vkds = pCreateInfo->pDepthStencilState;
	struct radv_depth_stencil_state *ds = &pipeline->graphics.ds;

	memset(ds, 0, sizeof(*ds));
	if (!vkds)
		return;
	ds->db_depth_control = S_028800_Z_ENABLE(vkds->depthTestEnable ? 1 : 0) |
		S_028800_Z_WRITE_ENABLE(vkds->depthWriteEnable ? 1 : 0) |
		S_028800_ZFUNC(vkds->depthCompareOp) |
		S_028800_DEPTH_BOUNDS_ENABLE(vkds->depthBoundsTestEnable ? 1 : 0);

	if (vkds->stencilTestEnable) {
		ds->db_depth_control |= S_028800_STENCIL_ENABLE(1) | S_028800_BACKFACE_ENABLE(1);
		ds->db_depth_control |= S_028800_STENCILFUNC(vkds->front.compareOp);
		ds->db_stencil_control |= S_02842C_STENCILFAIL(si_translate_stencil_op(vkds->front.failOp));
		ds->db_stencil_control |= S_02842C_STENCILZPASS(si_translate_stencil_op(vkds->front.passOp));
		ds->db_stencil_control |= S_02842C_STENCILZFAIL(si_translate_stencil_op(vkds->front.depthFailOp));

		ds->db_depth_control |= S_028800_STENCILFUNC_BF(vkds->back.compareOp);
		ds->db_stencil_control |= S_02842C_STENCILFAIL_BF(si_translate_stencil_op(vkds->back.failOp));
		ds->db_stencil_control |= S_02842C_STENCILZPASS_BF(si_translate_stencil_op(vkds->back.passOp));
		ds->db_stencil_control |= S_02842C_STENCILZFAIL_BF(si_translate_stencil_op(vkds->back.depthFailOp));
	}

	ds->db_depth_bounds_min = fui(vkds->minDepthBounds);
	ds->db_depth_bounds_max = fui(vkds->maxDepthBounds);
}

static uint32_t si_translate_fill(VkPolygonMode func)
{
	switch(func) {
	case VK_POLYGON_MODE_FILL:
		return V_028814_X_DRAW_TRIANGLES;
	case VK_POLYGON_MODE_LINE:
		return V_028814_X_DRAW_LINES;
	case VK_POLYGON_MODE_POINT:
		return V_028814_X_DRAW_POINTS;
	default:
		assert(0);
		return V_028814_X_DRAW_POINTS;
	}
}
static void
radv_pipeline_init_raster_state(struct radv_pipeline *pipeline,
				const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
	const VkPipelineRasterizationStateCreateInfo *vkraster = pCreateInfo->pRasterizationState;
	struct radv_raster_state *raster = &pipeline->graphics.raster;

	memset(raster, 0, sizeof(*raster));

	raster->spi_interp_control =
		S_0286D4_FLAT_SHADE_ENA(1) |
		S_0286D4_PNT_SPRITE_ENA(1) |
		S_0286D4_PNT_SPRITE_OVRD_X(V_0286D4_SPI_PNT_SPRITE_SEL_S) |
		S_0286D4_PNT_SPRITE_OVRD_Y(V_0286D4_SPI_PNT_SPRITE_SEL_T) |
		S_0286D4_PNT_SPRITE_OVRD_Z(V_0286D4_SPI_PNT_SPRITE_SEL_0) |
		S_0286D4_PNT_SPRITE_OVRD_W(V_0286D4_SPI_PNT_SPRITE_SEL_1) |
		S_0286D4_PNT_SPRITE_TOP_1(0); // vulkan is top to bottom - 1.0 at bottom

	raster->pa_cl_vs_out_cntl = S_02881C_VS_OUT_MISC_SIDE_BUS_ENA(1);
	raster->pa_cl_clip_cntl = S_028810_PS_UCP_MODE(3) |
		S_028810_DX_CLIP_SPACE_DEF(1) | // vulkan uses DX conventions.
		S_028810_ZCLIP_NEAR_DISABLE(vkraster->depthClampEnable ? 1 : 0) |
		S_028810_ZCLIP_FAR_DISABLE(vkraster->depthClampEnable ? 1 : 0) |
		S_028810_DX_RASTERIZATION_KILL(vkraster->rasterizerDiscardEnable ? 1 : 0) |
		S_028810_DX_LINEAR_ATTR_CLIP_ENA(1);

	raster->pa_su_vtx_cntl =
		S_028BE4_PIX_CENTER(1) | // TODO verify
		S_028BE4_QUANT_MODE(V_028BE4_X_16_8_FIXED_POINT_1_256TH);

	raster->pa_sc_mode_cntl_0 = 0;
	raster->pa_su_sc_mode_cntl =
		S_028814_FACE(vkraster->frontFace) |
		S_028814_CULL_FRONT(!!(vkraster->cullMode & VK_CULL_MODE_FRONT_BIT)) |
		S_028814_CULL_BACK(!!(vkraster->cullMode & VK_CULL_MODE_BACK_BIT)) |
		S_028814_POLY_MODE(vkraster->polygonMode != VK_POLYGON_MODE_FILL) |
		S_028814_POLYMODE_FRONT_PTYPE(si_translate_fill(vkraster->polygonMode)) |
		S_028814_POLYMODE_BACK_PTYPE(si_translate_fill(vkraster->polygonMode));
}

static uint32_t
si_translate_prim(enum VkPrimitiveTopology topology)
{
	switch (topology) {
	case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
		return V_008958_DI_PT_POINTLIST;
	case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
		return V_008958_DI_PT_LINELIST;
	case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
		return V_008958_DI_PT_LINESTRIP;
	case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
		return V_008958_DI_PT_TRILIST;
	case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
		return V_008958_DI_PT_TRISTRIP;
	case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
		return V_008958_DI_PT_TRIFAN;
	case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
		return V_008958_DI_PT_LINELIST_ADJ;
	case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
		return V_008958_DI_PT_LINESTRIP_ADJ;
	case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
		return V_008958_DI_PT_TRILIST_ADJ;
	case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
		return V_008958_DI_PT_TRISTRIP_ADJ;
	case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
		return V_008958_DI_PT_PATCH;
	default:
		assert(0);
		return 0;
	}
}

static unsigned si_map_swizzle(unsigned swizzle)
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
radv_pipeline_init_dynamic_state(struct radv_pipeline *pipeline,
				 const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
	radv_cmd_dirty_mask_t states = RADV_CMD_DIRTY_DYNAMIC_ALL;
	RADV_FROM_HANDLE(radv_render_pass, pass, pCreateInfo->renderPass);
	struct radv_subpass *subpass = &pass->subpasses[pCreateInfo->subpass];

	pipeline->dynamic_state = default_dynamic_state;

	if (pCreateInfo->pDynamicState) {
		/* Remove all of the states that are marked as dynamic */
		uint32_t count = pCreateInfo->pDynamicState->dynamicStateCount;
		for (uint32_t s = 0; s < count; s++)
			states &= ~(1 << pCreateInfo->pDynamicState->pDynamicStates[s]);
	}

	struct radv_dynamic_state *dynamic = &pipeline->dynamic_state;

	dynamic->viewport.count = pCreateInfo->pViewportState->viewportCount;
	if (states & (1 << VK_DYNAMIC_STATE_VIEWPORT)) {
		typed_memcpy(dynamic->viewport.viewports,
			     pCreateInfo->pViewportState->pViewports,
			     pCreateInfo->pViewportState->viewportCount);
	}

	dynamic->scissor.count = pCreateInfo->pViewportState->scissorCount;
	if (states & (1 << VK_DYNAMIC_STATE_SCISSOR)) {
		typed_memcpy(dynamic->scissor.scissors,
			     pCreateInfo->pViewportState->pScissors,
			     pCreateInfo->pViewportState->scissorCount);
	}

	if (states & (1 << VK_DYNAMIC_STATE_LINE_WIDTH)) {
		assert(pCreateInfo->pRasterizationState);
		dynamic->line_width = pCreateInfo->pRasterizationState->lineWidth;
	}

	if (states & (1 << VK_DYNAMIC_STATE_DEPTH_BIAS)) {
		assert(pCreateInfo->pRasterizationState);
		dynamic->depth_bias.bias =
			pCreateInfo->pRasterizationState->depthBiasConstantFactor;
		dynamic->depth_bias.clamp =
			pCreateInfo->pRasterizationState->depthBiasClamp;
		dynamic->depth_bias.slope =
			pCreateInfo->pRasterizationState->depthBiasSlopeFactor;
	}

	if (states & (1 << VK_DYNAMIC_STATE_BLEND_CONSTANTS)) {
		assert(pCreateInfo->pColorBlendState);
		typed_memcpy(dynamic->blend_constants,
			     pCreateInfo->pColorBlendState->blendConstants, 4);
	}

	/* If there is no depthstencil attachment, then don't read
	 * pDepthStencilState. The Vulkan spec states that pDepthStencilState may
	 * be NULL in this case. Even if pDepthStencilState is non-NULL, there is
	 * no need to override the depthstencil defaults in
	 * radv_pipeline::dynamic_state when there is no depthstencil attachment.
	 *
	 * From the Vulkan spec (20 Oct 2015, git-aa308cb):
	 *
	 *    pDepthStencilState [...] may only be NULL if renderPass and subpass
	 *    specify a subpass that has no depth/stencil attachment.
	 */
	if (subpass->depth_stencil_attachment != VK_ATTACHMENT_UNUSED) {
		if (states & (1 << VK_DYNAMIC_STATE_DEPTH_BOUNDS)) {
			assert(pCreateInfo->pDepthStencilState);
			dynamic->depth_bounds.min =
				pCreateInfo->pDepthStencilState->minDepthBounds;
			dynamic->depth_bounds.max =
				pCreateInfo->pDepthStencilState->maxDepthBounds;
		}

		if (states & (1 << VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK)) {
			assert(pCreateInfo->pDepthStencilState);
			dynamic->stencil_compare_mask.front =
				pCreateInfo->pDepthStencilState->front.compareMask;
			dynamic->stencil_compare_mask.back =
				pCreateInfo->pDepthStencilState->back.compareMask;
		}

		if (states & (1 << VK_DYNAMIC_STATE_STENCIL_WRITE_MASK)) {
			assert(pCreateInfo->pDepthStencilState);
			dynamic->stencil_write_mask.front =
				pCreateInfo->pDepthStencilState->front.writeMask;
			dynamic->stencil_write_mask.back =
				pCreateInfo->pDepthStencilState->back.writeMask;
		}

		if (states & (1 << VK_DYNAMIC_STATE_STENCIL_REFERENCE)) {
			assert(pCreateInfo->pDepthStencilState);
			dynamic->stencil_reference.front =
				pCreateInfo->pDepthStencilState->front.reference;
			dynamic->stencil_reference.back =
				pCreateInfo->pDepthStencilState->back.reference;
		}
	}

	pipeline->dynamic_state_mask = states;
}

static union ac_shader_variant_key
radv_compute_vs_key(const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
	union ac_shader_variant_key key;
	const VkPipelineVertexInputStateCreateInfo *input_state =
	                                         pCreateInfo->pVertexInputState;

	memset(&key, 0, sizeof(key));
	key.vs.instance_rate_inputs = 0;

	for (unsigned i = 0; i < input_state->vertexAttributeDescriptionCount; ++i) {
		unsigned binding;
		binding = input_state->pVertexAttributeDescriptions[i].binding;
		if (input_state->pVertexBindingDescriptions[binding].inputRate)
			key.vs.instance_rate_inputs |= 1u << i;
	}
	return key;
}

VkResult
radv_pipeline_init(struct radv_pipeline *pipeline,
		   struct radv_device *device,
		   struct radv_pipeline_cache *cache,
		   const VkGraphicsPipelineCreateInfo *pCreateInfo,
		   const struct radv_graphics_pipeline_create_info *extra,
		   const VkAllocationCallbacks *alloc)
{
	VkResult result;
	struct nir_shader *shader;
	struct radv_shader_module fs_m = {0};

	bool dump = getenv("RADV_DUMP_SHADERS");
	if (alloc == NULL)
		alloc = &device->alloc;

	pipeline->device = device;
	pipeline->layout = radv_pipeline_layout_from_handle(pCreateInfo->layout);

	radv_pipeline_init_dynamic_state(pipeline, pCreateInfo);
	const VkPipelineShaderStageCreateInfo *pStages[MESA_SHADER_STAGES] = { 0, };
	struct radv_shader_module *modules[MESA_SHADER_STAGES] = { 0, };
	for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
		gl_shader_stage stage = ffs(pCreateInfo->pStages[i].stage) - 1;
		pStages[stage] = &pCreateInfo->pStages[i];
		modules[stage] = radv_shader_module_from_handle(pStages[stage]->module);
	}

	radv_pipeline_init_blend_state(pipeline, pCreateInfo);

	/* */
	if (modules[MESA_SHADER_VERTEX]) {
		union ac_shader_variant_key key = radv_compute_vs_key(pCreateInfo);

		shader = radv_pipeline_compile(pipeline, modules[MESA_SHADER_VERTEX],
					       pStages[MESA_SHADER_VERTEX]->pName,
					       MESA_SHADER_VERTEX,
					       pStages[MESA_SHADER_VERTEX]->pSpecializationInfo, dump);
		pipeline->shaders[MESA_SHADER_VERTEX] = radv_shader_variant_create(device,
										   shader,
										   pipeline->layout,
										   &key, dump);
		pipeline->active_stages |= mesa_to_vk_shader_stage(MESA_SHADER_VERTEX);

		if (!modules[MESA_SHADER_VERTEX]->nir)
			ralloc_free(shader);
	}

	if (!modules[MESA_SHADER_FRAGMENT]) {
		nir_builder fs_b;
		nir_builder_init_simple_shader(&fs_b, NULL, MESA_SHADER_FRAGMENT, NULL);
		fs_b.shader->info.name = ralloc_strdup(fs_b.shader, "noop_fs");
		fs_m.nir = fs_b.shader;
		modules[MESA_SHADER_FRAGMENT] = &fs_m;
	}

	if (modules[MESA_SHADER_FRAGMENT]) {
		union ac_shader_variant_key key;
		key.fs.col_format = pipeline->graphics.blend.spi_shader_col_format;
		key.fs.is_int8 = radv_pipeline_compute_is_int8(pCreateInfo);

		const VkPipelineShaderStageCreateInfo *stage = pStages[MESA_SHADER_FRAGMENT];

		shader = radv_pipeline_compile(pipeline, modules[MESA_SHADER_FRAGMENT],
					       stage ? stage->pName : "main",
					       MESA_SHADER_FRAGMENT,
					       stage ? stage->pSpecializationInfo :NULL, dump);
		pipeline->shaders[MESA_SHADER_FRAGMENT] = radv_shader_variant_create(device,
										     shader,
										     pipeline->layout,
										     &key, dump);
		pipeline->active_stages |= mesa_to_vk_shader_stage(MESA_SHADER_FRAGMENT);

		if (!modules[MESA_SHADER_FRAGMENT]->nir || fs_m.nir)
			ralloc_free(shader);
	}

	radv_pipeline_init_depth_stencil_state(pipeline, pCreateInfo);
	radv_pipeline_init_raster_state(pipeline, pCreateInfo);

	pipeline->graphics.prim = si_translate_prim(pCreateInfo->pInputAssemblyState->topology);
	if (extra && extra->use_rectlist)
		pipeline->graphics.prim = V_008958_DI_PT_RECTLIST;
	pipeline->graphics.prim_restart_enable = pCreateInfo->pInputAssemblyState->primitiveRestartEnable;

	const VkPipelineVertexInputStateCreateInfo *vi_info =
		pCreateInfo->pVertexInputState;
	for (uint32_t i = 0; i < vi_info->vertexAttributeDescriptionCount; i++) {
		const VkVertexInputAttributeDescription *desc =
			&vi_info->pVertexAttributeDescriptions[i];
		const struct vk_format_description *format_desc;
		int first_non_void;
		uint32_t num_format, data_format;
		format_desc = vk_format_description(desc->format);
		first_non_void = vk_format_get_first_non_void_channel(desc->format);

		num_format = radv_translate_buffer_numformat(format_desc, first_non_void);
		data_format = radv_translate_buffer_dataformat(format_desc, first_non_void);

		pipeline->va_rsrc_word3[i] = S_008F0C_DST_SEL_X(si_map_swizzle(format_desc->swizzle[0])) |
			S_008F0C_DST_SEL_Y(si_map_swizzle(format_desc->swizzle[1])) |
			S_008F0C_DST_SEL_Z(si_map_swizzle(format_desc->swizzle[2])) |
			S_008F0C_DST_SEL_W(si_map_swizzle(format_desc->swizzle[3])) |
			S_008F0C_NUM_FORMAT(num_format) |
			S_008F0C_DATA_FORMAT(data_format);

		pipeline->va_offset[i] = desc->offset;
		pipeline->va_binding[i] = desc->binding;
	}
	pipeline->num_vertex_attribs = vi_info->vertexAttributeDescriptionCount;
	for (uint32_t i = 0; i < vi_info->vertexBindingDescriptionCount; i++) {
		const VkVertexInputBindingDescription *desc =
			&vi_info->pVertexBindingDescriptions[i];

		pipeline->binding_stride[desc->binding] = desc->stride;

		/* Step rate is programmed per vertex element (attribute), not
		 * binding. Set up a map of which bindings step per instance, for
		 * reference by vertex element setup. */
		switch (desc->inputRate) {
		default:
		case VK_VERTEX_INPUT_RATE_VERTEX:
			pipeline->instancing_enable[desc->binding] = false;
			break;
		case VK_VERTEX_INPUT_RATE_INSTANCE:
			pipeline->instancing_enable[desc->binding] = true;
			break;
		}
	}

	return VK_SUCCESS;
}

VkResult
radv_graphics_pipeline_create(
	VkDevice _device,
	VkPipelineCache _cache,
	const VkGraphicsPipelineCreateInfo *pCreateInfo,
	const struct radv_graphics_pipeline_create_info *extra,
	const VkAllocationCallbacks *pAllocator,
	VkPipeline *pPipeline)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_pipeline_cache, cache, _cache);
	struct radv_pipeline *pipeline;
	VkResult result;

	//   if (cache == NULL)
	//      cache = &device->default_pipeline_cache;
	pipeline = radv_alloc2(&device->alloc, pAllocator, sizeof(*pipeline), 8,
			       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (pipeline == NULL)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	memset(pipeline, 0, sizeof(*pipeline));
	result = radv_pipeline_init(pipeline, device, cache,
				    pCreateInfo, extra, pAllocator);
	if (result != VK_SUCCESS) {
		radv_free2(&device->alloc, pAllocator, pipeline);
		return result;
	}

	*pPipeline = radv_pipeline_to_handle(pipeline);

	return VK_SUCCESS;
}

VkResult radv_CreateGraphicsPipelines(
	VkDevice                                    _device,
	VkPipelineCache                             pipelineCache,
	uint32_t                                    count,
	const VkGraphicsPipelineCreateInfo*         pCreateInfos,
	const VkAllocationCallbacks*                pAllocator,
	VkPipeline*                                 pPipelines)
{
	VkResult result = VK_SUCCESS;
	unsigned i = 0;

	for (; i < count; i++) {
		result = radv_graphics_pipeline_create(_device,
						       pipelineCache,
						       &pCreateInfos[i],
						       NULL, pAllocator, &pPipelines[i]);
		if (result != VK_SUCCESS) {
			for (unsigned j = 0; j < i; j++) {
				radv_DestroyPipeline(_device, pPipelines[j], pAllocator);
			}

			return result;
		}
	}

	return VK_SUCCESS;
}

static VkResult radv_compute_pipeline_create(
	VkDevice                                    _device,
	VkPipelineCache                             _cache,
	const VkComputePipelineCreateInfo*          pCreateInfo,
	const VkAllocationCallbacks*                pAllocator,
	VkPipeline*                                 pPipeline)
{
	RADV_FROM_HANDLE(radv_device, device, _device);
	RADV_FROM_HANDLE(radv_pipeline_cache, cache, _cache);
	RADV_FROM_HANDLE(radv_shader_module, module, pCreateInfo->stage.module);
	struct radv_pipeline *pipeline;
	struct nir_shader *shader;
	bool dump = getenv("RADV_DUMP_SHADERS");
	//   if (cache == NULL)
	//      cache = &device->default_pipeline_cache;

	pipeline = radv_alloc2(&device->alloc, pAllocator, sizeof(*pipeline), 8,
			       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
	if (pipeline == NULL)
		return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

	memset(pipeline, 0, sizeof(*pipeline));
	pipeline->device = device;
	pipeline->layout = radv_pipeline_layout_from_handle(pCreateInfo->layout);

	shader = radv_pipeline_compile(pipeline, module,
				       pCreateInfo->stage.pName,
				       MESA_SHADER_COMPUTE,
				       pCreateInfo->stage.pSpecializationInfo, dump);

	for (int i = 0; i < 3; ++i)
		pipeline->compute.block_size[i] = shader->info.cs.local_size[i];

	pipeline->shaders[MESA_SHADER_COMPUTE] = radv_shader_variant_create(device,
									    shader,
									    pipeline->layout,
									    NULL, dump);

	ralloc_free(shader);
	*pPipeline = radv_pipeline_to_handle(pipeline);
	return VK_SUCCESS;
}
VkResult radv_CreateComputePipelines(
	VkDevice                                    _device,
	VkPipelineCache                             pipelineCache,
	uint32_t                                    count,
	const VkComputePipelineCreateInfo*          pCreateInfos,
	const VkAllocationCallbacks*                pAllocator,
	VkPipeline*                                 pPipelines)
{
	VkResult result = VK_SUCCESS;

	unsigned i = 0;
	for (; i < count; i++) {
		result = radv_compute_pipeline_create(_device, pipelineCache,
						      &pCreateInfos[i],
						      pAllocator, &pPipelines[i]);
		if (result != VK_SUCCESS) {
			for (unsigned j = 0; j < i; j++) {
				radv_DestroyPipeline(_device, pPipelines[j], pAllocator);
			}

			return result;
		}
	}

	return VK_SUCCESS;
}
