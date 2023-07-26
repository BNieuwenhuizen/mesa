/*
 * Copyright 2023 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "glsl_types.h"
#include "nir.h"
#include "nir_types.h"
#include "vtn_private.h"

static enum glsl_cooperative_matrix_use
vtn_cooperative_matrix_use_to_glsl(SpvCooperativeMatrixUse use)
{
   switch (use) {
   case SpvCooperativeMatrixUseMatrixAKHR:
      return GLSL_COOPERATIVE_MATRIX_USE_A;
   case SpvCooperativeMatrixUseMatrixBKHR:
      return GLSL_COOPERATIVE_MATRIX_USE_B;
   case SpvCooperativeMatrixUseMatrixAccumulatorKHR:
      return GLSL_COOPERATIVE_MATRIX_USE_ACCUMULATOR;
   default:
      unreachable("Unexpected cooperative matrix use");
   }
}

void
vtn_handle_cooperative_type(struct vtn_builder *b, struct vtn_value *val,
                            SpvOp opcode, const uint32_t *w, unsigned count)
{
   vtn_assert(opcode == SpvOpTypeCooperativeMatrixKHR);

   b->shader->info.cs.has_cooperative_matrix = true;

   struct vtn_type *component_type = vtn_get_type(b, w[2]);

   const mesa_scope scope = vtn_translate_scope(b, vtn_constant_uint(b, w[3]));
   const uint32_t rows = vtn_constant_uint(b, w[4]);
   const uint32_t cols = vtn_constant_uint(b, w[5]);

   vtn_assert(rows < 256);
   vtn_assert(cols < 256);

   enum glsl_cooperative_matrix_use use =
      vtn_cooperative_matrix_use_to_glsl(vtn_constant_uint(b, w[6]));

   val->type->base_type = vtn_base_type_cooperative_matrix;
   vtn_fail_if(!glsl_type_is_numeric(component_type->type),
               "OpTypeCooperativeMatrixKHR "
               "Component Type must be a scalar numerical type.");

   val->type->desc.element_type = glsl_get_base_type(component_type->type);
   val->type->desc.scope = scope;
   val->type->desc.rows = rows;
   val->type->desc.cols = cols;
   val->type->desc.use = use;

   val->type->type = glsl_cooperative_matrix_type(&val->type->desc);
   val->type->component_type = component_type;
}

static enum glsl_matrix_layout
vtn_matrix_layout_to_glsl(SpvCooperativeMatrixLayout layout)
{
   switch (layout) {
   case SpvCooperativeMatrixLayoutRowMajorKHR:
      return GLSL_MATRIX_LAYOUT_ROW_MAJOR;
   case SpvCooperativeMatrixLayoutColumnMajorKHR:
      return GLSL_MATRIX_LAYOUT_COLUMN_MAJOR;
   default:
      unreachable("Unexpected cooperative matrix layout");
   }
}

void
vtn_handle_cooperative_instruction(struct vtn_builder *b, SpvOp opcode,
                                   const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpCooperativeMatrixLoadKHR: {
      struct vtn_value *src_val = vtn_value(b, w[3], vtn_value_type_pointer);
      struct vtn_pointer *src = vtn_value_to_pointer(b, src_val);
      struct vtn_type *dst_type = vtn_get_type(b, w[1]);

      const SpvCooperativeMatrixLayout layout = vtn_constant_uint(b, w[4]);
      nir_def *stride = count > 5 ? vtn_get_nir_ssa(b, w[5]) : nir_imm_zero(&b->nb, 1, 32);

      SpvMemoryAccessMask access = SpvMemoryAccessMaskNone;
      if (count > 6) {
         unsigned idx = 6, alignment;
         SpvScope scope;
         vtn_get_mem_operands(b, w, count, &idx, &access, &alignment, NULL, &scope);
         vtn_emit_make_visible_barrier(b, access, scope, src->mode);
      }

      nir_def *def = nir_coop_load(&b->nb, vtn_pointer_to_ssa(b, src), stride,
                                   .matrix_desc = dst_type->desc,
                                   .matrix_layout = vtn_matrix_layout_to_glsl(layout));
      vtn_push_nir_ssa(b, w[2], def);
      break;
   }

   case SpvOpCooperativeMatrixStoreKHR: {
      struct vtn_value *dest_val = vtn_value(b, w[1], vtn_value_type_pointer);
      struct vtn_pointer *dest = vtn_value_to_pointer(b, dest_val);

      const SpvCooperativeMatrixLayout layout = vtn_constant_uint(b, w[3]);
      nir_def *stride = count > 4 ? vtn_get_nir_ssa(b, w[4]) : nir_imm_zero(&b->nb, 1, 32);

      SpvMemoryAccessMask access = SpvMemoryAccessMaskNone;
      if (count > 5) {
         unsigned idx = 5, alignment;
         SpvScope scope;
         vtn_get_mem_operands(b, w, count, &idx, &access, &alignment, &scope, NULL);
         vtn_emit_make_available_barrier(b, access, scope, dest->mode);
      }

      struct vtn_ssa_value *src = vtn_ssa_value(b, w[2]);
      vtn_assert(glsl_type_is_cooperative_matrix(src->type));

      nir_coop_store(&b->nb, vtn_pointer_to_ssa(b, dest), src->def, stride,
                     .matrix_desc = *glsl_get_cooperative_matrix_description(src->type),
                     .matrix_layout = vtn_matrix_layout_to_glsl(layout));
      break;
   }

   case SpvOpCooperativeMatrixLengthKHR: {
      struct vtn_type *type = vtn_get_type(b, w[3]);
      nir_def *def = nir_coop_length(&b->nb, .matrix_desc = type->desc);
      vtn_push_nir_ssa(b, w[2], def);
      break;
   }

   case SpvOpCooperativeMatrixMulAddKHR: {
      nir_def *mat_a = vtn_get_nir_ssa(b, w[3]);
      nir_def *mat_b = vtn_get_nir_ssa(b, w[4]);
      nir_def *mat_c = vtn_get_nir_ssa(b, w[5]);

      const bool saturate = w[6] & SpvCooperativeMatrixOperandsSaturatingAccumulationKHRMask;
      const unsigned signed_mask = w[6] & (SpvCooperativeMatrixOperandsMatrixASignedComponentsKHRMask |
                                           SpvCooperativeMatrixOperandsMatrixBSignedComponentsKHRMask |
                                           SpvCooperativeMatrixOperandsMatrixCSignedComponentsKHRMask |
                                           SpvCooperativeMatrixOperandsMatrixResultSignedComponentsKHRMask);

      STATIC_ASSERT((unsigned)SpvCooperativeMatrixOperandsMatrixASignedComponentsKHRMask == NIR_COOPERATIVE_MATRIX_A_SIGNED);
      STATIC_ASSERT((unsigned)SpvCooperativeMatrixOperandsMatrixBSignedComponentsKHRMask == NIR_COOPERATIVE_MATRIX_B_SIGNED);
      STATIC_ASSERT((unsigned)SpvCooperativeMatrixOperandsMatrixCSignedComponentsKHRMask == NIR_COOPERATIVE_MATRIX_C_SIGNED);
      STATIC_ASSERT((unsigned)SpvCooperativeMatrixOperandsMatrixResultSignedComponentsKHRMask == NIR_COOPERATIVE_MATRIX_RESULT_SIGNED);

      nir_def *def = nir_coop_muladd(&b->nb, mat_a, mat_b, mat_c,
                                     .matrix_desc = vtn_get_type(b, w[1])->desc,
                                     .saturate = saturate, .matrix_signed_mask = signed_mask);
      vtn_push_nir_ssa(b, w[2], def);
      break;
   }

   case SpvOpBitcast: {
      struct vtn_type *type = vtn_get_type(b, w[1]);
      vtn_assert(type->base_type == vtn_base_type_cooperative_matrix);
      nir_def *src = vtn_get_nir_ssa(b, w[3]);
      nir_def *def = nir_coop_bitcast(&b->nb, src,
                                      .matrix_desc = type->desc);
      vtn_push_nir_ssa(b, w[2], def);
      break;
   }

   default:
      unreachable("Unexpected opcode for cooperative matrix instruction");
   }
}

void
vtn_handle_cooperative_alu(struct vtn_builder *b, struct vtn_value *dest_val,
                           const struct glsl_type *dest_type, SpvOp opcode,
                           const uint32_t *w, unsigned count)
{
      vtn_assert(glsl_type_is_cooperative_matrix(dest_type));

      switch (opcode) {
      case SpvOpConvertFToU:
      case SpvOpConvertFToS:
      case SpvOpConvertSToF:
      case SpvOpConvertUToF:
      case SpvOpUConvert:
      case SpvOpSConvert:
      case SpvOpFConvert:
      case SpvOpFNegate:
      case SpvOpSNegate: {
         struct vtn_type *dst_type = vtn_get_type(b, w[1]);
         struct vtn_ssa_value *src_val = vtn_ssa_value(b, w[3]);

         unsigned src_bit_size =
            glsl_get_bit_size(glsl_get_cooperative_matrix_element(src_val->type));

         unsigned dst_bit_size =
            glsl_get_bit_size(glsl_get_cooperative_matrix_element(dst_type->type));

         bool ignored = false;
         nir_op op = vtn_nir_alu_op_for_spirv_opcode(b, opcode, &ignored, &ignored,
                                                     src_bit_size, dst_bit_size);

         nir_def *def = nir_coop_unary_op(&b->nb, src_val->def,
                                          .matrix_desc = dst_type->desc,
                                          .alu_op = op);
         vtn_push_nir_ssa(b, w[2], def);
         break;
      }

      case SpvOpFAdd:
      case SpvOpFSub:
      case SpvOpFMul:
      case SpvOpFDiv:
      case SpvOpIAdd:
      case SpvOpISub:
      case SpvOpIMul:
      case SpvOpSDiv:
      case SpvOpUDiv: {
         bool ignored = false;
         nir_op op = vtn_nir_alu_op_for_spirv_opcode(b, opcode, &ignored, &ignored, 0, 0);

         struct vtn_type *dst_type = vtn_get_type(b, w[1]);
         nir_def *mat_a = vtn_get_nir_ssa(b, w[3]);
         nir_def *mat_b = vtn_get_nir_ssa(b, w[4]);

         nir_def *def = nir_coop_binary_op(&b->nb, mat_a, mat_b,
                                           .matrix_desc = dst_type->desc,
                                           .alu_op = op);
         vtn_push_nir_ssa(b, w[2], def);
         break;
      }

      case SpvOpMatrixTimesScalar: {
         struct vtn_type *dst_type = vtn_get_type(b, w[1]);
         nir_def *mat = vtn_get_nir_ssa(b, w[3]);

         struct vtn_ssa_value *scalar_val = vtn_ssa_value(b, w[4]);
         vtn_assert(glsl_type_is_scalar(scalar_val->type));
         nir_op op = glsl_type_is_integer(scalar_val->type) ? nir_op_imul : nir_op_fmul;

         nir_def *def = nir_coop_scalar_op(&b->nb, mat, scalar_val->def,
                                           .matrix_desc = dst_type->desc,
                                           .alu_op = op);
         vtn_push_nir_ssa(b, w[2], def);
         break;
      }

      default:
         unreachable("invalid cooperative matrix alu instruction");
      }
}

struct vtn_ssa_value *
vtn_cooperative_matrix_extract(struct vtn_builder *b, struct vtn_ssa_value *mat,
                               const uint32_t *indices, unsigned num_indices)
{
   vtn_assert(glsl_type_is_cooperative_matrix(mat->type));

   vtn_assert(num_indices == 1);
   nir_def *index = nir_imm_intN_t(&b->nb, indices[0], 32);

   const struct glsl_type *element_type = glsl_get_cooperative_matrix_element(mat->type);
   struct vtn_ssa_value *ret = vtn_create_ssa_value(b, element_type);
   ret->def = nir_coop_extract(&b->nb, glsl_get_bit_size(element_type), mat->def, index);
   return ret;
}

struct vtn_ssa_value *
vtn_cooperative_matrix_insert(struct vtn_builder *b, struct vtn_ssa_value *mat,
                              struct vtn_ssa_value *insert, const uint32_t *indices,
                              unsigned num_indices)
{
   vtn_assert(glsl_type_is_cooperative_matrix(mat->type));

   vtn_assert(num_indices == 1);
   nir_def *index = nir_imm_intN_t(&b->nb, indices[0], 32);

   struct vtn_ssa_value *ret = vtn_create_ssa_value(b, mat->type);
   ret->def = nir_coop_insert(&b->nb, insert->def, mat->def, index,
                              .matrix_desc = *glsl_get_cooperative_matrix_description(ret->type));
   return ret;
}
