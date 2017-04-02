/*
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

#include "nir.h"
#include "util/set.h"
#include "util/hash_table.h"

/* This file contains various little helpers for doing simple linking in
 * NIR.  Eventually, we'll probably want a full-blown varying packing
 * implementation in here.  Right now, it just deletes unused things.
 */

static bool
mark_outputs_written_block(nir_block *block, void *void_set)
{
   struct set *live_set = void_set;

   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      /* Ignore stores of undefs */
      if (intrin->intrinsic == nir_intrinsic_store_var &&
          intrin->src[0].is_ssa &&
          intrin->src[0].ssa->parent_instr->type == nir_instr_type_ssa_undef)
         continue;

      unsigned num_vars = nir_intrinsic_infos[intrin->intrinsic].num_variables;
      for (unsigned i = 0; i < num_vars; i++) {
         nir_variable *var = intrin->variables[i]->var;
         if (var->data.mode == nir_var_shader_out)
            _mesa_set_add(live_set, var);
      }
   }

   return true;
}

bool
nir_remove_unwritten_outputs(nir_shader *shader)
{
   struct set *live = _mesa_set_create(NULL, _mesa_hash_pointer,
                                       _mesa_key_pointer_equal);

   nir_foreach_function(func, shader) {
      if (func->impl) {
        nir_foreach_block(block, func->impl) {
          mark_outputs_written_block(block, live);
        }
      }
   }

   bool progress = false;
   nir_foreach_variable_safe(var, &shader->outputs) {
      if (_mesa_set_search(live, var) == NULL) {
         /* Make it a global instead.
          *
          * TODO: For tess, we'll need to move this to SLM or something?
          */
         exec_node_remove(&var->node);
         exec_list_push_tail(&shader->globals, &var->node);
         progress = true;
      }
   }

   return progress;
}

static bool
remove_unused_io_vars(nir_shader *shader, struct exec_list *var_list,
                      uint64_t used_by_other_stage, uint64_t *still_valid)
{
   bool progress = false;
   uint64_t valid = 0;
   nir_foreach_variable_safe(var, var_list) {
      if (var->data.location >= VARYING_SLOT_MAX)
         continue;

      if (used_by_other_stage & nir_variable_get_io_mask(var, shader->stage)) {
         valid |= nir_variable_get_io_mask(var, shader->stage);
      } else {
         /* This one is invalid, make it a global variable instead */
         var->data.location = 0;
         var->data.mode = nir_var_global;
         exec_node_remove(&var->node);
         exec_list_push_tail(&shader->globals, &var->node);

         progress = true;
      }
   }

   *still_valid = valid;

   return progress;
}

bool
nir_remove_unused_varyings(nir_shader *producer, nir_shader *consumer)
{
   assert(producer->stage != MESA_SHADER_FRAGMENT);
   assert(consumer->stage != MESA_SHADER_VERTEX);

   uint64_t read = 0, written = 0;

   /* We might be able to get this from nir_shader_info, but we can't count
    * on it.  Just walk the lists and compute it here.
    */
   nir_foreach_variable(var, &producer->outputs)
      if (var->data.location < VARYING_SLOT_MAX)
         written |= nir_variable_get_io_mask(var, producer->stage);

   nir_foreach_variable(var, &consumer->inputs)
      if (var->data.location < VARYING_SLOT_MAX)
         read |= nir_variable_get_io_mask(var, consumer->stage);

   /* This one is always considered to be read */
   if (consumer->stage == MESA_SHADER_FRAGMENT) {
      read |= (1ull << VARYING_SLOT_POS) |
              (1ull << VARYING_SLOT_CLIP_DIST0) |
              (1ull << VARYING_SLOT_CLIP_DIST1) |
              (1ull << VARYING_SLOT_CULL_DIST0) |
              (1ull << VARYING_SLOT_CULL_DIST1) |
              (1ull << VARYING_SLOT_LAYER) |
              (1ull << VARYING_SLOT_VIEWPORT) |
              (1ull << VARYING_SLOT_PSIZ);
      written |= (1ull << VARYING_SLOT_PNTC);
   }

   if (producer->stage == MESA_SHADER_TESS_CTRL) {
      read |= (1ull << VARYING_SLOT_TESS_LEVEL_OUTER) |
              (1ull << VARYING_SLOT_TESS_LEVEL_INNER);
   }

   remove_unused_io_vars(producer, &producer->outputs, read,
                         &producer->info.outputs_written);

   remove_unused_io_vars(consumer, &consumer->inputs, written,
                         &consumer->info.inputs_read);

   return written != producer->info.outputs_written ||
          read != consumer->info.inputs_read;
}

bool
nir_remove_unread_outputs(nir_shader *shader, uint64_t outputs_read)
{
   return remove_unused_io_vars(shader, &shader->outputs, outputs_read,
                                &shader->info.outputs_written);
}

static void
compact_var_list(nir_shader *shader, struct exec_list *var_list,
                 uint64_t valid, uint64_t *slots_used)
{
   int remap[64];
   for (int i = VARYING_SLOT_VAR0, j = VARYING_SLOT_VAR0; i < VARYING_SLOT_MAX; i++) {
      if (valid & (1ull << i)) {
         remap[i] = j++;
      } else {
         remap[i] = -1;
      }
   }

   uint64_t slots_used_tmp = 0;
   nir_foreach_variable_safe(var, var_list) {
      assert(var->data.location >= 0);
      if (var->data.location >= VARYING_SLOT_MAX)
         continue;

      /* Only remap things that aren't built-ins */
      if (var->data.location >= VARYING_SLOT_VAR0) {
         assert(var->data.location < 64);
         assert(remap[var->data.location] >= 0);

         var->data.location = remap[var->data.location];
      }

      slots_used_tmp |= nir_variable_get_io_mask(var, shader->stage);
   }

   *slots_used = slots_used_tmp;
}

bool
nir_compact_varyings(nir_shader *producer, nir_shader *consumer)
{
   assert(producer->stage != MESA_SHADER_FRAGMENT);
   assert(consumer->stage != MESA_SHADER_VERTEX);

   /* We assume that this has been called more-or-less directly after
    * remove_unused_varyings.  At this point, all of the varyings that we
    * aren't going to be using have been completely removed and the
    * inputs_read and outputs_written fields in nir_shader_info reflect
    * this.  Therefore, the total set of valid slots is the OR of the two
    * sets of varyings;  this accounts for varyings which one side may need
    * to read/write even if the other doesn't.  This can happen if, for
    * instance, an array is used indirectly from one side causing it to be
    * unsplittable but directly from the other.
    */
   uint64_t written = producer->info.outputs_written;
   uint64_t read = consumer->info.inputs_read;
   uint64_t valid = written | read;

   compact_var_list(producer, &producer->outputs, valid,
                    &producer->info.outputs_written);
   compact_var_list(consumer, &consumer->inputs, valid,
                    &consumer->info.inputs_read);

   return written != producer->info.outputs_written ||
          read != consumer->info.inputs_read;
}
