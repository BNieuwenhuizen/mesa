from __future__ import print_function

template = """\
/* Copyright (C) 2015 Broadcom
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

#ifndef _NIR_BUILDER_OPCODES_
#define _NIR_BUILDER_OPCODES_

<%
opcode_remap = {
   'b2i' : 'b322i',
   'b2f' : 'b322f',
   'i2b' : 'i2b32',
   'f2b' : 'f2b32',

   'flt' : 'flt32',
   'fge' : 'fge32',
   'feq' : 'feq32',
   'fne' : 'fne32',
   'ilt' : 'ilt32',
   'ige' : 'ige32',
   'ieq' : 'ieq32',
   'ine' : 'ine32',
   'ult' : 'ult32',
   'uge' : 'uge32',

   'ball_iequal' : 'b32all_iequal',
   'bany_inequal' : 'b32any_inequal',
   'ball_fequal' : 'b32all_fequal',
   'bany_fnequal' : 'b32any_fnequal',

   'bcsel' : 'b32csel',
}

opcode_remap32 = { op32 : op for op, op32 in opcode_remap.items() }

def src_decl_list(num_srcs):
   return ', '.join('nir_ssa_def *src' + str(i) for i in range(num_srcs))

def src_list(num_srcs):
   return ', '.join('src' + str(i) if i < num_srcs else 'NULL' for i in range(4))
%>

% for name, opcode in sorted(opcodes.items()):
  % if name in opcode_remap:
    <% continue %>
  % elif name in opcode_remap32:
    <% builder_name = opcode_remap32[name] %>
  % else:
    <% builder_name = name %>
  % endif
static inline nir_ssa_def *
nir_${builder_name}(nir_builder *build, ${src_decl_list(opcode.num_inputs)})
{
   return nir_build_alu(build, nir_op_${name}, ${src_list(opcode.num_inputs)});
}
% endfor

/* Generic builder for system values. */
static inline nir_ssa_def *
nir_load_system_value(nir_builder *build, nir_intrinsic_op op, int index)
{
   nir_intrinsic_instr *load = nir_intrinsic_instr_create(build->shader, op);
   load->num_components = nir_intrinsic_infos[op].dest_components;
   load->const_index[0] = index;
   nir_ssa_dest_init(&load->instr, &load->dest,
                     nir_intrinsic_infos[op].dest_components, 32, NULL);
   nir_builder_instr_insert(build, &load->instr);
   return &load->dest.ssa;
}

% for name, opcode in filter(lambda v: v[1].sysval, sorted(INTR_OPCODES.items())):
static inline nir_ssa_def *
nir_${name}(nir_builder *build)
{
   return nir_load_system_value(build, nir_intrinsic_${name}, 0);
}
% endfor

#endif /* _NIR_BUILDER_OPCODES_ */"""

from nir_opcodes import opcodes
from nir_intrinsics import INTR_OPCODES
from mako.template import Template

print(Template(template).render(opcodes=opcodes, INTR_OPCODES=INTR_OPCODES))
