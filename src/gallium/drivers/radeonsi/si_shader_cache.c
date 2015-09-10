/*
 * Copyright 2015 Bas Nieuwenhuizen
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "si_pipe.h"
#include "si_shader.h"

#include "radeon/radeon_elf_util.h"

#include "util/mesa-sha1.h"

#include <sys/stat.h>
#include <fcntl.h>


static char *concatenate_path(const char *a, const char* b)
{
	char* ret;
	size_t a_len, b_len;

	a_len = strlen(a);
	b_len = strlen(b);

	ret = malloc(a_len + b_len + 2);
	if(!ret)
		return NULL;

	memcpy(ret, a, a_len);
	ret[a_len] = '/';

	memcpy(ret + a_len + 1, b, b_len);
	ret[a_len + b_len + 1] = 0;
	
	return ret;
}

static bool validate_and_create_dir(const char* path)
{
	struct stat st;
	
	if(stat(path, &st) == 0)
		return S_ISDIR(st.st_mode);

	return mkdir(path, 0755) == 0;
}

static char* si_shader_cache_filename(struct si_shader_cache *cache, const unsigned char* hash)
{
	char filename[41];

	_mesa_sha1_format(filename, hash);
	strcpy(filename + 40, "");
	return concatenate_path(cache->path, filename);
}

static bool hash_shader(struct si_shader *shader, unsigned char* hash)
{
	struct mesa_sha1 *sha_ctx;
	struct si_shader_selector sel = {};
	struct tgsi_header header = *(struct tgsi_header*)shader->selector->tokens;

	sel = *shader->selector;
	sel.tokens = NULL;
	sel.current = NULL;
	sel.num_shaders = 0;

	sha_ctx = _mesa_sha1_init();
	if(!sha_ctx)
		return false;

	if(!_mesa_sha1_update(sha_ctx, &sel, sizeof(struct si_shader_selector)))
		return false;

	if(!_mesa_sha1_update(sha_ctx, shader->selector->tokens, sizeof(struct tgsi_token) * (header.HeaderSize + header.BodySize)))
		return false;

	if(!_mesa_sha1_update(sha_ctx, &shader->key, sizeof(union si_shader_key)))
		return false;

	if(!_mesa_sha1_final(sha_ctx, hash))
		return false;

	return true;
}

static void deserialize_uint32_t(const uint8_t** data, uint32_t* value)
{
	memcpy(value, *data, 4);
	*data += 4;
}

static void deserialize_uint64_t(const uint8_t** data, uint64_t* value)
{
	memcpy(value, *data, 8);
	*data += 8;
}

static void deserialize_unsigned_32(const uint8_t** data, unsigned* value)
{
	uint32_t u32;
	deserialize_uint32_t(data, &u32);
	*value = u32;
}

static void deserialize_bool(const uint8_t** data, bool* value)
{
	uint8_t v;

	memcpy(&v, *data, 1);
	*data += 1;

	*value = v != 0;
}

static void serialize_uint32_t(uint8_t** data, uint32_t value)
{
	memcpy(*data, &value, 4);
	*data += 4;
}

static void serialize_uint64_t(uint8_t** data, uint64_t value)
{
	memcpy(*data, &value, 8);
	*data += 8;
}

static void serialize_unsigned_32(uint8_t** data, unsigned value)
{
	serialize_uint32_t(data, value);
}

static void serialize_bool(uint8_t** data, bool value) {
	uint8_t v = value ? 1 : 0;

	memcpy(*data, &v, 1);
	*data += 1;
}

static bool deserialize_shader(struct si_shader* shader, const uint8_t* data)
{
	unsigned name_size;
	unsigned i;
	struct radeon_shader_binary binary;

	memset(&binary, 0, sizeof(struct radeon_shader_binary));

	deserialize_unsigned_32(&data, &binary.code_size);
	deserialize_unsigned_32(&data, &binary.config_size);
	deserialize_unsigned_32(&data, &binary.config_size_per_symbol);
	deserialize_unsigned_32(&data, &binary.rodata_size);
	deserialize_unsigned_32(&data, &binary.global_symbol_count);
	deserialize_unsigned_32(&data, &binary.reloc_count);

	binary.relocs = calloc(binary.reloc_count, sizeof(binary.relocs[0]));
	if(!binary.relocs)
		return false;

	binary.code = malloc(binary.code_size);
	if(!binary.code)
		goto fail;

	binary.config = malloc(binary.config_size);
	if(!binary.config)
		goto fail;

	binary.rodata = malloc(binary.rodata_size);
	if(!binary.rodata)
		goto fail;

	binary.global_symbol_offsets = calloc(binary.global_symbol_count, sizeof(binary.global_symbol_offsets[0]));
	if(!binary.global_symbol_offsets)
		goto fail;

	memcpy(binary.code, data, binary.code_size);
	data += binary.code_size;

	memcpy(binary.config, data, binary.config_size);
	data += binary.config_size;

	memcpy(binary.rodata, data, binary.rodata_size);
	data += binary.rodata_size;

	memcpy(binary.global_symbol_offsets, data, binary.global_symbol_count * sizeof(binary.global_symbol_offsets[0]));
	data += binary.global_symbol_count * sizeof(binary.global_symbol_offsets[0]);

	for(i = 0; i < binary.reloc_count; ++i) {
		deserialize_uint64_t(&data, &binary.relocs[i].offset);
		deserialize_unsigned_32(&data, &name_size);

		binary.relocs[i].name = malloc(name_size + 1);
		if(!binary.relocs[i].name)
		goto fail;

		memcpy(binary.relocs[i].name, data, name_size);
		data += name_size;
		binary.relocs[i].name[name_size] = 0;
	}

	deserialize_unsigned_32(&data, &shader->num_sgprs);
	deserialize_unsigned_32(&data, &shader->num_vgprs);
	deserialize_unsigned_32(&data, &shader->lds_size);
	deserialize_unsigned_32(&data, &shader->spi_ps_input_ena);
	deserialize_unsigned_32(&data, &shader->float_mode);
	deserialize_unsigned_32(&data, &shader->scratch_bytes_per_wave);
	deserialize_unsigned_32(&data, &shader->spi_shader_col_format);
	deserialize_unsigned_32(&data, &shader->spi_shader_z_format);
	deserialize_unsigned_32(&data, &shader->db_shader_control);
	deserialize_unsigned_32(&data, &shader->cb_shader_mask);
	deserialize_unsigned_32(&data, &shader->nparam);

	for(i = 0; i < PIPE_MAX_SHADER_OUTPUTS; ++i)
		deserialize_unsigned_32(&data, &shader->vs_output_param_offset[i]);

	for(i = 0; i < PIPE_MAX_SHADER_OUTPUTS; ++i)
		deserialize_unsigned_32(&data, &shader->ps_input_param_offset[i]);

	for(i = 0; i < PIPE_MAX_SHADER_OUTPUTS; ++i)
		deserialize_unsigned_32(&data, &shader->ps_input_interpolate[i]);

	deserialize_bool(&data, &shader->uses_instanceid);
	deserialize_unsigned_32(&data, &shader->nr_pos_exports);
	deserialize_unsigned_32(&data, &shader->nr_param_exports);
	deserialize_bool(&data, &shader->is_gs_copy_shader);
	deserialize_bool(&data, &shader->dx10_clamp_mode);

	deserialize_unsigned_32(&data, &shader->ls_rsrc1);
	deserialize_unsigned_32(&data, &shader->ls_rsrc2);

	memcpy(&shader->binary, &binary, sizeof(binary));
	
	return true;
    
fail:
	radeon_shader_binary_free_members(&binary, true);
	free(binary.disasm_string);
	free(binary.global_symbol_offsets);
	return false;
}

static void serialize_shader(struct si_shader* shader, uint8_t* data)
{
	unsigned name_size;
	unsigned i;

	serialize_unsigned_32(&data, shader->binary.code_size);
	serialize_unsigned_32(&data, shader->binary.config_size);
	serialize_unsigned_32(&data, shader->binary.config_size_per_symbol);
	serialize_unsigned_32(&data, shader->binary.rodata_size);
	serialize_unsigned_32(&data, shader->binary.global_symbol_count);
	serialize_unsigned_32(&data, shader->binary.reloc_count);

	memcpy(data, shader->binary.code, shader->binary.code_size);
	data += shader->binary.code_size;

	memcpy(data, shader->binary.config, shader->binary.config_size);
	data += shader->binary.config_size;

	memcpy(data, shader->binary.rodata, shader->binary.rodata_size);
	data += shader->binary.rodata_size;

	memcpy(data, shader->binary.global_symbol_offsets, shader->binary.global_symbol_count * sizeof(shader->binary.global_symbol_offsets[0]));
	data += shader->binary.global_symbol_count * sizeof(shader->binary.global_symbol_offsets[0]);

	for(i = 0; i < shader->binary.reloc_count; ++i) {
		serialize_uint64_t(&data, shader->binary.relocs[i].offset);

		name_size = strlen(shader->binary.relocs[i].name);
		serialize_unsigned_32(&data, name_size);

		memcpy(data, shader->binary.relocs[i].name, name_size);
		data += name_size;
	}

	serialize_unsigned_32(&data, shader->num_sgprs);
	serialize_unsigned_32(&data, shader->num_vgprs);
	serialize_unsigned_32(&data, shader->lds_size);
	serialize_unsigned_32(&data, shader->spi_ps_input_ena);
	serialize_unsigned_32(&data, shader->float_mode);
	serialize_unsigned_32(&data, shader->scratch_bytes_per_wave);
	serialize_unsigned_32(&data, shader->spi_shader_col_format);
	serialize_unsigned_32(&data, shader->spi_shader_z_format);
	serialize_unsigned_32(&data, shader->db_shader_control);
	serialize_unsigned_32(&data, shader->cb_shader_mask);
	serialize_unsigned_32(&data, shader->nparam);

	for(i = 0; i < PIPE_MAX_SHADER_OUTPUTS; ++i)
		serialize_unsigned_32(&data, shader->vs_output_param_offset[i]);

	for(i = 0; i < PIPE_MAX_SHADER_OUTPUTS; ++i)
		serialize_unsigned_32(&data, shader->ps_input_param_offset[i]);

	for(i = 0; i < PIPE_MAX_SHADER_OUTPUTS; ++i)
		serialize_unsigned_32(&data, shader->ps_input_interpolate[i]);

	serialize_bool(&data, shader->uses_instanceid);
	serialize_unsigned_32(&data, shader->nr_pos_exports);
	serialize_unsigned_32(&data, shader->nr_param_exports);
	serialize_bool(&data, shader->is_gs_copy_shader);
	serialize_bool(&data, shader->dx10_clamp_mode);

	serialize_unsigned_32(&data, shader->ls_rsrc1);
	serialize_unsigned_32(&data, shader->ls_rsrc2);

	return true;
}

static size_t serialize_shader_size(struct si_shader* shader)
{
	size_t name_size, size;
	unsigned i;

	size = 0;

	/* binary variables of constant size */
	size += 6 * 4;

	size += shader->binary.code_size;
	size += shader->binary.config_size;
	size += shader->binary.rodata_size;
	size += shader->binary.global_symbol_count * sizeof(shader->binary.global_symbol_offsets[0]);
	for(i = 0; i < shader->binary.reloc_count; ++i) {
		name_size = strlen(shader->binary.relocs[i].name);

		size += 8 + 4;
		size += name_size;
	}

	size += (15 + 3 * PIPE_MAX_SHADER_OUTPUTS) * 4 + 3 * 1;

	size += sizeof(*shader) - offsetof(struct si_shader, num_sgprs);
	return size;
}

struct si_shader_cache *si_create_shader_cache(void)
{
	struct si_shader_cache *cache;
	char* homedir;

	cache = calloc(1, sizeof(struct si_shader_cache));
	if(!cache)
		return NULL;

	cache->path = getenv("RADEONSI_SHADER_CACHE_DIR");
	if(cache->path && !validate_and_create_dir(cache->path)) {
		fprintf(stderr, "$RADEONSI_SHADER_CACHE_DIR is not a directory\n");
		cache->path = NULL;
		goto fail;
	}

	if(!cache->path) {
		homedir = getenv("HOME");
		cache->path = concatenate_path(homedir, ".cache/mesa-radeonsi");
		if(!cache->path || !validate_and_create_dir(cache->path))
			goto fail;
	}

	return cache;

fail:
	free(cache->path);
	free(cache);
	return NULL;
}

void si_destroy_shader_cache(struct si_shader_cache *cache)
{
	if(!cache)
		return;

	free(cache->path);
	free(cache);
}

bool si_shader_cache_load(struct si_shader_cache *cache, struct si_shader *shader)
{
	off_t size;
	uint8_t hash[32];
	uint8_t* buf;
	char* filename;
	int fd;

	if(!cache)
		return false;

	hash_shader(shader, hash);
	
	filename = si_shader_cache_filename(cache, hash);
	
	fd = open(filename, O_RDONLY);
	free(filename);
	
	if(fd < 0)
		return false;
	
	size = lseek(fd, 0, SEEK_END);
	if(size == -1) {
		close(fd);
		return false;
	}
	
	buf = malloc(size);
	if(!buf)
		return false;
	
	if(pread(fd, buf, size, 0) != size) {
		printf("read binary failed\n");
		close(fd);
		free(buf);
		return false;
	}
	
	close(fd);
	
	if(!deserialize_shader(shader, buf)) {
		free(buf);
		return false;
	}
	
	free(buf);
	return true;
}

void si_shader_cache_save(struct si_shader_cache *cache, struct si_shader *shader)
{
	if(!cache)
		return;
	
	uint8_t hash[20];
	char* filename = NULL;
	char* filename_tmp = NULL;
	size_t size;

	int fd = -1;
	uint8_t* buf;

	hash_shader(shader, hash);

	size = serialize_shader_size(shader);

	buf = malloc(size);
	if(!buf)
		return;

	serialize_shader(shader, buf);

	filename = si_shader_cache_filename(cache, hash);
	if(!filename)
		goto out;
	
	filename_tmp = malloc(strlen(filename) + 8);
	if(!filename_tmp)
		goto out;
	strcpy(filename_tmp, filename);
	strcat(filename_tmp, "-XXXXXX");

	fd = mkstemp(filename_tmp);
	if(fd < 0)
		goto out;

	if(write(fd, buf, size) != size)
		goto out;

	if(rename(filename_tmp, filename))
		goto out;
	
	close(fd);
	fd = -1;

out:
	if(fd >= 0) {
		unlink(filename_tmp);
		close(fd);
	}

	free(filename_tmp);
	free(filename);
	free(buf);
}