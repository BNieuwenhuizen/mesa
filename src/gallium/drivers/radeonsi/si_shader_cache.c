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

#include "util/mesa-sha1.h"

#include <sys/stat.h>


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
	if(!cache)
		return false;
	
	return false;
}

void si_shader_cache_save(struct si_shader_cache *cache, struct si_shader *shader)
{
	if(!cache)
		return;
}