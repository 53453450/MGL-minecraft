/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_pipeline_cache_path.h - the pipeline binary archive's on-disk path
 * (P0-1, log 203).  Pure C (POSIX + CoreFoundation); the shell keeps only the
 * NSURL construction the Metal-cpp archive API wants.
 */

#ifndef MGL_PIPELINE_CACHE_PATH_H
#define MGL_PIPELINE_CACHE_PATH_H

#include <stddef.h>
#include <stdint.h>

#include "mgl_render.h"   /* mglRenderGetDeviceIdentity */

#ifdef __cplusplus
extern "C" {
#endif

/* Fills `out` with the archive path for `device`:
 *   <caches>/MGL/<sanitized bundle id or process name>/pipeline-<schema>-cpp-<device>.binaryarchive
 * Creates the intermediate directories.  Returns 0 on success. */
int mglPipelineCacheArchiveKey(const void *device, const char *schema,
                               char *out, size_t cap);

/* The .m's [NSFileManager fileExistsAtPath:] / removeItemAtURL:error:.  Both
 * treat "already gone" as success, like the Objective-C pair did. */
int mglPipelineCacheArchiveExists(const char *path);
int mglPipelineCacheArchiveRemove(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PIPELINE_CACHE_PATH_H */
