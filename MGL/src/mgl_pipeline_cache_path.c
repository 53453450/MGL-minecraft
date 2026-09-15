/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_pipeline_cache_path.c - the pipeline binary archive's on-disk path
 * (P0-1, log 203).  The shell used to build it with Foundation
 * (NSSearchPathForDirectoriesInDomains / NSBundle / NSProcessInfo /
 * NSFileManager / NSString), which is exactly the "Foundation -> POSIX" step
 * T5's removal path asks for.  The result is byte-for-byte the same path:
 *
 *   <caches>/MGL/<sanitized bundle id or process name>/pipeline-<schema>-<device>.binaryarchive
 *
 * <caches> is ~/Library/Caches (NSCachesDirectory in the user domain), with
 * TMPDIR as the fallback the .m used when the search returned nothing.
 * Sanitization replaces every non-alphanumeric character with '_' -- which is
 * what componentsSeparatedByCharactersInSet + componentsJoinedByString did,
 * including runs and trailing characters.
 *
 * CoreFoundation is a C API, so the bundle identifier stays available without
 * pulling Objective-C into this translation unit.
 */

#include "mgl_pipeline_cache_path.h"

#include <CoreFoundation/CoreFoundation.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* The .m's static MGLSafeArchivePathComponent, in C. */
static void mglPcpSanitize(const char *value, char *out, size_t cap)
{
    if (!out || cap == 0u) return;
    if (!value || value[0] == '\0') {
        snprintf(out, cap, "unknown");
        return;
    }
    size_t n = 0u;
    for (const char *p = value; *p != '\0' && n + 1u < cap; ++p) {
        const unsigned char c = (unsigned char)*p;
        const int alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                          (c >= 'A' && c <= 'Z');
        out[n++] = alnum ? (char)c : '_';
    }
    out[n] = '\0';
}

/* The .m fell back to NSProcessInfo.processName when the bundle had no id. */
static void mglPcpProcessName(char *out, size_t cap)
{
    const char *name = getprogname();
    if (!name || name[0] == '\0') {
        name = "unknown";
    }
    snprintf(out, cap, "%s", name);
}

static void mglPcpBundleIdentifier(char *out, size_t cap)
{
    out[0] = '\0';
    CFBundleRef bundle = CFBundleGetMainBundle();
    CFStringRef identifier = bundle ? CFBundleGetIdentifier(bundle) : NULL;
    if (!identifier) {
        return;
    }
    if (!CFStringGetCString(identifier, out, (CFIndex)cap,
                            kCFStringEncodingUTF8)) {
        out[0] = '\0';
    }
}

/* NSCachesDirectory in the user domain, TMPDIR as the fallback. */
static void mglPcpCachesDirectory(char *out, size_t cap)
{
    const char *home = getenv("HOME");
    if (home && home[0] != '\0') {
        snprintf(out, cap, "%s/Library/Caches", home);
        return;
    }
    const char *tmp = getenv("TMPDIR");
    if (tmp && tmp[0] != '\0') {
        size_t n = strlen(tmp);
        while (n > 1u && tmp[n - 1u] == '/') n--;
        snprintf(out, cap, "%.*s", (int)n, tmp);
        return;
    }
    snprintf(out, cap, "/tmp");
}

/* mkdir -p on `path` itself: the .m called
 * createDirectoryAtPath:withIntermediateDirectories:YES for the MGL directory,
 * so every component including the last one has to be created.  (Creating only
 * the prefixes left the leaf missing, and Metal's serializeToURL then failed
 * with "Invalid URL (domain=MTLBinaryArchiveDomain)" -- the dedicated oracle
 * caught it because the A/B filters BINARY ARCHIVE lines.) */
static void mglPcpCreateDirectories(const char *path)
{
    if (!path) return;
    char scratch[PATH_MAX];
    snprintf(scratch, sizeof(scratch), "%s", path);
    for (char *p = scratch + 1; *p != '\0'; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        (void)mkdir(scratch, 0700);
        *p = '/';
    }
    (void)mkdir(scratch, 0700);
}

int mglPipelineCacheArchiveKey(const void *device, const char *schema,
                               char *out, size_t cap)
{
    if (!out || cap == 0u) {
        return -1;
    }
    out[0] = '\0';

    char caches[PATH_MAX] = {0};
    mglPcpCachesDirectory(caches, sizeof(caches));

    char bundle_id[256] = {0};
    mglPcpBundleIdentifier(bundle_id, sizeof(bundle_id));
    char component[256] = {0};
    if (bundle_id[0] != '\0') {
        mglPcpSanitize(bundle_id, component, sizeof(component));
    } else {
        char process_name[256] = {0};
        mglPcpProcessName(process_name, sizeof(process_name));
        mglPcpSanitize(process_name, component, sizeof(component));
    }

    char dir[PATH_MAX] = {0};
    snprintf(dir, sizeof(dir), "%s/MGL/%s", caches, component);
    mglPcpCreateDirectories(dir);

    uint64_t registry_id = 0u;
    char device_name[256] = {0};
    if (device) {
        (void)mglRenderGetDeviceIdentity(device, &registry_id, device_name,
                                         sizeof(device_name));
    }
    char device_id[256] = {0};
    if (registry_id != 0u) {
        snprintf(device_id, sizeof(device_id), "%016llx",
                 (unsigned long long)registry_id);
    } else {
        char device_name_string[256] = {0};
        if (device_name[0] != '\0') {
            snprintf(device_name_string, sizeof(device_name_string), "%s",
                     device_name);
        } else {
            snprintf(device_name_string, sizeof(device_name_string), "unknown");
        }
        mglPcpSanitize(device_name_string, device_id, sizeof(device_id));
    }

    snprintf(out, cap, "%s/pipeline-%s-cpp-%s.binaryarchive", dir,
             schema ? schema : "v5", device_id);
    return 0;
}

int mglPipelineCacheArchiveExists(const char *path)
{
    struct stat st = {0};
    return (path && stat(path, &st) == 0) ? 1 : 0;
}

int mglPipelineCacheArchiveRemove(const char *path)
{
    if (!path) {
        return 0;
    }
    if (unlink(path) == 0) {
        return 1;
    }
    return errno == ENOENT ? 1 : 0;
}
