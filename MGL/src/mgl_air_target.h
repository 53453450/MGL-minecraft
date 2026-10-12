/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_target.h - the one description of what the compiler targets.
 *
 * The same target is baked into five places, by two different serializers:
 *
 *   LLVM module   target triple                 air64_v28-apple-macosx26.0.0
 *   LLVM module   !air.version                  2.8.0
 *   LLVM module   !air.language_version         Metal 4.0.0
 *   MTLB record   VERS tag (per function)       air 2.8, language 4.0
 *   MTLB header   platform/format/target/OS     0x8001, 2.9, 0x81, 26.0
 *
 * Metal accepts the library only while those agree, and the mismatch used to
 * be reachable by editing any one of five literals: the failure surfaces as
 * `newLibrary` returning nil inside a shader compile, far from the edit.  All
 * five now derive from `mglAirTarget()`.
 *
 * Note the two version numbers that are *not* the same thing: the AIR language
 * version (`air.version`, 2.8) and the MTLB container file-format version
 * (format 2.9).  `container_major/minor` keeps them separate on purpose.
 *
 * This header deliberately depends on nothing but the standard library - the
 * MTLB container serializer includes it and must not pull in LLVM IR headers -
 * so the module-metadata writer is declared against an incomplete
 * `llvm::Module` and defined in the .cpp.
 */

#ifndef MGL_AIR_TARGET_H
#define MGL_AIR_TARGET_H

#include <stddef.h>
#include <stdint.h>

#include <string>

namespace llvm {
class Module;
}

namespace mgl {

struct AirTarget {
    /* !air.language_version's leading string operand. */
    const char *language;

    /* !air.version: AIR language version. */
    uint32_t air_major, air_minor, air_patch;

    /* !air.language_version's trailing version operands. */
    uint32_t language_major, language_minor, language_patch;

    /* Deployment target.  The triple spells it in full
     * (macosx<major>.<minor>.<patch>); the container header has a field for
     * the major and the minor only. */
    uint32_t os_major, os_minor, os_patch;

    /* MTLB container file-format version, written to the file header.  This is
     * a different number from air_major/air_minor and must stay that way. */
    uint32_t container_major, container_minor;

    /* MTLB file header platform and target fields. */
    uint32_t platform;
    uint32_t target;
};

/* The target this build emits.  Every call site reads from here. */
const AirTarget &mglAirTarget(void);

/* "air64_v<air_major><air_minor>-apple-macosx<os_major>.<os_minor>.<os_patch>".
 * The AIR version is encoded as one digit per component, so the components
 * must be single digits - mglAirTargetIsConsistent checks that. */
std::string mglAirTargetTriple(const AirTarget &t);

/* Write !air.version and !air.language_version onto a module.  Defined in the
 * .cpp so the LLVM IR types stay out of this header. */
void mglAirTargetApplyModuleMetadata(llvm::Module &module, const AirTarget &t);

/* Check the derivations the loader depends on.  Returns true when they hold,
 * otherwise false with a one-line reason in `err`.  The checks that matter are
 * the ones about information the container cannot carry - a non-zero patch
 * component or a too-wide field would make the serialized bytes disagree with
 * the module metadata without any visible edit. */
bool mglAirTargetIsConsistent(const AirTarget &t, char *err, size_t errcap);

} /* namespace mgl */

#endif /* MGL_AIR_TARGET_H */
