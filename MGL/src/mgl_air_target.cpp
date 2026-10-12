/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_target.cpp
 * The single definition of the AIR / Metal / MTLB container target (see
 * mgl_air_target.h).  Both serializers - the LLVM module metadata in
 * mgl_air_backend.cpp and the MTLB container in mgl_metallib_writer.cpp - read
 * their version numbers from mglAirTarget() so the two artifacts cannot drift
 * apart.
 */

#include "mgl_air_target.h"

#include <stdio.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

namespace mgl {

namespace {

/* The numbers live here and nowhere else.
 *
 * Target triple and OS deployment target go together: the triple says
 * `air64_v28-apple-macosx26.0.0`, the container header says OS 26.0, and
 * !air.language_version says Metal 4.0.  The MTLB file-format version (2.9) is
 * a *serialization* version and is intentionally not the AIR version (2.8);
 * see mglAirTargetIsConsistent. */
const AirTarget kAirTarget = {
    /* language         */ "Metal",
    /* air              */ 2u, 8u, 0u,
    /* language         */ 4u, 0u, 0u,
    /* os               */ 26u, 0u, 0u,
    /* container format */ 2u, 9u,
    /* platform / target */ 0x8001u, 0x81u,
};

} /* namespace */

const AirTarget &mglAirTarget(void) {
    return kAirTarget;
}

std::string mglAirTargetTriple(const AirTarget &t) {
    char buf[96];
    snprintf(buf, sizeof(buf), "air64_v%u%u-apple-macosx%u.%u.%u",
             (unsigned)t.air_major, (unsigned)t.air_minor,
             (unsigned)t.os_major, (unsigned)t.os_minor,
             (unsigned)t.os_patch);
    return std::string(buf);
}

void mglAirTargetApplyModuleMetadata(llvm::Module &module, const AirTarget &t) {
    llvm::LLVMContext &ctx = module.getContext();
    auto u32 = [&ctx](uint32_t value) {
        return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), value));
    };

    llvm::NamedMDNode *ver = module.getOrInsertNamedMetadata("air.version");
    ver->addOperand(llvm::MDNode::get(
        ctx, {u32(t.air_major), u32(t.air_minor), u32(t.air_patch)}));

    llvm::NamedMDNode *lver =
        module.getOrInsertNamedMetadata("air.language_version");
    lver->addOperand(llvm::MDNode::get(
        ctx, {llvm::MDString::get(ctx, t.language), u32(t.language_major),
              u32(t.language_minor), u32(t.language_patch)}));
}

bool mglAirTargetIsConsistent(const AirTarget &t, char *err, size_t errcap) {
    auto reject = [&](const char *reason) {
        if (err && errcap) {
            snprintf(err, errcap, "%s", reason);
        }
        return false;
    };

    if (!t.language || !t.language[0]) {
        return reject("!air.language_version has no language name");
    }
    /* The triple encodes both AIR components into a single digit group
     * ("v28"), so anything wider would silently address a different target. */
    if (t.air_major > 9u || t.air_minor > 9u) {
        return reject("the triple packs the AIR version as v<major><minor>; "
                      "each component must be a single digit");
    }
    /* The container header stores osMajor/osMinor as u16 and the record VERS
     * tag stores air/language major+minor as u16, but neither carries a patch
     * component.  A non-zero patch therefore makes the serialized bytes
     * disagree with the module metadata while every literal still looks
     * right. */
    if (t.air_patch != 0u || t.language_patch != 0u) {
        return reject("the MTLB VERS tag carries major/minor only, so a "
                      "non-zero AIR or language patch would leave the "
                      "container disagreeing with the module metadata");
    }
    if (t.os_patch != 0u) {
        return reject("the MTLB header carries major/minor only, so a non-zero "
                      "OS patch would leave the container disagreeing with the "
                      "module's target triple");
    }
    if (t.container_major > 0xFFFFu || t.container_minor > 0xFFFFu) {
        return reject("the MTLB file-format version is written as two u16");
    }
    if (t.platform > 0xFFFFu) {
        return reject("the MTLB header platform field is written as a u16");
    }
    if (t.target > 0xFFu) {
        return reject("the MTLB header target field is written as a u8");
    }
    if (t.os_major > 0xFFFFu || t.os_minor > 0xFFFFu) {
        return reject("the MTLB header OS fields are written as two u16");
    }
    return true;
}

} /* namespace mgl */
