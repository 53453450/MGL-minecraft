/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

//------------------------------------------------------------------------------------------------
// Shared metal-cpp includes and Objective-C device bridge.
//
// The private implementation macros may be defined in exactly one translation
// unit. mgl_render_cpp.cpp owns them; all other C++ files include declarations.
//
// C callers use the opaque interfaces in mgl_render_cpp.h and mgl_air_loader.h.
//------------------------------------------------------------------------------------------------
#pragma once

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

namespace mgl {

// Wraps an existing id<MTLDevice>. The C++ renderer retains and releases its
// own reference independently from the Objective-C owner.
MTL::Device* wrapDevice(void* objcDevice);

} // namespace mgl
