/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

//------------------------------------------------------------------------------------------------
// Pure C entry points for the C++ renderer facade.
//
// The GL state layer and Objective-C shell use this header without exposing
// MTL::* types. mgl_render.cpp is the only metal-cpp implementation TU.
//------------------------------------------------------------------------------------------------
#pragma once

#ifndef MGL_RENDER_H
#define MGL_RENDER_H


#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
/* The facade declarations use GLuint/GLenum/GLboolean. */
#include <GL/glcorearb.h>
#include "mgl_render_values.h"

/* Umbrella over the per-subsystem facades.  Each mgl_render_api_<x>.h defines
 * the value layouts and declares the entries of one subsystem; include the
 * one you need instead of this header. */
#include "mgl_render_api_command.h"
#include "mgl_render_api_binding.h"
#include "mgl_render_api_texture.h"
#include "mgl_render_api_buffer.h"
#include "mgl_render_api_pixel.h"
#include "mgl_render_api_readback.h"
#include "mgl_render_api_draw.h"
#include "mgl_render_api_query.h"
#include "mgl_render_api_lifecycle.h"

#endif
