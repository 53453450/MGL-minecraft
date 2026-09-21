/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_PIXEL_H
#define MGL_RENDER_PIXEL_H

#include <stdint.h>

float mglReadbackMissingChannelFloat(int src_channel_idx);
float mglRead16or32SourceFloat(const uint8_t* s, int idx, int is16u, int is16s, int is16f);
uint8_t mglExpandUNormBitsTo8(uint32_t value, uint32_t bits);
uint32_t mglRenderDepth24Stencil8ToFloatBits(const uint8_t* src);
uint32_t mglRenderDepthUint32ToFloatBits(uint32_t raw);
uint32_t mglRenderDepth24ToFloatBits(const uint8_t* src);
uint8_t mglRenderResolveR8SnormSwizzledComponent(uint32_t swizzle, uint8_t red);
uint16_t mglRenderResolveR16UnormSwizzledComponent(uint32_t swizzle, uint16_t red);
uint16_t mglRenderResolveR16SnormSwizzledComponent(uint32_t swizzle, int16_t red);
uint16_t mglRenderResolveR16FloatSwizzledComponent(uint32_t swizzle, uint16_t red);
uint32_t mglRenderResolveR32FloatSwizzledComponent(uint32_t swizzle, uint32_t red);
int64_t mglRenderResolveIntegerSwizzledComponent( uint32_t swizzle, int64_t red, int64_t green, int64_t blue, int64_t alpha, uint32_t components);
int64_t mglRenderReadIntegerTexelComponent( const uint8_t* texel, uint32_t component, uint32_t component_bytes, int is_signed);
void mglRenderWriteIntegerTexelComponent( uint8_t* texel, uint32_t component, uint32_t component_bytes, int is_signed, int64_t value);


int mglReadbackFormatChannelMap(uint32_t format, int* slots,
                                          int src_idx[4]);
uint32_t mglSizeForType(uint32_t type);
uint32_t mglNumComponentsForFormat(uint32_t format);
int mglPixelTypeIsPacked(uint32_t type);
int mglReadbackRGB10A2TypeAccepted(uint32_t type);
int mglReadbackRG11B10TypeAccepted(uint32_t type);
int mglReadback16or32TypeAccepted(uint32_t type);
int mglReadbackUnorm8ScalarTypeAccepted(uint32_t type);
int mglReadbackUnorm8PackedTypeAccepted(uint32_t type);
int mglRenderIntegerFormatLayout(
    uint32_t internal_format, uint32_t* out_components,
    uint32_t* out_component_bytes, int* out_signed);
int mglRenderPixelFormatMatchesSwizzleBakeStorage(
    uint32_t internal_format, uint32_t storage_pixel_format);
int32_t mglRenderResolveR8IntegerSwizzledComponent(
    uint32_t swizzle, int32_t red, int is_signed);

#endif
