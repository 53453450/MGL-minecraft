/*
 * Pure-C renderer value state shared by the Objective-C GL orchestration
 * layer and the Metal-cpp backend.  These values are the stable numeric ABI
 * consumed by mgl_render_cpp.cpp; no Metal framework type crosses this header.
 */

#ifndef MGL_RENDER_VALUES_H
#define MGL_RENDER_VALUES_H

enum {
    MGLTextureType1D = 0,
    MGLTextureType1DArray = 1,
    MGLTextureType2D = 2,
    MGLTextureType2DArray = 3,
    MGLTextureType2DMultisample = 4,
    MGLTextureTypeCube = 5,
    MGLTextureTypeCubeArray = 6,
    MGLTextureType3D = 7,
    MGLTextureType2DMultisampleArray = 8,
    MGLTextureTypeTextureBuffer = 9,
};

enum {
    MGLTextureUsageUnknown = 0x0000,
    MGLTextureUsageShaderRead = 0x0001,
    MGLTextureUsageShaderWrite = 0x0002,
    MGLTextureUsageRenderTarget = 0x0004,
    MGLTextureUsagePixelFormatView = 0x0010,
};

enum {
    MGLStorageModeShared = 0,
    MGLStorageModePrivate = 2,
    MGLResourceStorageModeShared = 0,
};

enum {
    MGLLoadActionDontCare = 0,
    MGLLoadActionLoad = 1,
    MGLLoadActionClear = 2,
};

enum {
    MGLStoreActionDontCare = 0,
    MGLStoreActionStore = 1,
    MGLStoreActionMultisampleResolve = 2,
};

enum {
    MGLCompareFunctionNever = 0,
    MGLCompareFunctionLess = 1,
    MGLCompareFunctionEqual = 2,
    MGLCompareFunctionLessEqual = 3,
    MGLCompareFunctionGreater = 4,
    MGLCompareFunctionNotEqual = 5,
    MGLCompareFunctionGreaterEqual = 6,
    MGLCompareFunctionAlways = 7,
};

enum {
    MGLCommandBufferStatusNotEnqueued = 0,
    MGLCommandBufferStatusEnqueued = 1,
    MGLCommandBufferStatusCommitted = 2,
    MGLCommandBufferStatusScheduled = 3,
    MGLCommandBufferStatusCompleted = 4,
    MGLCommandBufferStatusError = 5,
};

enum {
    MGLPrimitiveTypePoint = 0,
    MGLPrimitiveTypeLine = 1,
    MGLPrimitiveTypeLineStrip = 2,
    MGLPrimitiveTypeTriangle = 3,
    MGLPrimitiveTypeTriangleStrip = 4,
};

enum {
    MGLCullModeNone = 0,
    MGLCullModeFront = 1,
    MGLCullModeBack = 2,
};

enum {
    MGLWindingClockwise = 0,
    MGLWindingCounterClockwise = 1,
};

enum {
    MGLDepthClipModeClip = 0,
    MGLDepthClipModeClamp = 1,
};

enum {
    MGLColorWriteMaskNone = 0,
    MGLColorWriteMaskAll = 0x0f,
};

enum {
    MGLPrimitiveTopologyClassUnspecified = 0,
    MGLPrimitiveTopologyClassPoint = 1,
    MGLPrimitiveTopologyClassLine = 2,
    MGLPrimitiveTopologyClassTriangle = 3,
};

enum {
    MGLTessellationPartitionModePow2 = 0,
    MGLTessellationPartitionModeInteger = 1,
    MGLTessellationPartitionModeFractionalOdd = 2,
    MGLTessellationPartitionModeFractionalEven = 3,
};

enum {
    MGLTessellationFactorStepFunctionConstant = 0,
    MGLTessellationFactorStepFunctionPerPatch = 1,
    MGLTessellationFactorStepFunctionPerInstance = 2,
    MGLTessellationFactorStepFunctionPerPatchAndPerInstance = 3,
};

enum {
    MGLTessellationFactorFormatHalf = 0,
};

enum {
    MGLTessellationControlPointIndexTypeNone = 0,
    MGLTessellationControlPointIndexTypeUInt16 = 1,
    MGLTessellationControlPointIndexTypeUInt32 = 2,
};

enum {
    MGLMultisampleDepthResolveFilterSample0 = 0,
    MGLMultisampleStencilResolveFilterSample0 = 0,
};

enum {
    MGLBlendFactorZero = 0,
    MGLBlendFactorOne = 1,
    MGLBlendFactorSourceColor = 2,
    MGLBlendFactorOneMinusSourceColor = 3,
    MGLBlendFactorSourceAlpha = 4,
    MGLBlendFactorOneMinusSourceAlpha = 5,
    MGLBlendFactorDestinationColor = 6,
    MGLBlendFactorOneMinusDestinationColor = 7,
    MGLBlendFactorDestinationAlpha = 8,
    MGLBlendFactorOneMinusDestinationAlpha = 9,
    MGLBlendFactorSourceAlphaSaturated = 10,
    MGLBlendFactorBlendColor = 11,
    MGLBlendFactorOneMinusBlendColor = 12,
    MGLBlendFactorBlendAlpha = 13,
    MGLBlendFactorOneMinusBlendAlpha = 14,
    MGLBlendFactorSource1Color = 15,
    MGLBlendFactorOneMinusSource1Color = 16,
    MGLBlendFactorSource1Alpha = 17,
    MGLBlendFactorOneMinusSource1Alpha = 18,
};

enum {
    MGLBlendOperationAdd = 0,
    MGLBlendOperationSubtract = 1,
    MGLBlendOperationReverseSubtract = 2,
    MGLBlendOperationMin = 3,
    MGLBlendOperationMax = 4,
};

enum {
    MGLVertexFormatInvalid = 0,
    MGLVertexFormatUChar2 = 1,
    MGLVertexFormatUChar3 = 2,
    MGLVertexFormatUChar4 = 3,
    MGLVertexFormatChar2 = 4,
    MGLVertexFormatChar3 = 5,
    MGLVertexFormatChar4 = 6,
    MGLVertexFormatUChar2Normalized = 7,
    MGLVertexFormatUChar3Normalized = 8,
    MGLVertexFormatUChar4Normalized = 9,
    MGLVertexFormatChar2Normalized = 10,
    MGLVertexFormatChar3Normalized = 11,
    MGLVertexFormatChar4Normalized = 12,
    MGLVertexFormatUShort2 = 13,
    MGLVertexFormatUShort3 = 14,
    MGLVertexFormatUShort4 = 15,
    MGLVertexFormatShort2 = 16,
    MGLVertexFormatShort3 = 17,
    MGLVertexFormatShort4 = 18,
    MGLVertexFormatUShort2Normalized = 19,
    MGLVertexFormatUShort3Normalized = 20,
    MGLVertexFormatUShort4Normalized = 21,
    MGLVertexFormatShort2Normalized = 22,
    MGLVertexFormatShort3Normalized = 23,
    MGLVertexFormatShort4Normalized = 24,
    MGLVertexFormatHalf2 = 25,
    MGLVertexFormatHalf3 = 26,
    MGLVertexFormatHalf4 = 27,
    MGLVertexFormatFloat = 28,
    MGLVertexFormatFloat2 = 29,
    MGLVertexFormatFloat3 = 30,
    MGLVertexFormatFloat4 = 31,
    MGLVertexFormatInt = 32,
    MGLVertexFormatInt2 = 33,
    MGLVertexFormatInt3 = 34,
    MGLVertexFormatInt4 = 35,
    MGLVertexFormatUInt = 36,
    MGLVertexFormatUInt2 = 37,
    MGLVertexFormatUInt3 = 38,
    MGLVertexFormatUInt4 = 39,
    MGLVertexFormatInt1010102Normalized = 40,
    MGLVertexFormatUInt1010102Normalized = 41,
    MGLVertexFormatUChar = 45,
    MGLVertexFormatChar = 46,
    MGLVertexFormatUCharNormalized = 47,
    MGLVertexFormatCharNormalized = 48,
    MGLVertexFormatUShort = 49,
    MGLVertexFormatShort = 50,
    MGLVertexFormatUShortNormalized = 51,
    MGLVertexFormatShortNormalized = 52,
    MGLVertexFormatHalf = 53,
};

#endif /* MGL_RENDER_VALUES_H */
