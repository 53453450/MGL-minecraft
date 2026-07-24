// MGLRenderer+ProgramBinding.m
// Program SPIR-V resource binding queries (count, binding, GL binding,
// required size, Metal slot, texture type/location) extracted from
// MGLRenderer.m.  All methods are read-only queries over the active
// program's spirv_resources_list; they hold no Metal state of their own.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+ProgramBinding_Private.h"

@implementation MGLRenderer (ProgramBinding)

#pragma mark programs
- (int) getProgramBindingCount: (int) stage type: (int) type
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramBindingCount", stage);
        return 0;
    }
    switch(type)
    {
        case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
        case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
        case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
        case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
        case SPVC_RESOURCE_TYPE_STAGE_INPUT:
        case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
        case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
            break;

        default:
            NSLog(@"MGL ERROR: Unknown resource type %d in getProgramBindingCount (stage=%d)", type, stage);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (ptr == NULL)
        return 0;

    return ptr->spirv_resources_list[stage][type].count;
}

- (int) getProgramBinding: (int) stage type: (int) type index: (int) index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramBinding", stage);
        return 0;
    }
    switch(type)
    {
       case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
       case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
       case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
       case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
       case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
       case SPVC_RESOURCE_TYPE_STAGE_INPUT:
       case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
       case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
       case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
       case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
       case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
           break;

       default:
            NSLog(@"MGL ERROR: Unknown resource type %d in getProgramBinding (stage=%d)", type, stage);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        NSLog(@"MGL ERROR: getProgramBinding with no current program for stage=%s (name=%u pipeline=%u)",
              mglShaderStageName(stage),
              (unsigned)ctx->active_state->program_name,
              (unsigned)ctx->active_state->var.program_pipeline_binding);
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        NSLog(@"MGL WARNING: getProgramBinding index out of range index=%d count=%d stage=%d type=%d",
              index, count, stage, type);
        return 0;
    }

    return ptr->spirv_resources_list[stage][type].list[index].binding;
}

- (int)getProgramGLBinding:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES || type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        return 0;
    }

    return (int)ptr->spirv_resources_list[stage][type].list[index].gl_binding;
}

- (NSUInteger)getProgramBindingRequiredSize:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }

    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    return (NSUInteger)ptr->spirv_resources_list[stage][type].list[index].required_size;
}

- (NSInteger)getProgramMetalBufferIndexForStage:(int)stage clientBinding:(GLuint)clientBinding
{
    static const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
    };

    Program *ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return (NSInteger)clientBinding;
    }

    for (size_t t = 0; t < (sizeof(resourceTypes) / sizeof(resourceTypes[0])); t++) {
        int type = resourceTypes[t];
        if (type < 0 || type >= _MAX_SPIRV_RES) {
            continue;
        }

        SpirvResourceList *list = &ptr->spirv_resources_list[stage][type];
        for (GLuint i = 0; i < list->count; i++) {
            SpirvResource *res = &list->list[i];
            if (mglShouldSkipStageBufferResource(ptr, stage, type, res)) {
                continue;
            }
            GLuint resourceClientBinding = mglClientBufferBindingForResource(type, res);
            if (resourceClientBinding == clientBinding) {
                return (NSInteger)mglMetalResourceSlot(res);
            }
        }
    }

    return -1;
}

- (MTLTextureType)getProgramDeclaredTextureType:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    switch ((SpvDim)res->image_dim) {
        case SpvDim1D:
            return res->image_arrayed ? MTLTextureType1DArray : MTLTextureType1D;
        case SpvDim2D:
            if (res->image_multisampled) {
                return res->image_arrayed ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
            }
            return res->image_arrayed ? MTLTextureType2DArray : MTLTextureType2D;
        case SpvDim3D:
            return MTLTextureType3D;
        case SpvDimCube:
            return res->image_arrayed ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
        case SpvDimBuffer:
            return MTLTextureTypeTextureBuffer;
        default:
            return 0;
    }
}

- (MTLTextureType)getProgramExpectedTextureType:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    /* Per-resource cache: the MSL string is immutable post-link, so the
     * texture type for a given resource never changes between relinks.
     * Stored directly on SpirvResource to avoid per-draw stringWithFormat
     * + NSDictionary lookup.  Uses a separate valid flag because
     * MTLTextureType1D == 0 on this SDK. */
    MTLTextureType mslType;
    if (res->cached_msl_texture_type_valid) {
        mslType = (MTLTextureType)res->cached_msl_texture_type;
    } else {
        mslType = mglExpectedTextureTypeFromMSL(ptr->spirv[stage].msl_str, res->binding);
        res->cached_msl_texture_type = (uint32_t)mslType;
        res->cached_msl_texture_type_valid = 1u;
    }

    MTLTextureType spirvType = 0;
    switch ((SpvDim)res->image_dim) {
        case SpvDim1D:
            spirvType = res->image_arrayed ? MTLTextureType1DArray : MTLTextureType1D;
            break;
        case SpvDim2D:
            if (res->image_multisampled) {
                spirvType = res->image_arrayed ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
            } else {
                spirvType = res->image_arrayed ? MTLTextureType2DArray : MTLTextureType2D;
            }
            break;
        case SpvDim3D:
            spirvType = MTLTextureType3D;
            break;
        case SpvDimCube:
            spirvType = res->image_arrayed ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
            break;
        case SpvDimBuffer:
            spirvType = MTLTextureTypeTextureBuffer;
            break;
        default:
            spirvType = 0;
            break;
    }

    if (mslType != 0 && mslType != spirvType) {
        static uint64_t s_mslTextureTypeOverrideCount = 0;
        uint64_t hit = ++s_mslTextureTypeOverrideCount;
        if (hit <= 8ull || (hit % 2048ull) == 0ull) {
            NSLog(@"MGL TEX EXPECT override from MSL stage=%d type=%d index=%d binding=%u name=%s spirvType=%lu mslType=%lu imageDim=%u hit=%llu",
                  stage,
                  type,
                  index,
                  (unsigned)res->binding,
                  res->name ? res->name : "(null)",
                  (unsigned long)spirvType,
                  (unsigned long)mslType,
                  (unsigned)res->image_dim,
                  (unsigned long long)hit);
        }
        return mslType;
    }

    return mslType ? mslType : spirvType;
}

- (MGLTextureDataKind)getProgramExpectedTextureDataKind:(int)stage type:(int)type index:(int)index
{
    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return MGLTextureDataKindUnknown;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return MGLTextureDataKindUnknown;
    }

    Program *ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return MGLTextureDataKindUnknown;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return MGLTextureDataKindUnknown;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    /* Per-resource cache: 0 = uncached (MGLTextureDataKindUnknown == 0,
     * but we cache the *resolved* kind, which is never Unknown — see
     * fallback below — so 0 reliably means "not yet cached"). */
    if (res->cached_msl_data_kind != 0u) {
        return (MGLTextureDataKind)res->cached_msl_data_kind;
    }

    MGLTextureDataKind mslKind =
        mglExpectedTextureDataKindFromMSL(ptr->spirv[stage].msl_str, res->binding);
    MGLTextureDataKind resolvedKind =
        mslKind != MGLTextureDataKindUnknown ? mslKind : MGLTextureDataKindFloat;
    res->cached_msl_data_kind = (uint32_t)resolvedKind;
    return resolvedKind;
}

- (NSUInteger)getProgramBindingRequiredSizeForStage:(int)stage clientBinding:(GLuint)clientBinding
{
    static const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
    };

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }

    NSUInteger required = 0;
    for (size_t t = 0; t < (sizeof(resourceTypes) / sizeof(resourceTypes[0])); t++) {
        int type = resourceTypes[t];
        int count = [self getProgramBindingCount:stage type:type];
        for (int i = 0; i < count; i++) {
            Program *program = mglResolveProgramForStageFromState(ctx, stage);
            if (!program || type < 0 || type >= _MAX_SPIRV_RES ||
                i < 0 || i >= (int)program->spirv_resources_list[stage][type].count) {
                continue;
            }

            SpirvResource *resource = &program->spirv_resources_list[stage][type].list[i];
            if (mglShouldSkipStageBufferResource(program, stage, type, resource)) {
                continue;
            }

            GLuint resourceClientBinding =
                mglClientBufferBindingForResource(type,
                                                  resource);
            if (resourceClientBinding != clientBinding) {
                continue;
            }

            NSUInteger candidate = [self getProgramBindingRequiredSize:stage type:type index:i];
            if (candidate > required) {
                required = candidate;
            }
        }
    }

    return required;
}

- (int) getProgramLocation: (int) stage type: (int) type index: (int) index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramLocation", stage);
        return 0;
    }
    switch(type)
    {
       case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
       case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
       case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
       case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
       case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
       case SPVC_RESOURCE_TYPE_STAGE_INPUT:
       case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
       case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
           break;

       default:
            NSLog(@"MGL WARNING: unsupported SPIRV-Cross resource type %d in getProgramLocation", type);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        NSLog(@"MGL ERROR: getProgramLocation with no current program for stage=%s (name=%u pipeline=%u)",
              mglShaderStageName(stage),
              (unsigned)ctx->active_state->program_name,
              (unsigned)ctx->active_state->var.program_pipeline_binding);
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        NSLog(@"MGL WARNING: getProgramLocation index out of range index=%d count=%d stage=%d type=%d",
              index, count, stage, type);
        return 0;
    }

    return ptr->spirv_resources_list[stage][type].list[index].location;
}

@end
