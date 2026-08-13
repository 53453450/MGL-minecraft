// MGLRenderer+ProgramBinding.m
// Program shader resource binding queries (count, binding, GL binding,
// required size, Metal slot, texture type/location) extracted from
// MGLRenderer.m.  All methods are read-only queries over the active
// program's shader_resources_list; they hold no Metal state of their own.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+ProgramBinding_Private.h"

/* === AIR-reflected texture type / data kind helpers === */
MTLTextureType mglDeclaredTextureTypeFromResource(const MGLShaderResource *res)
{
    if (!res) {
        return 0;
    }
    switch ((MGLImageDimension)res->image_dim) {
        case MGL_IMAGE_DIM_1D:
            return res->image_arrayed ? MTLTextureType1DArray : MTLTextureType1D;
        case MGL_IMAGE_DIM_2D:
            if (res->image_multisampled) {
                return res->image_arrayed ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
            }
            return res->image_arrayed ? MTLTextureType2DArray : MTLTextureType2D;
        case MGL_IMAGE_DIM_3D:
            return MTLTextureType3D;
        case MGL_IMAGE_DIM_CUBE:
            return res->image_arrayed ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
        case MGL_IMAGE_DIM_BUFFER:
            return MTLTextureTypeTextureBuffer;
        default:
            return 0;
    }
}

MTLTextureType mglExpectedTextureTypeForResource(Program *program, int stage, MGLShaderResource *res)
{
    if (!program || !res || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }

    return mglDeclaredTextureTypeFromResource(res);
}

MGLTextureDataKind mglExpectedTextureDataKindForResource(Program *program, int stage, MGLShaderResource *res)
{
    if (!program || !res || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return MGLTextureDataKindUnknown;
    }

    return res->texture_data_kind != MGL_SHADER_TEXTURE_DATA_UNKNOWN
        ? (MGLTextureDataKind)res->texture_data_kind
        : MGLTextureDataKindFloat;
}

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
        case _UNIFORM_BUFFER_RES:
        case _UNIFORM_CONSTANT_RES:
        case _STORAGE_BUFFER_RES:
        case _ATOMIC_COUNTER_RES:
        case _PUSH_CONSTANT_RES:
        case _STAGE_INPUT_RES:
        case _STAGE_OUTPUT_RES:
        case _SAMPLED_IMAGE_RES:
        case _SEPARATE_IMAGE_RES:
        case _SEPARATE_SAMPLERS_RES:
        case _STORAGE_IMAGE_RES:
            break;

        default:
            NSLog(@"MGL ERROR: Unknown resource type %d in getProgramBindingCount (stage=%d)", type, stage);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (ptr == NULL)
        return 0;

    return ptr->shader_resources_list[stage][type].count;
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
       case _UNIFORM_BUFFER_RES:
       case _UNIFORM_CONSTANT_RES:
       case _STORAGE_BUFFER_RES:
       case _ATOMIC_COUNTER_RES:
       case _PUSH_CONSTANT_RES:
       case _STAGE_INPUT_RES:
       case _STAGE_OUTPUT_RES:
       case _SAMPLED_IMAGE_RES:
       case _SEPARATE_IMAGE_RES:
       case _SEPARATE_SAMPLERS_RES:
       case _STORAGE_IMAGE_RES:
           break;

       default:
            NSLog(@"MGL ERROR: Unknown resource type %d in getProgramBinding (stage=%d)", type, stage);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        NSLog(@"MGL ERROR: getProgramBinding with no current program for stage=%s (name=%u pipeline=%u)",
              mglShaderStageName(stage),
              (unsigned)MGL_STATE(ctx)->program_name,
              (unsigned)MGL_STATE(ctx)->var.program_pipeline_binding);
        return 0;
    }

    int count = ptr->shader_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        NSLog(@"MGL WARNING: getProgramBinding index out of range index=%d count=%d stage=%d type=%d",
              index, count, stage, type);
        return 0;
    }

    return ptr->shader_resources_list[stage][type].list[index].binding;
}

- (int)getProgramGLBinding:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES || type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }

    int count = ptr->shader_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        return 0;
    }

    return (int)ptr->shader_resources_list[stage][type].list[index].gl_binding;
}

- (NSUInteger)getProgramBindingRequiredSize:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }

    if (index < 0 || index >= (int)ptr->shader_resources_list[stage][type].count) {
        return 0;
    }

    return (NSUInteger)ptr->shader_resources_list[stage][type].list[index].required_size;
}

- (NSInteger)getProgramMetalBufferIndexForStage:(int)stage clientBinding:(GLuint)clientBinding
{
    static const int resourceTypes[] = {
        _UNIFORM_BUFFER_RES,
        _UNIFORM_CONSTANT_RES,
        _STORAGE_BUFFER_RES,
        _ATOMIC_COUNTER_RES,
        _PUSH_CONSTANT_RES
    };

    Program *ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return (NSInteger)clientBinding;
    }

    for (size_t t = 0; t < (sizeof(resourceTypes) / sizeof(resourceTypes[0])); t++) {
        int type = resourceTypes[t];
        if (type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
            continue;
        }

        MGLShaderResourceList *list = &ptr->shader_resources_list[stage][type];
        for (GLuint i = 0; i < list->count; i++) {
            MGLShaderResource *res = &list->list[i];
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
    if (type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }
    if (index < 0 || index >= (int)ptr->shader_resources_list[stage][type].count) {
        return 0;
    }

    MGLShaderResource *res = &ptr->shader_resources_list[stage][type].list[index];
    return mglDeclaredTextureTypeFromResource(res);
}

- (MTLTextureType)getProgramExpectedTextureType:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return 0;
    }
    if (index < 0 || index >= (int)ptr->shader_resources_list[stage][type].count) {
        return 0;
    }

    MGLShaderResource *res = &ptr->shader_resources_list[stage][type].list[index];
    return mglExpectedTextureTypeForResource(ptr, stage, res);
}

- (MGLTextureDataKind)getProgramExpectedTextureDataKind:(int)stage type:(int)type index:(int)index
{
    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return MGLTextureDataKindUnknown;
    }
    if (type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
        return MGLTextureDataKindUnknown;
    }

    Program *ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        return MGLTextureDataKindUnknown;
    }
    if (index < 0 || index >= (int)ptr->shader_resources_list[stage][type].count) {
        return MGLTextureDataKindUnknown;
    }

    MGLShaderResource *res = &ptr->shader_resources_list[stage][type].list[index];
    return mglExpectedTextureDataKindForResource(ptr, stage, res);
}

- (NSUInteger)getProgramBindingRequiredSizeForStage:(int)stage clientBinding:(GLuint)clientBinding
{
    static const int resourceTypes[] = {
        _UNIFORM_BUFFER_RES,
        _UNIFORM_CONSTANT_RES,
        _STORAGE_BUFFER_RES,
        _ATOMIC_COUNTER_RES,
        _PUSH_CONSTANT_RES
    };

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }

    /* Resolve the program once and read shader_resources_list directly. */
    Program *program = mglResolveProgramForStageFromState(ctx, stage);
    if (!program) {
        return 0;
    }

    NSUInteger required = 0;
    for (size_t t = 0; t < (sizeof(resourceTypes) / sizeof(resourceTypes[0])); t++) {
        int type = resourceTypes[t];
        if (type < 0 || type >= MGL_MAX_SHADER_RESOURCES) {
            continue;
        }

        MGLShaderResourceList *list = &program->shader_resources_list[stage][type];
        for (GLuint i = 0; i < list->count; i++) {
            MGLShaderResource *resource = &list->list[i];
            if (mglShouldSkipStageBufferResource(program, stage, type, resource)) {
                continue;
            }

            GLuint resourceClientBinding =
                mglClientBufferBindingForResource(type, resource);
            if (resourceClientBinding != clientBinding) {
                continue;
            }

            NSUInteger candidate = (NSUInteger)resource->required_size;
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
       case _UNIFORM_BUFFER_RES:
       case _UNIFORM_CONSTANT_RES:
       case _STORAGE_BUFFER_RES:
       case _ATOMIC_COUNTER_RES:
       case _PUSH_CONSTANT_RES:
       case _STAGE_INPUT_RES:
       case _SAMPLED_IMAGE_RES:
       case _STORAGE_IMAGE_RES:
           break;

       default:
            NSLog(@"MGL WARNING: unsupported shader resource type %d in getProgramLocation", type);
            return 0;
    }

    ptr = mglResolveProgramForStageFromState(ctx, stage);
    if (!ptr) {
        NSLog(@"MGL ERROR: getProgramLocation with no current program for stage=%s (name=%u pipeline=%u)",
              mglShaderStageName(stage),
              (unsigned)MGL_STATE(ctx)->program_name,
              (unsigned)MGL_STATE(ctx)->var.program_pipeline_binding);
        return 0;
    }

    int count = ptr->shader_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        NSLog(@"MGL WARNING: getProgramLocation index out of range index=%d count=%d stage=%d type=%d",
              index, count, stage, type);
        return 0;
    }

    return ptr->shader_resources_list[stage][type].list[index].location;
}

@end
