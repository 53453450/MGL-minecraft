/*
 * mgl_spirv_compile.h
 * MGL
 *
 * SPIR-V Compilation Pipeline Subsystem (Category B).
 *
 * Extracted from program.c.  Contains the SPIR-V → MSL compilation pipeline:
 * MSL binding map helpers, MSL string manipulation utilities, MSL fix/patch
 * functions, MSL uniform buffer packing helpers, MSL patch pipeline step
 * functions, compile-state & link helpers, and the top-level non-static
 * entry points (error_callback, addShadersToProgram, parseSPIRVShaderToMetal,
 * mglCompileMSLCaptureVariant, mglProgramPipelinePerVertexCompatible).
 *
 * Dependencies: glm_context.h (Program, GLMContext, SpirvResource, GL types) +
 * msl_patch_pipeline.h (MSLPatchContext) + glslang_c_interface.h
 * (glslang_program_t) + mgl_uniform_reflection.h (Category A helpers).
 */

#ifndef mgl_spirv_compile_h
#define mgl_spirv_compile_h

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "glcorearb.h"
#include "glm_context.h"

#include <glslang_c_interface.h>

#include "msl_patch_pipeline.h"
#include "mgl_uniform_reflection.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Macros moved from program.c ---- */

#ifndef MGL_VERBOSE_PROGRAM_LOGS
#define MGL_VERBOSE_PROGRAM_LOGS 0
#endif

#define MGL_TEXEL_BUFFER_TEXTURE_WIDTH 4096u
#define MGL_INTERNAL_UNIFORM_BUFFER_NAME_BASE 0xf0000000u
#define MGL_FRAG_COORD_PARAMS_MSL_NAME "_mglFragCoordParams"
#define MGL_FRAG_COORD_PARAMS_BUFFER_INDEX 30

/* ---- Types moved from program.c ---- */

typedef enum MGLMSLBindingKind {
    MGL_MSL_BINDING_TEXTURE,
    MGL_MSL_BINDING_BUFFER,
    MGL_MSL_BINDING_SAMPLER
} MGLMSLBindingKind;

typedef struct MGLMSLBindingEntry {
    MGLMSLBindingKind kind;
    GLuint index;
    const char *segment;
    size_t segment_len;
} MGLMSLBindingEntry;

#define MGL_MSL_BINDING_MAP_MAX 512u

typedef struct MGLMSLBindingMap {
    MGLMSLBindingEntry entries[MGL_MSL_BINDING_MAP_MAX];
    size_t count;
} MGLMSLBindingMap;

typedef struct MGLMSLStructLayout {
    char name[128];
    size_t size;
    size_t align;
} MGLMSLStructLayout;

typedef struct MGLMSLStructDeps {
    char name[128];
    char member_types[64][128];
    unsigned member_type_count;
} MGLMSLStructDeps;

/* ---- Category D forward declaration (stays in program.c, made non-static) ---- */

GLuint mglProgramAttachedShaderCount(Program *program, GLuint stage);

/* ---- Group B.1: MSL Binding Map Helpers ---- */

bool mglMSLIdentifierChar(char c);
GLboolean mglSegmentContainsIdentifier(const char *segment,
                                       size_t segment_len,
                                       const char *name);
const char *mglPreviousMSLArgumentBoundary(const char *msl, const char *attribute);
const char *mglNextMSLArgumentBoundary(const char *attribute);
GLboolean mglParseMSLBindingAttribute(const char *attribute,
                                      const char *prefix,
                                      GLuint *index_out);
void mglMSLBindingMapAdd(MGLMSLBindingMap *map,
                         MGLMSLBindingKind kind,
                         GLuint index,
                         const char *segment_start,
                         const char *segment_end);
void mglBuildMSLBindingMap(const char *msl, MGLMSLBindingMap *map);
GLboolean mglFindMSLResourceIndexInMap(const MGLMSLBindingMap *map,
                                       MGLMSLBindingKind kind,
                                       const char *name,
                                       GLuint *index_out);
GLint mglFindMSLResourceArraySizeInMap(const MGLMSLBindingMap *map,
                                       MGLMSLBindingKind kind,
                                       const char *name);
GLboolean mglMSLBufferSlotConflicts(Program *pptr, int stage, GLuint slot);
void applyMSLResourceBindings(Program *pptr, int stage, char **msl_ptr);

/* ---- Group B.2: MSL String Manipulation Utilities ---- */

void replace_all_substr(char **pstr, const char *from, const char *to);
GLboolean mglReplaceMSLIdentifier(char **msl_ptr,
                                  const char *from,
                                  const char *to);
GLboolean mglReplaceMSLIdentifierBeforeChar(char **msl_ptr,
                                            const char *from,
                                            const char *to,
                                            char required_after);
GLboolean mglInsertStringAt(char **pstr, const char *position, const char *insertion);
GLboolean mglFindMSLIdentifierBefore(const char *start,
                                     const char *pos,
                                     char *out,
                                     size_t out_size);
size_t count_substr(const char *str, const char *needle);
int mglMSLDoubleReplacementLength(const char *token);
const char *mglFindMSLEntryParameterOpen(const char *msl);
const char *mglFindMSLEntryParameterClose(const char *msl);

/* ---- Group B.3: MSL Fix/Patch Functions ---- */

void applyMSLFragCoordOriginFix(int stage, char **msl_ptr);
GLboolean mglInjectMSLPointSizeParams(char **msl_ptr);
void mglInjectMSLPointSizeBuiltin(int stage, char **msl_ptr);
void mglFixMSLImage2DRectImageSize(char **msl_ptr);
void mglFixMSLTcsStageIn(char **msl_ptr);
void mglEnsurePatchId3Param(char **msl_ptr);
void mglFixMSLTessCullDistanceOutputs(char **msl_ptr);
void mglFixMSLTesAsComputeKernel(Program *program, char **msl_ptr);
void mglInjectMSLAtomicCounterArguments(Program *program, int stage, char **msl_ptr);
void mglFixMSLPlainStructPointerArrayAccess(Program *program,
                                            int stage,
                                            char **msl);
void mglLowerMSLDoubleTypesToFloat(char **pstr);
GLboolean mglBuildMSLResourceNameVariant(const char *name,
                                         unsigned variant,
                                         char *out,
                                         size_t out_size);
GLboolean mglFindMSLUserLocationForName(const char *msl, const char *name, GLuint *location_out);
GLboolean mglFindMSLUserLocationForResourceName(const char *msl,
                                                const char *name,
                                                GLuint *location_out,
                                                char *matched_name,
                                                size_t matched_name_size);
GLboolean mglReplaceMSLUserLocationForResourceName(char **msl_ptr,
                                                   const char *name,
                                                   GLuint current_location,
                                                   GLuint desired_location,
                                                   char *matched_name,
                                                   size_t matched_name_size);
GLboolean mglProgramHasPassthroughGeometryShader(Program *ptr);
GLboolean mglShaderSourceHasToken(const char *start, const char *end, const char *token);

/* ---- Group B.4: MSL Uniform Buffer Packing Helpers ---- */

size_t mglMSLVectorCSize(unsigned components);
size_t mglStd140VectorAlign(unsigned components);
GLboolean mglMSLUniformTypeLayout(const char *trimmed,
                                  size_t *c_size_out,
                                  size_t *std140_align_out,
                                  size_t *actual_align_out);
GLboolean mglMSLNameInList(char names[][128], unsigned count, const char *name);
GLboolean mglMSLAddName(char names[][128], unsigned *count, unsigned capacity, const char *name);
GLboolean mglGLSLBlockDeclContainsToken(const char *glsl_src,
                                        const char *block_name,
                                        const char *token);
GLboolean mglMSLTokenLooksStructLike(const char *token);
MGLMSLStructDeps *mglMSLFindStructDeps(MGLMSLStructDeps *deps,
                                       unsigned count,
                                       const char *name);
void mglMSLCollectStructDeps(const char *msl,
                             MGLMSLStructDeps *deps,
                             unsigned *dep_count,
                             unsigned dep_capacity);
size_t mglMSLDeclaratorArrayCount(const char *trimmed);
const MGLMSLStructLayout *mglMSLFindStructLayout(const MGLMSLStructLayout *layouts,
                                                 unsigned count,
                                                 const char *name);
GLboolean mglMSLStructMemberLayout(const char *trimmed,
                                   const MGLMSLStructLayout *layouts,
                                   unsigned layout_count,
                                   size_t *c_size_out,
                                   size_t *std140_align_out,
                                   size_t *actual_align_out);
void applyMSLUniformBufferPacking(Program *pptr, int stage);

/* ---- Group B.5: MSL Patch Pipeline Step Functions ---- */

GLboolean mglPatchRemoveRestrict(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchFixSamplerShadowing(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchFixUnknownTextureType(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchStripThreadConstRef(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchRenameLengthSquared(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchLowerDoubleTypes(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchFixEndPortalLayer(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchFullscreenFBYFlip(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchFragCoordOriginFix(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchFixPlainStructPointerArray(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchInjectAtomicCounterArgs(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchApplyResourceBindings(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchInjectPointSizeBuiltin(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchFixImage2DRectImageSize(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchTesAsComputeKernel(MSLPatchContext *ctx, char **msl_ptr);
GLboolean mglPatchTcsStageInFix(MSLPatchContext *ctx, char **msl_ptr);

/* ---- Group B.6: Compile State & Link Helpers ---- */

void clearStageCompileState(Program *pptr, int stage);
GLboolean mglProgramPerVertexSignature(Program *program, int stage, unsigned *signature);
GLboolean mglLinkedProgramPerVertexCompatible(Program *program);
GLint mglDefaultAttribLocationForName(const char *name);
GLint mglProgramVertexInputOrdinal(Program *pptr, const char *name);
GLboolean mglProgramHasVertexInputNamed(Program *pptr, const char *name);
GLint mglContextualDefaultAttribLocationForName(Program *pptr, const char *name);
GLint mglDesiredAttribLocationForName(Program *pptr, const char *name);
void applyVertexInputLocations(Program *pptr);
void applyMultiDimArrayUniformNames(Program *pptr);
void applyFragmentOutputLocationIndices(Program *pptr);
void alignFragmentInputLocationsToVertexOutputs(Program *pptr);
GLboolean mglSpirvVaryingTypesCompatible(const SpirvResource *a,
                                         const SpirvResource *b);
SpirvResource *mglFindVaryingByName(SpirvResourceList *list,
                                    const char *name,
                                    const SpirvResource *type_peer);
SpirvResource *mglFindVaryingByLocation(SpirvResourceList *list,
                                        GLuint location,
                                        const SpirvResource *type_peer);
void mglBridgeSkippedGeometryShaderVaryings(Program *pptr);
void mglApplyPlainUniformInitializers(GLMContext ctx, Program *program, int stage);
bool compileStageFromLinkedProgram(GLMContext ctx, Program *pptr, glslang_program_t *glsl_program, int stage);

/* ---- Group B.7: Non-static Entry Points ---- */

void error_callback(void *userdata, const char *error);
void addShadersToProgram(GLMContext ctx, Program *pptr, glslang_program_t *glsl_program);
char *parseSPIRVShaderToMetal(GLMContext ctx, Program *ptr, int stage);
char *mglCompileMSLCaptureVariant(GLMContext ctx, Program *ptr, int stage);
GLboolean mglProgramPipelinePerVertexCompatible(Program *const *stage_programs);

#ifdef __cplusplus
}
#endif

#endif /* mgl_spirv_compile_h */
