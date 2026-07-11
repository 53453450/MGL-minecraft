/*
 * mgl_uniform_reflection.h
 * MGL
 *
 * Uniform/Attribute Reflection Subsystem (Category A).
 *
 * Pure reflection helpers extracted from program.c.  None of these functions
 * take a GLMContext ctx parameter; they operate on Program*, SpirvResource*,
 * spvc_compiler/spvc_type handles, or raw GLSL/SPIR-V data.
 *
 * Dependencies: glm_context.h (Program, SpirvResource, SpirvUBOMember,
 * SpirvResourceList, GL types, MAX_BINDABLE_BUFFERS, TEXTURE_UNITS) +
 * spirv_cross_c.h (spvc_*) + spirv.h (SpvDecoration*, SpvId).
 */

#ifndef mgl_uniform_reflection_h
#define mgl_uniform_reflection_h

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <mach/vm_types.h>

#include "spirv_cross_c.h"
#include "spirv.h"

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Macros moved from program.c ---- */

#define MGL_ACTIVE_MAX_PATHS   512
#define MGL_ACTIVE_MAX_DEPTH   32

/* ---- Types moved from program.c ---- */

typedef struct {
    GLuint  indices[MGL_ACTIVE_MAX_DEPTH];
    GLuint  len;
} MGLActivePath;

typedef struct {
    MGLActivePath  paths[MGL_ACTIVE_MAX_PATHS];
    GLuint         count;
} MGLActivePathSet;

/* ---- Inline helper shared by Category A and B ---- */

static inline size_t mglRoundUpSize(size_t value, size_t alignment)
{
    return alignment ? ((value + alignment - 1) / alignment) * alignment : value;
}

/* ---- Group A.1: String/GLSL Parsing Helpers ---- */

const char *mglMemStr(const char *haystack, size_t haystack_len, const char *needle);
GLboolean mglRangeContainsToken(const char *begin, const char *end, const char *token);
GLboolean mglGLSLDeclaresRowMajorUBOMember(const char *glsl_src,
                                           const char *block_name,
                                           const char *member_name);
char *mglRecoverMemberNameFromGLSLComposite(const char *glsl_src,
                                            const char *composite_name,
                                            unsigned member_index,
                                            GLboolean require_block_brace);
char *mglRecoverUBOMemberNameFromGLSL(const char *glsl_src,
                                     const char *block_name,
                                     unsigned member_index);
char *mglRecoverStructMemberNameFromGLSL(const char *glsl_src,
                                         const char *struct_name,
                                         unsigned member_index);
char *mglGLSLTypeNameForMemberInComposite(const char *glsl_src,
                                          const char *composite_name,
                                          const char *member_name,
                                          GLboolean require_block_brace);
char *mglGLSLCompositeTypeNameForPath(const char *glsl_src,
                                      const char *block_name,
                                      const char *path);
char *mglDupRange(const char *begin, const char *end);
char *mglGLSLUBOInstanceName(const char *glsl_src, const char *block_name);
GLuint mglGLSLUBOArraySize(const char *glsl_src, const char *block_name);
char *mglBuildUBOMemberQueryName(const SpirvResource *ubo, const SpirvUBOMember *member);
char *mglGLSLAccessPathForUBOMember(const char *glsl_src,
                                    const char *block_name,
                                    const char *instance_name,
                                    const char *member_name);
GLboolean mglGLSLNameLooksLikeType(const char *name);
char *mglLeafNameFromPath(const char *path);
GLboolean mglGLSLContainsToken(const char *src, const char *token);

/* ---- Group A.2: SPIRV-Cross Type/Location/Size Query Helpers ---- */

GLuint mglGLTypeFromSPVCType(spvc_type type);
GLint mglGLArraySizeFromSPVCType(spvc_type type);
GLuint mglGLBoolTypeForVectorSize(unsigned vec_size);
GLuint mglGLTypeFromSPVCTypeAndGLSL(spvc_type type,
                                    const char *glsl_src,
                                    const char *block_name,
                                    const char *name);
char *mglJoinUBOMemberPath(const char *prefix, const char *member_name);
char *mglAppendArrayZeroSuffix(const char *name, unsigned num_dims);
GLuint mglMetalTypeAlignmentFromSPVC(spvc_compiler compiler, spvc_type type);
GLuint mglComputeMSLStructMemberOffset(spvc_compiler compiler,
                                        spvc_type struct_type,
                                        unsigned member_index);
GLuint mglComputeMSLStructSize(spvc_compiler compiler, spvc_type struct_type);
GLboolean mglSpvcStructMemberOffset(spvc_compiler compiler,
                                    spvc_type struct_type,
                                    spvc_type_id struct_type_id,
                                    unsigned member_index,
                                    GLuint *out);
GLint mglSpvcStructMemberMatrixStride(spvc_compiler compiler,
                                      spvc_type struct_type,
                                      spvc_type_id struct_type_id,
                                      unsigned member_index);
GLint mglSpvcStructMemberArrayStride(spvc_compiler compiler,
                                     spvc_type struct_type,
                                     spvc_type_id struct_type_id,
                                     unsigned member_index);
GLint mglGLTypeLocationCount(GLuint gl_type, GLint array_size);
GLint mglSPVCTypeLocationCount(spvc_compiler compiler, spvc_type type);
GLint mglSPVCTypeLocationStep(spvc_compiler compiler, spvc_type type);

/* ---- Group A.3: UBO Member Reflection ---- */

GLboolean mglAppendReflectedUBOMember(SpirvResource *ubo,
                                      GLuint *count,
                                      const char *name,
                                      GLuint gl_type,
                                      GLuint offset,
                                      GLint array_stride,
                                      GLint matrix_stride,
                                      GLboolean is_row_major,
                                      GLint size,
                                      GLint location_offset,
                                      GLint top_level_array_size,
                                      GLint top_level_array_stride);
GLboolean mglReflectUBOStructMember(Program *program,
                                    int stage,
                                    spvc_compiler compiler,
                                    SpirvResource *ubo,
                                    spvc_type struct_type,
                                    spvc_type_id struct_type_id,
                                    unsigned member_index,
                                    const char *prefix,
                                    GLuint base_offset,
                                    GLboolean inherited_row_major,
                                    GLuint *count,
                                    GLint location_offset,
                                    const spvc_buffer_range *active_ranges,
                                    size_t num_active_ranges,
                                    const MGLActivePathSet *active_paths,
                                    GLuint *current_path,
                                    GLuint current_path_len,
                                    GLint top_level_array_size,
                                    GLint top_level_array_stride);
GLboolean mglReflectUBOMemberLeaves(Program *program,
                                    int stage,
                                    spvc_compiler compiler,
                                    SpirvResource *ubo,
                                    spvc_type struct_type,
                                    spvc_type_id struct_type_id,
                                    const char *prefix,
                                    GLuint base_offset,
                                    GLboolean inherited_row_major,
                                    GLuint *count,
                                    GLint location_offset,
                                    const spvc_buffer_range *active_ranges,
                                    size_t num_active_ranges,
                                    const MGLActivePathSet *active_paths,
                                    GLuint *current_path,
                                    GLuint current_path_len,
                                    GLint top_level_array_size,
                                    GLint top_level_array_stride);

/* ---- Group A.4: Active Path / SPIR-V Binary Analysis ---- */

GLuint mglSpvResolveAccessChainRoot(GLuint result_id,
                                     GLuint bound,
                                     const GLuint *const_values,
                                     const GLboolean *is_const,
                                     const GLuint *chain_base,
                                     const GLuint *chain_num_idx,
                                     const GLuint *chain_idx_ids,
                                     const GLboolean *chain_valid,
                                     GLuint idx_stride,
                                     GLuint path_out[MGL_ACTIVE_MAX_DEPTH],
                                     GLuint *path_len_out);
void mglCollectActivePaths(const unsigned int *spirv, size_t word_count,
                           GLuint var_id,
                           MGLActivePathSet *out);
GLboolean mglActivePathHasPrefix(const MGLActivePathSet *set,
                                 const GLuint *prefix, GLuint prefix_len);
GLboolean mglActivePathExactMatch(const MGLActivePathSet *set,
                                  const GLuint *path, GLuint path_len);
GLboolean mglByteOffsetIsActive(GLuint offset,
                                GLint array_stride,
                                GLint array_size,
                                const spvc_buffer_range *active_ranges,
                                size_t num_active_ranges);
unsigned mglSPIRVFindAccessChainConstantIndices(const unsigned int *ir,
                                                 size_t ir_size_bytes,
                                                 unsigned var_id,
                                                 unsigned *out_indices,
                                                 unsigned max_indices);

/* ---- Group A.5: Active Uniform/Block/Attrib Query Functions ---- */

GLboolean mglUniformBlockNameSeen(Program *program, int max_stage, GLuint max_index,
                                  const char *name, GLuint gl_binding);
GLuint mglProgramUniformBlockArraySize(const SpirvResource *block);
GLint mglActiveUniformBlockCount(Program *program);
GLint mglActiveAtomicCounterBufferCount(Program *program);
GLint mglActiveUniformBlockMaxNameLength(Program *program);
SpirvResourceList *mglProgramActiveAttribList(Program *program);
GLboolean mglProgramActiveAttribHasName(const SpirvResource *res);
GLint mglProgramActiveAttribCount(Program *program);
SpirvResource *mglProgramActiveAttribAt(Program *program, GLuint index);
GLint mglProgramActiveAttribMaxNameLength(Program *program);
GLenum mglProgramActiveAttribType(const SpirvResource *res);

/* ---- Group A.6: Sampler Uniform Location Unification ---- */

GLint mglSyntheticSamplerUniformLocation(int stage, int res_type, GLuint index);
GLint mglFindExplicitUniformLocation(const char *glsl_src, const char *resource_name);
GLint mglSamplerUniformLocationFromReflection(GLuint reflected_location,
                                              int stage,
                                              int res_type,
                                              GLuint index,
                                              const char *glsl_src,
                                              const char *resource_name);
bool mglIsSamplerResourceType(int res_type);
bool mglUniformNameLooksSamplerLike(const char *name);
bool mglUniformConstantBaseTypeIsSamplerLike(spvc_basetype basetype);
bool mglProgramResourceLooksSamplerLike(const SpirvResource *res, int res_type);
bool mglSamplerResourceNamesMatch(const char *a, const char *b);
void mglUnifySamplerUniformLocations(Program *program);

/* ---- Group A.7: Plain Uniform Location Assignment ---- */

SpirvResource *mglFindAssignedPlainUniformResource(Program *program, const char *name);
GLint mglFirstFreePlainUniformLocation(const bool used[MAX_BINDABLE_BUFFERS]);
void mglAssignPlainUniformLocations(Program *program);
GLint mglDefaultSamplerUnitForProgramResource(Program *program, const SpirvResource *res);
void mglApplyDefaultSamplerUnit(Program *program, int stage, int res_type, SpirvResource *res);

/* ---- Group A.8: Misc Reflection Utilities ---- */

void mglFreeSpirvResourceOwnedFields(SpirvResource *res);
GLint mglPlainUniformResourceLocationForProgram(const SpirvResource *res);
GLboolean mglParseScalarUniformInitializer(const char *src,
                                           const char *name,
                                           spvc_basetype basetype,
                                           uint8_t *value,
                                           GLsizeiptr *size_out);

#ifdef __cplusplus
}
#endif

#endif /* mgl_uniform_reflection_h */
