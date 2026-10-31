/* GLSL 4.0 subroutine tables shared by the AIR stage reflection and the
 * program object.  Deliberately dependency-free: both mgl_shader_abi.h and
 * mgl_types_program.h include it, and neither can include the other. */
#ifndef MGL_SUBROUTINE_TABLE_H
#define MGL_SUBROUTINE_TABLE_H

#include <stdint.h>

#define MGL_AIR_MAX_SUBROUTINE_TYPES    8
/* Per-type implementation capacity.  The GL limit (GL_MAX_SUBROUTINES) is 256
 * program-wide; the reflection table is a fixed-size struct in the ABI, so it
 * covers 64 per type, which leaves room for the sparse layout(index = N)
 * reservations CTS exercises without turning MGLAIRStageInfo into a
 * quarter-megabyte.  Exceeding it is a hard link/codegen error, never a
 * silent truncation. */
#define MGL_AIR_MAX_SUBROUTINE_IMPLS    64
/* Largest array of subroutine uniforms one glUniformSubroutinesuiv call may
 * apply at once; bounds the local value buffer. */
#define MGL_AIR_MAX_SUBROUTINE_UNIFORM_ARRAY 256
#define MGL_AIR_MAX_SUBROUTINE_UNIFORMS 32
#define MGL_AIR_SUBROUTINE_NAME_MAX     64

/* An explicit layout(location = N) is reported as N; this means "unset". */
#define MGL_SUBROUTINE_LOCATION_NONE (-1)

typedef struct MGLAIRSubroutineTypeInfo {
    char name[MGL_AIR_SUBROUTINE_NAME_MAX];
    uint32_t impl_count;
    char impls[MGL_AIR_MAX_SUBROUTINE_IMPLS][MGL_AIR_SUBROUTINE_NAME_MAX];
    /* Index each implementation occupies in the program-wide index space.
     * An implementation declared with layout(index = N) keeps N; the others
     * fill the gaps in declaration order (GLSL 4.60 §6.1.2).  `impls` is
     * indexed by that absolute index, so a sparse layout leaves empty slots;
     * `impl_names` lists the named ones in ascending index order for the
     * enumerating queries (glGetActiveSubroutineName takes an ordinal, not an
     * index -- GL 4.6 §7.10). */
    uint32_t impl_indices[MGL_AIR_MAX_SUBROUTINE_IMPLS];
    uint32_t impl_name_count;
    char impl_names[MGL_AIR_MAX_SUBROUTINE_IMPLS][MGL_AIR_SUBROUTINE_NAME_MAX];
    uint32_t impl_name_index[MGL_AIR_MAX_SUBROUTINE_IMPLS];
} MGLAIRSubroutineTypeInfo;

typedef struct MGLAIRSubroutineTable {
    uint32_t type_count;
    MGLAIRSubroutineTypeInfo types[MGL_AIR_MAX_SUBROUTINE_TYPES];
    /* Flattened subroutine uniforms in declaration order: an index here is the
     * location glGetSubroutineUniformLocation reports.  An arrayed uniform
     * (`subroutine uniform T g[2];`) contributes one entry per element, in
     * element order, matching GL 4.6 §7.10 (one location per element). */
    uint32_t uniform_count;
    char uniform_names[MGL_AIR_MAX_SUBROUTINE_UNIFORMS][MGL_AIR_SUBROUTINE_NAME_MAX];
    /* Which type each uniform selects from (index into `types`). */
    uint32_t uniform_type[MGL_AIR_MAX_SUBROUTINE_UNIFORMS];
    /* Element index within the declared array (0 for a non-array uniform),
     * the declared array size (1 for a non-array uniform), and the explicit
     * location or MGL_SUBROUTINE_LOCATION_NONE. */
    uint32_t uniform_element[MGL_AIR_MAX_SUBROUTINE_UNIFORMS];
    uint32_t uniform_array_size[MGL_AIR_MAX_SUBROUTINE_UNIFORMS];      /* span of this row: 1 */
    uint32_t uniform_declared_elements[MGL_AIR_MAX_SUBROUTINE_UNIFORMS]; /* all rows of a uniform */
    int32_t uniform_location[MGL_AIR_MAX_SUBROUTINE_UNIFORMS];
    /* Total implementations across all types: the size of the program-wide
     * subroutine index space (GL 4.6 §7.10 indexes are unique per stage).
     * An explicit layout(index = N) reserves N, so this is
     * max(count, highest explicit index + 1) for each type. */
    uint32_t total_impl_count;
    /* Where each type's implementations start in that index space. */
    uint32_t type_impl_base[MGL_AIR_MAX_SUBROUTINE_TYPES];
} MGLAIRSubroutineTable;

/* Location bookkeeping shared by the GL query family and
 * glUniformSubroutinesuiv.  Both must agree on where a uniform lives: an
 * explicit layout(location = N) wins, otherwise the linker assigns the lowest
 * free slot in declaration order (GL 4.6 §7.10). */
int mglSubroutineUniformImplicitLocation(const MGLAIRSubroutineTable *table,
                                         uint32_t uniform);

/* Location of one subroutine uniform (an explicit layout(location = N) else the
 * linker-assigned slot). */
int mglSubroutineUniformLocationAt(const MGLAIRSubroutineTable *table,
                                   uint32_t uniform);

/* Link-time validation (GL 4.6 §7.10): subroutine-uniform locations must be
 * unique within the stage and below `maxLocations`.  Returns 1 when the table
 * is linkable, 0 when it is not. */
/* Returns 1 when the table is linkable, 0 otherwise.  On failure `err` (when
 * non-NULL) receives a one-line reason: the caller used to print an empty
 * buffer borrowed from the reflection step, which said nothing. */
int mglSubroutineValidateTable(const MGLAIRSubroutineTable *table,
                               uint32_t maxLocations,
                               char *err, size_t errcap);

/* ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS: one past the highest assigned location.
 * This is also the exact `count` glUniformSubroutinesuiv requires. */
int mglSubroutineUniformLocationSpan(const MGLAIRSubroutineTable *table);

/* GL 4.6 §7.10 counts an arrayed subroutine uniform (`subroutine uniform
 * T g[2];`) as ONE subroutine uniform for the enumerating queries
 * (glGetActiveSubroutineUniform*, the program resource API), while the
 * location space stays one location per element.  This maps a
 * distinct-uniform ordinal to its element-0 row.  Returns 0 and sets
 * *row_out on success, -1 when the ordinal is out of range. */
int mglSubroutineDistinctUniformRow(const MGLAIRSubroutineTable *table,
                                    GLuint ordinal, uint32_t *row_out);

/* Base name of a uniform row ("g[0][1]" -> "g"); a scalar row copies
 * through.  The enumerating queries report this spelling for arrays. */
void mglSubroutineUniformBaseName(const char *row_name, char *out, size_t cap);

#endif
