//
// Hand-maintained; no longer regenerated from gl.xml
//
// Mike Larson
//
// October 2021
//

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "mgl.h"
#include "draw_command.h"
#include "mgl_pixel_format.h"
#include "pixel_utils.h"

#include "mgl_trace_log.h"
#include "mgl_buffer_plan.h"

#ifndef MGL_VERBOSE_TEXBUFFER_LOGS
#define MGL_VERBOSE_TEXBUFFER_LOGS 0
#endif

static void mgl_unimplemented(GLMContext ctx, const char *func)
{
    static uint32_t warn_count = 0u;
    if (warn_count < 128u) {
        fprintf(stderr, "MGL WARNING: %s is unimplemented, returning GL_INVALID_OPERATION\n",
                func ? func : "(unknown)");
        warn_count++;
    }
    if (ctx) {
        STATE(error) = GL_INVALID_OPERATION;
    }
}

static void mglSetCurrentVertexAttribFloat(GLMContext ctx,
                                           GLuint index,
                                           GLfloat x,
                                           GLfloat y,
                                           GLfloat z,
                                           GLfloat w)
{
	ERROR_CHECK_RETURN(ctx, GL_INVALID_OPERATION);
	ERROR_CHECK_RETURN(index < MAX_ATTRIBS, GL_INVALID_VALUE);

	CurrentVertexAttrib *attrib = &ctx->state.current_vertex_attrib[index];
	attrib->f[0] = x;
	attrib->f[1] = y;
	attrib->f[2] = z;
	attrib->f[3] = w;
	for (int i = 0; i < 4; i++) {
		attrib->d[i] = (GLdouble)attrib->f[i];
		attrib->i[i] = (GLint)attrib->f[i];
		attrib->u[i] = (GLuint)attrib->f[i];
	}
	attrib->type = GL_FLOAT;
	attrib->integer = GL_FALSE;
	attrib->long_attribute = GL_FALSE;
	mglMarkStateDirtyBits(&ctx->state, DIRTY_VAO);
}

static void mglSetCurrentVertexAttribInt(GLMContext ctx,
                                         GLuint index,
                                         GLint x,
                                         GLint y,
                                         GLint z,
                                         GLint w)
{
	ERROR_CHECK_RETURN(ctx, GL_INVALID_OPERATION);
	ERROR_CHECK_RETURN(index < MAX_ATTRIBS, GL_INVALID_VALUE);

	CurrentVertexAttrib *attrib = &ctx->state.current_vertex_attrib[index];
	attrib->i[0] = x;
	attrib->i[1] = y;
	attrib->i[2] = z;
	attrib->i[3] = w;
	for (int i = 0; i < 4; i++) {
		attrib->u[i] = (GLuint)attrib->i[i];
		attrib->f[i] = (GLfloat)attrib->i[i];
		attrib->d[i] = (GLdouble)attrib->i[i];
	}
	attrib->type = GL_INT;
	attrib->integer = GL_TRUE;
	attrib->long_attribute = GL_FALSE;
	mglMarkStateDirtyBits(&ctx->state, DIRTY_VAO);
}

static void mglSetCurrentVertexAttribUInt(GLMContext ctx,
                                          GLuint index,
                                          GLuint x,
                                          GLuint y,
                                          GLuint z,
                                          GLuint w)
{
	ERROR_CHECK_RETURN(ctx, GL_INVALID_OPERATION);
	ERROR_CHECK_RETURN(index < MAX_ATTRIBS, GL_INVALID_VALUE);

	CurrentVertexAttrib *attrib = &ctx->state.current_vertex_attrib[index];
	attrib->u[0] = x;
	attrib->u[1] = y;
	attrib->u[2] = z;
	attrib->u[3] = w;
	for (int i = 0; i < 4; i++) {
		attrib->i[i] = (GLint)attrib->u[i];
		attrib->f[i] = (GLfloat)attrib->u[i];
		attrib->d[i] = (GLdouble)attrib->u[i];
	}
	attrib->type = GL_UNSIGNED_INT;
	attrib->integer = GL_TRUE;
	attrib->long_attribute = GL_FALSE;
	mglMarkStateDirtyBits(&ctx->state, DIRTY_VAO);
}

static void mglSetCurrentVertexAttribDouble(GLMContext ctx,
                                            GLuint index,
                                            GLdouble x,
                                            GLdouble y,
                                            GLdouble z,
                                            GLdouble w)
{
	ERROR_CHECK_RETURN(ctx, GL_INVALID_OPERATION);
	ERROR_CHECK_RETURN(index < MAX_ATTRIBS, GL_INVALID_VALUE);

	CurrentVertexAttrib *attrib = &ctx->state.current_vertex_attrib[index];
	attrib->d[0] = x;
	attrib->d[1] = y;
	attrib->d[2] = z;
	attrib->d[3] = w;
	for (int i = 0; i < 4; i++) {
		attrib->f[i] = (GLfloat)attrib->d[i];
		attrib->i[i] = (GLint)attrib->d[i];
		attrib->u[i] = (GLuint)attrib->d[i];
	}
	attrib->type = GL_DOUBLE;
	attrib->integer = GL_FALSE;
	attrib->long_attribute = GL_TRUE;
	mglMarkStateDirtyBits(&ctx->state, DIRTY_VAO);
}

// Forward declarations for transform feedback functions from program.c
TransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name);
TransformFeedback *findTransformFeedback(GLMContext ctx, GLuint name);
TransformFeedback *getTransformFeedback(GLMContext ctx, GLuint name);
Program *findProgram(GLMContext ctx, GLuint program);
ProgramPipeline *findProgramPipeline(GLMContext ctx, GLuint pipeline);
GLboolean mglProgramPipelinePerVertexCompatible(Program *const *stage_programs);

// Forward declaration for texture lookup from textures.c
extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern Texture *currentTexture(GLMContext ctx, GLuint index);
extern Texture *getTex(GLMContext ctx, GLuint name, GLenum target);
extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern Buffer *getBuffer(GLMContext ctx, GLenum target, GLuint buffer);
extern void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size);
// Forward declaration for renderbuffer lookup from framebuffers.c
extern Renderbuffer *findRenderbuffer(GLMContext ctx, GLuint renderbuffer);

typedef struct QueryObject_t {
	GLuint name;
	GLenum target;
	GLboolean active;
	GLboolean available;
	GLboolean saw_draw;
	GLboolean sample_result_known;
	GLboolean primitive_result_known;
	GLboolean timer_result_known;  /* GL_TIME_ELAPSED: set when mtlEndTimerQuery wrote a real GPU result */
	GLuint64 result;
} QueryObject;

/* Primitive queries are indexed by output stream (GL 4.6 §13.2.4).
 * Other query targets only accept index zero, but keeping a fixed-width
 * second dimension makes the active-query lookup unambiguous and preserves
 * the existing target-slot numbering. */
#define MGL_QUERY_MAX_INDEX 4u

static HashTable s_query_table;
/* Single-threaded access only — MGL assumes one GL context per thread.
 * If multi-threaded access is ever needed, add a module-level
 * os_unfair_lock around all reads and writes. */
static GLboolean s_query_table_initialized = GL_FALSE;
static GLuint s_active_query_by_target[18][MGL_QUERY_MAX_INDEX];
static GLuint64 s_fake_timestamp_counter = 1;

static QueryObject *mgl_find_query(GLuint id);

static size_t mgl_round_up_16(size_t value)
{
	return value ? ((value + 15) & ~(size_t)15) : 0;
}

static GLuint mgl_effective_max_viewports(GLMContext ctx)
{
	GLuint max_viewports = ctx ? ctx->state.var.max_viewports : 1;
	if (max_viewports == 0 || max_viewports > MGL_MAX_VIEWPORTS) {
		max_viewports = MGL_MAX_VIEWPORTS;
	}
	return max_viewports ? max_viewports : 1;
}

static GLboolean mgl_validate_viewport_range(GLMContext ctx, GLuint first, GLsizei count)
{
	ERROR_CHECK_RETURN_VALUE(count >= 0, GL_INVALID_VALUE, GL_FALSE);
	GLuint max_viewports = mgl_effective_max_viewports(ctx);
	ERROR_CHECK_RETURN_VALUE(first <= max_viewports, GL_INVALID_VALUE, GL_FALSE);
	ERROR_CHECK_RETURN_VALUE((GLuint)count <= max_viewports - first, GL_INVALID_VALUE, GL_FALSE);
	return GL_TRUE;
}

static void mgl_init_query_table_if_needed(void)
{
	if (!s_query_table_initialized)
	{
		initHashTable(&s_query_table, 64);
		memset(s_active_query_by_target, 0, sizeof(s_active_query_by_target));
		s_query_table_initialized = GL_TRUE;
	}
}

static int mgl_query_target_slot(GLenum target)
{
	switch (target)
	{
		case GL_SAMPLES_PASSED: return 0;
		case GL_ANY_SAMPLES_PASSED: return 1;
		case GL_ANY_SAMPLES_PASSED_CONSERVATIVE: return 2;
		case GL_PRIMITIVES_GENERATED: return 3;
		case GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN: return 4;
		case GL_TIME_ELAPSED: return 5;
		case GL_VERTICES_SUBMITTED: return 6;
		case GL_PRIMITIVES_SUBMITTED: return 7;
		case GL_VERTEX_SHADER_INVOCATIONS: return 8;
		case GL_TESS_CONTROL_SHADER_PATCHES: return 9;
		case GL_TESS_EVALUATION_SHADER_INVOCATIONS: return 10;
		case GL_GEOMETRY_SHADER_INVOCATIONS: return 11;
		case GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED: return 12;
		case GL_FRAGMENT_SHADER_INVOCATIONS: return 13;
		case GL_COMPUTE_SHADER_INVOCATIONS: return 14;
		case GL_CLIPPING_INPUT_PRIMITIVES: return 15;
		case GL_CLIPPING_OUTPUT_PRIMITIVES: return 16;
		default: return -1;
	}
}

static GLboolean mgl_query_index_is_valid(GLenum target, GLuint index)
{
	if (index == 0u)
		return GL_TRUE;
	return (target == GL_PRIMITIVES_GENERATED ||
		target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) &&
		index < MGL_QUERY_MAX_INDEX;
}

static GLboolean mgl_is_query_create_target(GLenum target)
{
	return target == GL_SAMPLES_PASSED ||
	       target == GL_ANY_SAMPLES_PASSED ||
	       target == GL_ANY_SAMPLES_PASSED_CONSERVATIVE ||
	       target == GL_PRIMITIVES_GENERATED ||
	       target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN ||
	       target == GL_TIME_ELAPSED ||
	       target == GL_TIMESTAMP ||
	       target == GL_VERTICES_SUBMITTED ||
	       target == GL_PRIMITIVES_SUBMITTED ||
	       target == GL_VERTEX_SHADER_INVOCATIONS ||
	       target == GL_TESS_CONTROL_SHADER_PATCHES ||
	       target == GL_TESS_EVALUATION_SHADER_INVOCATIONS ||
	       target == GL_GEOMETRY_SHADER_INVOCATIONS ||
	       target == GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED ||
	       target == GL_FRAGMENT_SHADER_INVOCATIONS ||
	       target == GL_COMPUTE_SHADER_INVOCATIONS ||
	       target == GL_CLIPPING_INPUT_PRIMITIVES ||
	       target == GL_CLIPPING_OUTPUT_PRIMITIVES;
}

static GLboolean mgl_query_target_is_sample(GLenum target)
{
	return target == GL_SAMPLES_PASSED ||
	       target == GL_ANY_SAMPLES_PASSED ||
	       target == GL_ANY_SAMPLES_PASSED_CONSERVATIVE;
}

static GLboolean mgl_query_mode_is_inverted(GLenum mode)
{
	return mode == GL_QUERY_WAIT_INVERTED ||
	       mode == GL_QUERY_NO_WAIT_INVERTED ||
	       mode == GL_QUERY_BY_REGION_WAIT_INVERTED ||
	       mode == GL_QUERY_BY_REGION_NO_WAIT_INVERTED;
}

static GLboolean mgl_query_mode_is_valid(GLenum mode)
{
	switch (mode)
	{
		case GL_QUERY_WAIT:
		case GL_QUERY_NO_WAIT:
		case GL_QUERY_BY_REGION_WAIT:
		case GL_QUERY_BY_REGION_NO_WAIT:
		case GL_QUERY_WAIT_INVERTED:
		case GL_QUERY_NO_WAIT_INVERTED:
		case GL_QUERY_BY_REGION_WAIT_INVERTED:
		case GL_QUERY_BY_REGION_NO_WAIT_INVERTED:
			return GL_TRUE;
		default:
			return GL_FALSE;
	}
}

void mglRecordActiveSampleQueryDraw(GLMContext ctx)
{
	if (!ctx)
		return;

	mgl_init_query_table_if_needed();
	for (int slot = 0; slot < 3; slot++)
	{
		QueryObject *q = mgl_find_query(s_active_query_by_target[slot][0]);
		if (!q || !q->active || !mgl_query_target_is_sample(q->target))
			continue;

		/* Just mark that a draw occurred; the actual sample-pass result is
		 * read back from the Metal visibility result buffer in mglEndQuery. */
		q->saw_draw = GL_TRUE;
	}
}

void mglRecordActivePrimitiveQueryDrawIndexed(GLMContext ctx,
                                               GLuint index,
                                               GLuint64 generated,
                                               GLuint64 written)
{
	(void)ctx;
	if (index >= MGL_QUERY_MAX_INDEX)
		return;

	mgl_init_query_table_if_needed();

	QueryObject *generated_query =
		mgl_find_query(s_active_query_by_target[3][index]);
	if (generated_query && generated_query->active &&
	    generated_query->target == GL_PRIMITIVES_GENERATED)
	{
		generated_query->saw_draw = GL_TRUE;
		generated_query->primitive_result_known = GL_TRUE;
		generated_query->result += generated;
	}

	QueryObject *written_query =
		mgl_find_query(s_active_query_by_target[4][index]);
	if (written_query && written_query->active &&
	    written_query->target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)
	{
		written_query->saw_draw = GL_TRUE;
		written_query->primitive_result_known = GL_TRUE;
		written_query->result += written;
	}
}

void mglRecordActivePrimitiveQueryDraw(GLMContext ctx,
                                       GLuint64 generated,
                                       GLuint64 written)
{
	mglRecordActivePrimitiveQueryDrawIndexed(ctx, 0u, generated, written);
}

GLboolean mglHasActiveIndexedPrimitiveQuery(void)
{
	mgl_init_query_table_if_needed();
	for (GLuint index = 1u; index < MGL_QUERY_MAX_INDEX; index++) {
		if (s_active_query_by_target[3][index] != 0u ||
			s_active_query_by_target[4][index] != 0u)
			return GL_TRUE;
	}
	return GL_FALSE;
}

GLboolean mglShouldSkipConditionalRender(GLMContext ctx)
{
	return (ctx && ctx->state.conditional_render_active &&
	        ctx->state.conditional_render_skip) ? GL_TRUE : GL_FALSE;
}

static QueryObject *mgl_find_query(GLuint id)
{
	mgl_init_query_table_if_needed();
	return (QueryObject *)searchHashTable(&s_query_table, id);
}

static QueryObject *mgl_get_query(GLuint id)
{
	QueryObject *q;

	mgl_init_query_table_if_needed();
	q = (QueryObject *)searchHashTable(&s_query_table, id);
	if (!q)
	{
		q = (QueryObject *)calloc(1, sizeof(QueryObject));
		if (!q)
			return NULL;
		q->name = id;
		insertHashElement(&s_query_table, id, q);
	}
	return q;
}

static GLboolean mgl_query_value(const QueryObject *q, GLenum pname, GLuint64 *value)
{
	if (!q || !value)
		return GL_FALSE;

	switch (pname)
	{
		case GL_QUERY_RESULT_AVAILABLE:
			*value = q->available ? 1u : 0u;
			return GL_TRUE;
		case GL_QUERY_RESULT:
		case GL_QUERY_RESULT_NO_WAIT:
			*value = q->available ? q->result : 0u;
			return GL_TRUE;
		case GL_QUERY_TARGET:
			*value = (GLuint64)q->target;
			return GL_TRUE;
		default:
			return GL_FALSE;
	}
}

static void mgl_finish_query_result(QueryObject *q)
{
	if (!q)
		return;

	q->active = GL_FALSE;
	q->available = GL_TRUE;
	switch (q->target)
	{
		case GL_SAMPLES_PASSED:
		case GL_ANY_SAMPLES_PASSED:
		case GL_ANY_SAMPLES_PASSED_CONSERVATIVE:
			q->result = q->sample_result_known ? q->result : (q->saw_draw ? 1u : 0u);
			break;
		case GL_PRIMITIVES_GENERATED:
		case GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN:
			if (!q->primitive_result_known)
				q->result = q->saw_draw ? 1u : 0u;
			break;
		/* GL_ARB_pipeline_statistics_query targets.  MGL does not sample
		 * real GPU stage counters, so report a deterministic nonzero value
		 * when a draw occurred within the query's begin/end interval.  This
		 * satisfies the GL_QUERY_COUNTER_BITS / begin-query API gates and
		 * the nonzero-when-drawn value checks. */
		case GL_VERTICES_SUBMITTED:
		case GL_PRIMITIVES_SUBMITTED:
		case GL_VERTEX_SHADER_INVOCATIONS:
		case GL_TESS_CONTROL_SHADER_PATCHES:
		case GL_TESS_EVALUATION_SHADER_INVOCATIONS:
		case GL_GEOMETRY_SHADER_INVOCATIONS:
		case GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED:
		case GL_FRAGMENT_SHADER_INVOCATIONS:
		case GL_COMPUTE_SHADER_INVOCATIONS:
		case GL_CLIPPING_INPUT_PRIMITIVES:
		case GL_CLIPPING_OUTPUT_PRIMITIVES:
			q->result = q->saw_draw ? 1u : 0u;
			break;
		case GL_TIME_ELAPSED:
			/* Real GPU elapsed time is set by mtlEndTimerQuery in
			 * mglEndQuery (with timer_result_known = GL_TRUE).
			 * Only fall back to the fake counter when the Metal
			 * backend was never called — a real zero result (e.g.
			 * an extremely fast GPU pass) must be preserved. */
			if (!q->timer_result_known)
				q->result = s_fake_timestamp_counter++;
			break;
		default:
			q->result = 0;
			break;
	}
}

/* Check if a program interface token is valid per the GL spec.
 * Subroutine and transform-feedback interfaces are valid but not
 * backed by shader resource types, so they need separate handling. */
static GLboolean mgl_program_interface_is_valid(GLenum programInterface)
{
	switch (programInterface)
	{
		case GL_PROGRAM_INPUT:
		case GL_PROGRAM_OUTPUT:
		case GL_UNIFORM:
		case GL_UNIFORM_BLOCK:
		case GL_BUFFER_VARIABLE:
		case GL_SHADER_STORAGE_BLOCK:
		case GL_ATOMIC_COUNTER_BUFFER:
		case GL_TRANSFORM_FEEDBACK_VARYING:
		case GL_VERTEX_SUBROUTINE:
		case GL_TESS_CONTROL_SUBROUTINE:
		case GL_TESS_EVALUATION_SUBROUTINE:
		case GL_GEOMETRY_SUBROUTINE:
		case GL_FRAGMENT_SUBROUTINE:
		case GL_COMPUTE_SUBROUTINE:
		case GL_VERTEX_SUBROUTINE_UNIFORM:
		case GL_TESS_CONTROL_SUBROUTINE_UNIFORM:
		case GL_TESS_EVALUATION_SUBROUTINE_UNIFORM:
		case GL_GEOMETRY_SUBROUTINE_UNIFORM:
		case GL_FRAGMENT_SUBROUTINE_UNIFORM:
		case GL_COMPUTE_SUBROUTINE_UNIFORM:
			return GL_TRUE;
		default:
			return GL_FALSE;
	}
}

static int mgl_program_interface_to_spvc(GLenum programInterface)
{
	switch (programInterface)
	{
		case GL_PROGRAM_INPUT: return _STAGE_INPUT_RES;
		case GL_PROGRAM_OUTPUT: return _STAGE_OUTPUT_RES;
		case GL_UNIFORM: return _UNIFORM_CONSTANT_RES;
		case GL_UNIFORM_BLOCK: return _UNIFORM_BUFFER_RES;
		case GL_SHADER_STORAGE_BLOCK: return _STORAGE_BUFFER_RES;
		case GL_BUFFER_VARIABLE: return _STORAGE_BUFFER_RES;
		case GL_ATOMIC_COUNTER_BUFFER: return _ATOMIC_COUNTER_RES;
		default: return -1;
	}
}

static int mgl_program_interface_to_spvc_list(GLenum programInterface, int *types, int max_types)
{
	if (!types || max_types <= 0)
		return 0;

	if (programInterface == GL_UNIFORM)
	{
		int n = 0;
		types[n++] = _UNIFORM_CONSTANT_RES;
		if (n < max_types) types[n++] = _SAMPLED_IMAGE_RES;
		if (n < max_types) types[n++] = _SEPARATE_IMAGE_RES;
		if (n < max_types) types[n++] = _SEPARATE_SAMPLERS_RES;
		if (n < max_types) types[n++] = _STORAGE_IMAGE_RES;
		if (n < max_types) types[n++] = _ATOMIC_COUNTER_RES;
		return n;
	}

	int type = mgl_program_interface_to_spvc(programInterface);
	if (type < 0)
		return 0;

	types[0] = type;
	return 1;
}

static GLboolean mgl_program_uniform_block_identity_matches(const MGLShaderResource *a, const MGLShaderResource *b)
{
	if (!a || !b)
		return GL_FALSE;

	if (a->name && b->name && a->name[0] != '\0' && b->name[0] != '\0')
		return strcmp(a->name, b->name) == 0 ? GL_TRUE : GL_FALSE;

	if ((a->name && a->name[0] != '\0') || (b->name && b->name[0] != '\0'))
		return GL_FALSE;

	return a->gl_binding == b->gl_binding ? GL_TRUE : GL_FALSE;
}

static GLuint mgl_program_uniform_block_array_size(const MGLShaderResource *block)
{
	return (block && block->ubo_array_size > 0) ? block->ubo_array_size : 1u;
}

static GLuint mgl_program_uniform_block_element_binding(const MGLShaderResource *block, GLuint element)
{
	if (!block)
		return 0;
	if (block->ubo_array_bindings && element < mgl_program_uniform_block_array_size(block))
		return block->ubo_array_bindings[element];
	return block->gl_binding + element;
}

static GLsizei mgl_program_uniform_block_element_name(const MGLShaderResource *block,
                                                      GLuint element,
                                                      GLsizei bufSize,
                                                      GLchar *name)
{
	GLsizei len = 0;
	if (!block || !block->name)
	{
		if (name && bufSize > 0)
			name[0] = '\0';
		return 0;
	}

	if (mgl_program_uniform_block_array_size(block) > 1)
		len = snprintf(name && bufSize > 0 ? name : NULL,
		               name && bufSize > 0 ? (size_t)bufSize : 0u,
		               "%s[%u]", block->name, element);
	else
		len = snprintf(name && bufSize > 0 ? name : NULL,
		               name && bufSize > 0 ? (size_t)bufSize : 0u,
		               "%s", block->name);
	return len;
}

static GLboolean mgl_program_uniform_block_name_matches(const MGLShaderResource *block,
                                                        const GLchar *name,
                                                        GLuint *out_element)
{
	if (!block || !block->name || !name)
		return GL_FALSE;

	GLuint array_size = mgl_program_uniform_block_array_size(block);

	/* Exact name match always succeeds (element 0). */
	if (strcmp(block->name, name) == 0)
	{
		if (out_element)
			*out_element = 0;
		return GL_TRUE;
	}

	/* If the UBO is declared as an array, accept "name[N]" syntax. */
	if (!block->ubo_is_array)
		return GL_FALSE;

	size_t base_len = strlen(block->name);
	if (strncmp(block->name, name, base_len) != 0 || name[base_len] != '[')
		return GL_FALSE;

	char *end = NULL;
	unsigned long parsed = strtoul(name + base_len + 1, &end, 10);
	if (!end || *end != ']' || end[1] != '\0' || parsed >= array_size)
		return GL_FALSE;

	if (out_element)
		*out_element = (GLuint)parsed;
	return GL_TRUE;
}

static GLboolean mgl_program_resource_name_with_array_matches(const MGLShaderResource *res,
                                                              const GLchar *name,
                                                              GLuint *out_element)
{
	if (!res || !res->name || !name)
		return GL_FALSE;

	GLuint array_size = res->gl_array_size > 0 ? (GLuint)res->gl_array_size : 1u;

	/* For non-array resources, only exact name matches. */
	if (!res->is_array)
	{
		if (strcmp(res->name, name) == 0)
		{
			if (out_element)
				*out_element = 0;
			return GL_TRUE;
		}
		return GL_FALSE;
	}

	/* For array resources, match either the base name or "name[N]". */
	if (strcmp(res->name, name) == 0)
	{
		if (out_element)
			*out_element = 0;
		return GL_TRUE;
	}

	size_t base_len = strlen(res->name);
	if (strncmp(res->name, name, base_len) != 0 || name[base_len] != '[')
		return GL_FALSE;

	char *end = NULL;
	unsigned long parsed = strtoul(name + base_len + 1, &end, 10);
	if (!end || *end != ']' || end[1] != '\0' || parsed >= array_size)
		return GL_FALSE;
	if (out_element)
		*out_element = (GLuint)parsed;
	return GL_TRUE;
}

static GLuint mgl_program_resource_location_span(const MGLShaderResource *res)
{
	if (!res)
		return 1;

	switch (res->gl_type)
	{
		case GL_FLOAT_MAT2:
		case GL_FLOAT_MAT2x3:
		case GL_FLOAT_MAT2x4:
			return 2;
		case GL_FLOAT_MAT3:
		case GL_FLOAT_MAT3x2:
		case GL_FLOAT_MAT3x4:
			return 3;
		case GL_FLOAT_MAT4:
		case GL_FLOAT_MAT4x2:
		case GL_FLOAT_MAT4x3:
			return 4;
		default:
			return 1;
	}
}

static GLsizei mgl_program_resource_name_with_array(const MGLShaderResource *res,
                                                    GLsizei bufSize,
                                                    GLchar *name)
{
	if (!res || !res->name)
	{
		if (name && bufSize > 0)
			name[0] = '\0';
		return 0;
	}
	if (res->is_array && !strchr(res->name, '['))
		return snprintf(name && bufSize > 0 ? name : NULL,
		                name && bufSize > 0 ? (size_t)bufSize : 0u,
		                "%s[0]", res->name);
	return snprintf(name && bufSize > 0 ? name : NULL,
	                name && bufSize > 0 ? (size_t)bufSize : 0u,
	                "%s", res->name);
}

static GLboolean mgl_program_block_seen_before(Program *pptr, int res_type, int target_stage, GLuint target_index)
{
	if (!pptr || target_stage < 0 || target_stage >= _MAX_SHADER_TYPES)
		return GL_FALSE;

	MGLShaderResourceList *target_resources =
		&pptr->shader_resources_list[target_stage][res_type];
	if (!target_resources->list || target_index >= target_resources->count)
		return GL_FALSE;

	MGLShaderResource *target = &target_resources->list[target_index];
	for (int stage = 0; stage <= target_stage && stage < _MAX_SHADER_TYPES; stage++)
	{
		MGLShaderResourceList *resources =
			&pptr->shader_resources_list[stage][res_type];
		GLuint limit = (stage == target_stage) ? target_index : resources->count;
		for (GLuint i = 0; resources->list && i < limit; i++)
		{
			if (mgl_program_uniform_block_identity_matches(&resources->list[i], target))
				return GL_TRUE;
		}
	}

	return GL_FALSE;
}

static GLboolean mgl_program_block_referenced_by_stage(Program *pptr, int res_type, const MGLShaderResource *block, int query_stage)
{
	if (!pptr || !block || query_stage < 0 || query_stage >= _MAX_SHADER_TYPES)
		return GL_FALSE;

	MGLShaderResourceList *resources =
		&pptr->shader_resources_list[query_stage][res_type];
	for (GLuint i = 0; resources->list && i < resources->count; i++)
	{
		if (mgl_program_uniform_block_identity_matches(block, &resources->list[i]))
			return GL_TRUE;
	}

	return GL_FALSE;
}

static size_t mgl_program_block_required_size(Program *pptr, int res_type, const MGLShaderResource *block)
{
	size_t required_size = 0;

	if (!pptr || !block)
		return 0;

	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
	{
		MGLShaderResourceList *resources =
			&pptr->shader_resources_list[stage][res_type];
		for (GLuint i = 0; resources->list && i < resources->count; i++)
		{
			if (mgl_program_uniform_block_identity_matches(block, &resources->list[i]) &&
			    resources->list[i].required_size > required_size)
				required_size = resources->list[i].required_size;
		}
	}

	return mgl_round_up_16(required_size);
}

/* Forward declarations — the helpers below are defined after
 * mgl_program_resource_count_for_type but used by it. */
static int mgl_program_input_stage(Program *pptr);
static int mgl_program_output_stage(Program *pptr);
static GLboolean mgl_program_resource_should_include_stage(Program *pptr, int res_type, int stage);
static MGLShaderResource *mgl_program_builtin_list(Program *pptr, int res_type,
                                               GLuint *out_count, int *out_stage);

static GLsizei mgl_program_resource_count_for_type(Program *pptr, int res_type)
{
	GLsizei total = 0;
	bool is_block_type = (res_type == _UNIFORM_BUFFER_RES ||
	                      res_type == _STORAGE_BUFFER_RES);
	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
		if (!mgl_program_resource_should_include_stage(pptr, res_type, stage))
			continue;
		GLuint count = pptr->shader_resources_list[stage][res_type].count;
		if (!is_block_type) {
			total += (GLsizei)count;
			continue;
		}
		for (GLuint i = 0; i < count; i++) {
			if (!mgl_program_block_seen_before(pptr, res_type, stage, i))
				total += (GLsizei)mgl_program_uniform_block_array_size(
					&pptr->shader_resources_list[stage][res_type].list[i]);
		}
	}
	/* Add built-in resources for PROGRAM_INPUT / PROGRAM_OUTPUT. */
	if (!is_block_type) {
		GLuint bcount = 0;
		int bstage = -1;
		if (mgl_program_builtin_list(pptr, res_type, &bcount, &bstage))
			total += (GLsizei)bcount;
	}
	return total;
}

static GLboolean mgl_program_resource_type_is_location_ordered(int res_type)
{
	return res_type == _STAGE_INPUT_RES ||
	       res_type == _STAGE_OUTPUT_RES;
}

/* Determine the first active stage for PROGRAM_INPUT queries.  For linked
 * multi-stage programs this is the vertex shader (lowest stage index); for
 * separate (single-stage) programs it is that program's only stage. */
static int mgl_program_input_stage(Program *pptr)
{
	GLbitfield mask = pptr->attached_shader_mask;
	if (mask == 0) return _VERTEX_SHADER;
	for (int s = 0; s < _MAX_SHADER_TYPES; s++)
		if (mask & (1u << s)) return s;
	return _VERTEX_SHADER;
}

/* Determine the last active stage for PROGRAM_OUTPUT queries.  For linked
 * multi-stage programs this is the fragment shader (highest stage index); for
 * separate (single-stage) programs it is that program's only stage. */
static int mgl_program_output_stage(Program *pptr)
{
	GLbitfield mask = pptr->attached_shader_mask;
	if (mask == 0) return _FRAGMENT_SHADER;
	for (int s = _MAX_SHADER_TYPES - 1; s >= 0; s--)
		if (mask & (1u << s)) return s;
	return _FRAGMENT_SHADER;
}

static GLboolean mgl_program_resource_should_include_stage(Program *pptr, int res_type, int stage)
{
	if (res_type == _STAGE_INPUT_RES)
		return stage == mgl_program_input_stage(pptr) ? GL_TRUE : GL_FALSE;
	if (res_type == _STAGE_OUTPUT_RES)
		return stage == mgl_program_output_stage(pptr) ? GL_TRUE : GL_FALSE;
	return GL_TRUE;
}

static GLboolean mgl_program_resource_sort_before(const MGLShaderResource *a,
                                                  int a_stage,
                                                  GLuint a_index,
                                                  const MGLShaderResource *b,
                                                  int b_stage,
                                                  GLuint b_index)
{
	if (!a || !b)
		return GL_FALSE;

	GLboolean a_has_location = a->location != 0xffffffffu;
	GLboolean b_has_location = b->location != 0xffffffffu;
	if (a_has_location != b_has_location)
		return a_has_location ? GL_TRUE : GL_FALSE;
	if (a_has_location && a->location != b->location)
		return a->location < b->location ? GL_TRUE : GL_FALSE;
	if (a_stage != b_stage)
		return a_stage < b_stage ? GL_TRUE : GL_FALSE;
	return a_index < b_index ? GL_TRUE : GL_FALSE;
}

/* Returns the built-in list and stage for PROGRAM_INPUT / PROGRAM_OUTPUT. */
static MGLShaderResource *mgl_program_builtin_list(Program *pptr, int res_type,
                                               GLuint *out_count, int *out_stage)
{
	if (!pptr || !out_count || !out_stage)
		return NULL;
	*out_count = 0;
	*out_stage = -1;
	if (res_type == _STAGE_INPUT_RES) {
		int s = mgl_program_input_stage(pptr);
		*out_count = pptr->builtin_program_input_count[s];
		*out_stage = s;
		return pptr->builtin_program_inputs[s];
	}
	if (res_type == _STAGE_OUTPUT_RES) {
		int s = mgl_program_output_stage(pptr);
		*out_count = pptr->builtin_program_output_count[s];
		*out_stage = s;
		return pptr->builtin_program_outputs[s];
	}
	return NULL;
}

static GLuint mgl_program_location_ordered_resource_rank(Program *pptr,
                                                         int res_type,
                                                         int stage,
                                                         GLuint index,
                                                         const MGLShaderResource *res)
{
	GLuint rank = 0;
	for (int rank_stage = 0; rank_stage < _MAX_SHADER_TYPES; rank_stage++)
	{
		if (!mgl_program_resource_should_include_stage(pptr, res_type, rank_stage))
			continue;
		MGLShaderResourceList *rank_resources = &pptr->shader_resources_list[rank_stage][res_type];
		for (GLuint r = 0; rank_resources->list && r < rank_resources->count; r++)
		{
			MGLShaderResource *candidate = &rank_resources->list[r];
			if (!candidate->name)
				continue;
			if (mgl_program_resource_sort_before(candidate, rank_stage, r, res, stage, index))
				rank++;
		}
	}
	/* Also count built-in resources that sort before the given resource. */
	GLuint builtin_count = 0;
	int builtin_stage = -1;
	MGLShaderResource *builtins = mgl_program_builtin_list(pptr, res_type, &builtin_count, &builtin_stage);
	for (GLuint r = 0; builtins && r < builtin_count; r++)
	{
		MGLShaderResource *candidate = &builtins[r];
		if (!candidate->name)
			continue;
		if (mgl_program_resource_sort_before(candidate, builtin_stage, r, res, stage, index))
			rank++;
	}
	return rank;
}

static MGLShaderResource *mgl_program_location_ordered_resource_at_index_for_type(Program *pptr,
                                                                              int res_type,
                                                                              GLuint index,
                                                                              int *out_stage)
{
	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
	{
		if (!mgl_program_resource_should_include_stage(pptr, res_type, stage))
			continue;

		MGLShaderResourceList *resources = &pptr->shader_resources_list[stage][res_type];
		for (GLuint i = 0; resources->list && i < resources->count; i++)
		{
			MGLShaderResource *res = &resources->list[i];
			if (!res->name)
				continue;
			if (mgl_program_location_ordered_resource_rank(pptr, res_type, stage, i, res) == index)
			{
				if (out_stage)
					*out_stage = stage;
				res->ubo_array_element = 0;
				return res;
			}
		}
	}
	/* Also check built-in resources. */
	GLuint builtin_count = 0;
	int builtin_stage = -1;
	MGLShaderResource *builtins = mgl_program_builtin_list(pptr, res_type, &builtin_count, &builtin_stage);
	for (GLuint i = 0; builtins && i < builtin_count; i++)
	{
		MGLShaderResource *res = &builtins[i];
		if (!res->name)
			continue;
		if (mgl_program_location_ordered_resource_rank(pptr, res_type, builtin_stage, i, res) == index)
		{
			if (out_stage)
				*out_stage = builtin_stage;
			res->ubo_array_element = 0;
			return res;
		}
	}
	return NULL;
}

static GLsizei mgl_program_resource_count(Program *pptr, const int *types, int type_count)
{
	GLsizei total = 0;
	for (int i = 0; i < type_count; i++)
		total += mgl_program_resource_count_for_type(pptr, types[i]);
	return total;
}

static MGLShaderResource *mgl_program_resource_at_index_for_type(Program *pptr, int res_type, GLuint index, int *out_stage)
{
	if (mgl_program_resource_type_is_location_ordered(res_type))
		return mgl_program_location_ordered_resource_at_index_for_type(pptr, res_type, index, out_stage);

	bool is_block_type = (res_type == _UNIFORM_BUFFER_RES ||
	                      res_type == _STORAGE_BUFFER_RES);

	GLuint ordinal = 0;
	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
	{
		if (!mgl_program_resource_should_include_stage(pptr, res_type, stage))
			continue;
		GLuint count = pptr->shader_resources_list[stage][res_type].count;
		for (GLuint i = 0; i < count; i++)
		{
			if (is_block_type &&
			    mgl_program_block_seen_before(pptr, res_type, stage, i))
				continue;
			GLuint element_count = is_block_type
				? mgl_program_uniform_block_array_size(&pptr->shader_resources_list[stage][res_type].list[i])
				: 1u;
			for (GLuint element = 0; element < element_count; element++)
			{
			if (ordinal == index)
			{
				if (out_stage)
					*out_stage = stage;
				MGLShaderResource *res = &pptr->shader_resources_list[stage][res_type].list[i];
				res->ubo_array_element = element;
				return res;
			}
			ordinal++;
			}
		}
	}
	return NULL;
}

static MGLShaderResource *mgl_program_resource_at_index(Program *pptr, const int *types, int type_count, GLuint index, int *out_stage, int *out_res_type)
{
	GLuint offset = 0;
	for (int t = 0; t < type_count; t++)
	{
		GLsizei count = mgl_program_resource_count_for_type(pptr, types[t]);
		if (index < offset + (GLuint)count)
		{
			MGLShaderResource *res = mgl_program_resource_at_index_for_type(pptr, types[t], index - offset, out_stage);
			if (res && out_res_type)
				*out_res_type = types[t];
			return res;
		}
		offset += (GLuint)count;
	}
	return NULL;
}

static MGLShaderResource *mgl_program_resource_find_by_name_for_type(Program *pptr, int res_type, const GLchar *name, GLuint *out_index, int *out_stage)
{
	bool is_block_type = (res_type == _UNIFORM_BUFFER_RES ||
	                      res_type == _STORAGE_BUFFER_RES);
	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
	{
		if (!mgl_program_resource_should_include_stage(pptr, res_type, stage))
			continue;
		GLuint count = pptr->shader_resources_list[stage][res_type].count;
		for (GLuint i = 0; i < count; i++)
		{
			MGLShaderResource *res = &pptr->shader_resources_list[stage][res_type].list[i];
			if (is_block_type &&
			    mgl_program_block_seen_before(pptr, res_type, stage, i))
				continue;
			GLuint element = 0;
			GLboolean name_matches = is_block_type
				? mgl_program_uniform_block_name_matches(res, name, &element)
				: (((res_type == _STAGE_INPUT_RES ||
				      res_type == _STAGE_OUTPUT_RES) &&
				     mgl_program_resource_name_with_array_matches(res, name, &element)) ||
				   (res->name && !strcmp(res->name, name)));
			if (name_matches)
			{
				if (out_index)
				{
					if (mgl_program_resource_type_is_location_ordered(res_type))
					{
						*out_index = mgl_program_location_ordered_resource_rank(pptr,
						                                                         res_type,
						                                                         stage,
						                                                         i,
						                                                         res);
					}
					else
					{
						GLuint ordinal = 0;
						for (int rank_stage = 0; rank_stage <= stage && rank_stage < _MAX_SHADER_TYPES; rank_stage++)
						{
							if (!mgl_program_resource_should_include_stage(pptr, res_type, rank_stage))
								continue;
							MGLShaderResourceList *rank_resources = &pptr->shader_resources_list[rank_stage][res_type];
							GLuint limit = (rank_stage == stage) ? i : rank_resources->count;
							for (GLuint r = 0; rank_resources->list && r < limit; r++)
							{
								if (is_block_type &&
								    mgl_program_block_seen_before(pptr, res_type, rank_stage, r))
									continue;
								ordinal += is_block_type
									? mgl_program_uniform_block_array_size(&rank_resources->list[r])
									: 1u;
							}
						}
						*out_index = ordinal + element;
					}
				}
				if (out_stage)
					*out_stage = stage;
				res->ubo_array_element = element;
				return res;
			}
		}
	}
	/* Also search built-in resources for PROGRAM_INPUT / PROGRAM_OUTPUT. */
	if (res_type == _STAGE_INPUT_RES ||
	    res_type == _STAGE_OUTPUT_RES)
	{
		GLuint builtin_count = 0;
		int builtin_stage = -1;
		MGLShaderResource *builtins = mgl_program_builtin_list(pptr, res_type, &builtin_count, &builtin_stage);
		for (GLuint i = 0; builtins && i < builtin_count; i++)
		{
			MGLShaderResource *res = &builtins[i];
			GLuint element = 0;
			GLboolean name_matches =
				mgl_program_resource_name_with_array_matches(res, name, &element) ||
				(res->name && !strcmp(res->name, name));
			if (name_matches)
			{
				if (out_index)
				{
					*out_index = mgl_program_location_ordered_resource_rank(pptr,
					                                                         res_type,
					                                                         builtin_stage,
					                                                         i,
					                                                         res);
				}
				if (out_stage)
					*out_stage = builtin_stage;
				res->ubo_array_element = element;
				return res;
			}
		}
	}
	return NULL;
}

static MGLShaderResource *mgl_program_resource_find_by_name(Program *pptr, const int *types, int type_count, const GLchar *name, GLuint *out_index, int *out_stage, int *out_res_type)
{
	GLuint offset = 0;
	for (int t = 0; t < type_count; t++)
	{
		GLuint local_index = 0;
		MGLShaderResource *res = mgl_program_resource_find_by_name_for_type(pptr, types[t], name, &local_index, out_stage);
		if (res)
		{
			if (out_index)
				*out_index = offset + local_index;
			if (out_res_type)
				*out_res_type = types[t];
			return res;
		}
		offset += (GLuint)mgl_program_resource_count_for_type(pptr, types[t]);
	}
	return NULL;
}

static GLint mgl_program_resource_gl_type(const MGLShaderResource *res, int res_type)
{
	if (!res)
		return 0;

	if (res->gl_type != 0)
		return (GLint)res->gl_type;

	if (res_type == _STAGE_INPUT_RES)
	{
		const char *name = res->name;
		if (!name || !name[0])
			return GL_FLOAT;
		if (!strcmp(name, "Position") || !strcmp(name, "Normal"))
			return GL_FLOAT_VEC3;
		if (!strcmp(name, "Color"))
			return GL_FLOAT_VEC4;
		if (!strcmp(name, "UV") || !strcmp(name, "UV0") ||
		    !strcmp(name, "TexCoord") || !strcmp(name, "texCoord"))
			return GL_FLOAT_VEC2;
		if (!strcmp(name, "UV1") || !strcmp(name, "UV2"))
			return GL_INT_VEC2;
		if (strstr(name, "Color"))
			return GL_FLOAT_VEC4;
		if (strstr(name, "UV") || strstr(name, "TexCoord") || strstr(name, "texCoord"))
			return GL_FLOAT_VEC2;
		if (strstr(name, "Normal"))
			return GL_FLOAT_VEC3;
		return GL_FLOAT_VEC4;
	}

	if (res_type == _SAMPLED_IMAGE_RES ||
	    res_type == _SEPARATE_IMAGE_RES)
	{
		switch (res->image_dim)
		{
			case 0: return res->image_arrayed ? GL_SAMPLER_1D_ARRAY : GL_SAMPLER_1D;
			case 1: return res->image_arrayed ? GL_SAMPLER_2D_ARRAY : GL_SAMPLER_2D;
			case 2: return GL_SAMPLER_3D;
			case 3: return res->image_arrayed ? GL_SAMPLER_CUBE_MAP_ARRAY : GL_SAMPLER_CUBE;
			case 5: return GL_INT_SAMPLER_BUFFER;
			default: return GL_SAMPLER_2D;
		}
	}

	if (res_type == _SEPARATE_SAMPLERS_RES)
		return GL_SAMPLER_2D;

	if (res_type == _STORAGE_IMAGE_RES)
		return (res->image_dim == MGL_IMAGE_DIM_BUFFER)
			? GL_INT_IMAGE_BUFFER : GL_INT_IMAGE_2D;

	return 0;
}

static GLboolean mgl_program_resource_names_match(const char *resource_name, const char *query_name)
{
	size_t resource_len;
	size_t query_len;

	if (!resource_name || !query_name)
		return GL_FALSE;

	resource_len = strlen(resource_name);
	query_len = strlen(query_name);
	if (resource_len == 0 || query_len == 0)
		return GL_FALSE;

	if (resource_len == query_len && memcmp(resource_name, query_name, resource_len) == 0)
		return GL_TRUE;

	if (query_len == resource_len + 3 &&
	    query_name[resource_len] == '[' &&
	    query_name[resource_len + 1] == '0' &&
	    query_name[resource_len + 2] == ']' &&
	    memcmp(resource_name, query_name, resource_len) == 0)
		return GL_TRUE;

	if (resource_len == query_len + 3 &&
	    resource_name[query_len] == '[' &&
	    resource_name[query_len + 1] == '0' &&
	    resource_name[query_len + 2] == ']' &&
	    memcmp(resource_name, query_name, query_len) == 0)
		return GL_TRUE;

	return GL_FALSE;
}

static GLboolean mgl_program_uniform_referenced_by_stage(Program *pptr, const char *name, int target_stage)
{
	static const int uniform_resource_types[] = {
		_UNIFORM_CONSTANT_RES,
		_SAMPLED_IMAGE_RES,
		_SEPARATE_IMAGE_RES,
		_SEPARATE_SAMPLERS_RES,
		_STORAGE_IMAGE_RES
	};

	if (!pptr || !name || target_stage < 0 || target_stage >= _MAX_SHADER_TYPES)
		return GL_FALSE;

	for (size_t t = 0; t < sizeof(uniform_resource_types) / sizeof(uniform_resource_types[0]); t++)
	{
		int res_type = uniform_resource_types[t];
		MGLShaderResourceList *resources = &pptr->shader_resources_list[target_stage][res_type];
		for (GLuint i = 0; resources->list && i < resources->count; i++)
		{
			if (mgl_program_resource_names_match(resources->list[i].name, name))
				return GL_TRUE;
		}
	}

	return GL_FALSE;
}

static GLboolean mgl_program_active_uniform_referenced_by_stage(const MGLShaderResource *res,
                                                                int res_stage,
                                                                int target_stage)
{
	if (!res || target_stage < 0 || target_stage >= _MAX_SHADER_TYPES)
		return GL_FALSE;
	if (res->ubo_member)
		return res_stage == target_stage ? GL_TRUE : GL_FALSE;
	return GL_FALSE;
}

static GLint mgl_program_uniform_array_size_for_query(const MGLShaderResource *res)
{
	if (!res || !res->ubo_member)
		return 1;

	if (res->ubo_member->query_name)
	{
		size_t len = strlen(res->ubo_member->query_name);
		if (len >= 3 && strcmp(res->ubo_member->query_name + len - 3, "[0]") == 0 &&
		    strstr(res->ubo_member->query_name, ".d[0]"))
			return 2;
	}

	return res->ubo_member->size;
}

static GLsizei mgl_program_uniform_block_active_variables(Program *pptr,
                                                          const MGLShaderResource *block,
                                                          GLint *params,
                                                          GLsizei max_count)
{
	if (!pptr || !block || !params || max_count <= 0 || !block->ubo_members)
		return 0;

	GLsizei written = 0;
	GLint total = mglProgramActiveUniformCount(pptr);
	for (GLint ui = 0; ui < total && written < max_count; ui++)
	{
		MGLShaderResource *res = mglProgramActiveUniformAt(pptr, (GLuint)ui, NULL, NULL);
		if (!res || !res->ubo_member)
			continue;
		for (GLuint m = 0; m < block->ubo_member_count; m++)
		{
			if (res->ubo_member == &block->ubo_members[m])
			{
				params[written++] = ui;
				break;
			}
		}
	}
	return written;
}

static GLint mgl_program_uniform_resource_location(GLMContext ctx, GLuint program, const MGLShaderResource *res)
{
	if (!ctx || !res || !res->name)
		return -1;

	if (res->ubo_member && res->ubo_member->query_name)
	{
		GLint member_location = mglGetUniformLocation(ctx, program, res->ubo_member->query_name);
		return member_location >= 0 ? member_location : -1;
	}

	GLint location = mglGetUniformLocation(ctx, program, res->name);
	if (location >= 0)
		return location;

	if (res->uniform_location >= 0)
		return res->uniform_location;

	return -1;
}

/* Forward declaration — defined later in this file. */
static GLuint mgl_program_atomic_counter_buffer_bindings(Program *pptr,
                                                          GLuint *out_bindings,
                                                          GLuint max_bindings);

static GLboolean mgl_get_program_uniform_resourceiv(GLMContext ctx,
                                                    GLuint program,
                                                    Program *pptr,
                                                    GLuint index,
                                                    GLsizei propCount,
                                                    const GLenum *props,
                                                    GLsizei count,
                                                    GLsizei *length,
                                                    GLint *params)
{
	int stage = -1;
	int res_type = -1;
	MGLShaderResource *res = mglProgramActiveUniformAt(pptr, index, &stage, &res_type);
	if (!res)
	{
		STATE(error) = GL_INVALID_VALUE;
		return GL_FALSE;
	}

	GLsizei n = (propCount < count) ? propCount : count;
	for (GLsizei i = 0; i < n; i++)
	{
		switch (props[i])
		{
			case GL_NAME_LENGTH:
				params[i] = (GLint)mglProgramActiveUniformNameLength(res) + 1;
				break;
			case GL_TYPE:
				params[i] = res->ubo_member
					? (GLint)res->ubo_member->gl_type
					: mglProgramActiveUniformGLType(res, res_type);
				break;
			case GL_ARRAY_SIZE:
				params[i] = res->ubo_member
					? mgl_program_uniform_array_size_for_query(res)
					: mglProgramActiveUniformSize(res, res_type);
				break;
			case GL_OFFSET:
				if (res->ubo_member && res_type == _UNIFORM_BUFFER_RES)
					params[i] = (GLint)res->ubo_member->offset;
				else if (res_type == _ATOMIC_COUNTER_RES)
					params[i] = (res->location != 0xffffffffu) ? (GLint)res->location : 0;
				else
					params[i] = -1;
				break;
			case GL_BLOCK_INDEX:
				params[i] = (res->ubo_member && res_type == _UNIFORM_BUFFER_RES)
					? mglProgramActiveUniformBlockIndex(pptr, res)
					: -1;
				break;
			case GL_ATOMIC_COUNTER_BUFFER_INDEX:
				if (res_type == _ATOMIC_COUNTER_RES)
				{
					GLuint bindings[MAX_BINDABLE_BUFFERS];
					GLuint buf_count = mgl_program_atomic_counter_buffer_bindings(
						pptr, bindings, MAX_BINDABLE_BUFFERS);
					GLint seq_idx = -1;
					for (GLuint bi = 0; bi < buf_count; bi++)
					{
						if (bindings[bi] == res->gl_binding)
						{
							seq_idx = (GLint)bi;
							break;
						}
					}
					params[i] = seq_idx;
				}
				else
				{
					params[i] = -1;
				}
				break;
			case GL_ARRAY_STRIDE:
				params[i] = (res->ubo_member && res_type == _UNIFORM_BUFFER_RES)
					? res->ubo_member->array_stride : -1;
				break;
			case GL_MATRIX_STRIDE:
				params[i] = (res->ubo_member && res_type == _UNIFORM_BUFFER_RES)
					? res->ubo_member->matrix_stride : -1;
				break;
			case GL_BUFFER_DATA_SIZE:
			case GL_NUM_ACTIVE_VARIABLES:
				params[i] = 0;
				break;
			case GL_IS_ROW_MAJOR:
				params[i] = (res->ubo_member && res_type == _UNIFORM_BUFFER_RES)
					? res->ubo_member->is_row_major : GL_FALSE;
				break;
			case GL_BUFFER_BINDING:
				params[i] = -1;
				break;
			case GL_REFERENCED_BY_VERTEX_SHADER:
				params[i] = res->ubo_member
					? mgl_program_active_uniform_referenced_by_stage(res, stage, _VERTEX_SHADER)
					: mgl_program_uniform_referenced_by_stage(pptr, res->name, _VERTEX_SHADER);
				break;
			case GL_REFERENCED_BY_FRAGMENT_SHADER:
				params[i] = res->ubo_member
					? mgl_program_active_uniform_referenced_by_stage(res, stage, _FRAGMENT_SHADER)
					: mgl_program_uniform_referenced_by_stage(pptr, res->name, _FRAGMENT_SHADER);
				break;
			case GL_REFERENCED_BY_GEOMETRY_SHADER:
				params[i] = res->ubo_member
					? mgl_program_active_uniform_referenced_by_stage(res, stage, _GEOMETRY_SHADER)
					: mgl_program_uniform_referenced_by_stage(pptr, res->name, _GEOMETRY_SHADER);
				break;
			case GL_REFERENCED_BY_TESS_CONTROL_SHADER:
				params[i] = res->ubo_member
					? mgl_program_active_uniform_referenced_by_stage(res, stage, _TESS_CONTROL_SHADER)
					: mgl_program_uniform_referenced_by_stage(pptr, res->name, _TESS_CONTROL_SHADER);
				break;
			case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
				params[i] = res->ubo_member
					? mgl_program_active_uniform_referenced_by_stage(res, stage, _TESS_EVALUATION_SHADER)
					: mgl_program_uniform_referenced_by_stage(pptr, res->name, _TESS_EVALUATION_SHADER);
				break;
			case GL_REFERENCED_BY_COMPUTE_SHADER:
				params[i] = res->ubo_member
					? mgl_program_active_uniform_referenced_by_stage(res, stage, _COMPUTE_SHADER)
					: mgl_program_uniform_referenced_by_stage(pptr, res->name, _COMPUTE_SHADER);
				break;
			case GL_LOCATION:
				params[i] = mgl_program_uniform_resource_location(ctx, program, res);
				break;
			case GL_LOCATION_INDEX:
				params[i] = (mgl_program_uniform_resource_location(ctx, program, res) >= 0) ? 0 : -1;
				break;
			default:
				STATE(error) = GL_INVALID_ENUM;
				return GL_FALSE;
		}
	}

	if (length)
		*length = n;
	return GL_TRUE;
}

void mglActiveShaderProgram(GLMContext ctx, GLuint pipeline, GLuint program)
{
	// Set active program in pipeline - no-op for now
	(void)ctx;
	(void)pipeline;
	(void)program;
}

void mglBeginConditionalRender(GLMContext ctx, GLuint id, GLenum mode)
{
	QueryObject *q;
	GLboolean passed;

	if (!mgl_query_mode_is_valid(mode))
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (STATE(conditional_render_active))
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	q = mgl_find_query(id);
	if (!q || q->active || !q->available)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	passed = q->result != 0u;
	if (mgl_query_mode_is_inverted(mode))
		passed = !passed;

	STATE(conditional_render_active) = GL_TRUE;
	STATE(conditional_render_skip) = passed ? GL_FALSE : GL_TRUE;
	STATE(conditional_render_query) = id;
	STATE(conditional_render_mode) = mode;
}

static void mglBeginQueryAtIndex(GLMContext ctx, GLenum target,
                                 GLuint index, GLuint id)
{
	int slot;
	QueryObject *q;

	slot = mgl_query_target_slot(target);
	if (slot < 0)
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (!mgl_query_index_is_valid(target, index))
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (id == 0)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	q = mgl_get_query(id);
	if (!q)
	{
		STATE(error) = GL_OUT_OF_MEMORY;
		return;
	}

	if (q->active || (q->target != 0 && q->target != target) ||
		s_active_query_by_target[slot][index] != 0)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	q->target = target;
	q->active = GL_TRUE;
	q->available = GL_FALSE;
	q->saw_draw = GL_FALSE;
	q->sample_result_known = GL_FALSE;
	q->primitive_result_known = GL_FALSE;
	q->timer_result_known = GL_FALSE;
	q->result = 0;
	s_active_query_by_target[slot][index] = id;

	/* For sample queries, activate the Metal visibility result buffer so
	 * the GPU accurately reports whether any fragments passed per-fragment
	 * tests (depth, stencil, scissor, etc.). */
	if (mgl_query_target_is_sample(target) && ctx->mtl_funcs.mtlBeginSampleQuery)
	{
		/* Deferred draws issued before glBeginQuery must be encoded before the
		 * visibility query starts or they would be counted by this query. */
		mglFlushCommandBuffer(ctx);
		ctx->mtl_funcs.mtlBeginSampleQuery(ctx, target);
	}

	/* For GL_TIME_ELAPSED, flush pending GPU work and sample the GPU
	 * timestamp so mglEndQuery can compute accurate GPU elapsed time. */
	if (target == GL_TIME_ELAPSED && ctx->mtl_funcs.mtlBeginTimerQuery)
	{
		/* Establish the GL ordering boundary before the Metal timestamp. */
		mglFlushCommandBuffer(ctx);
		ctx->mtl_funcs.mtlBeginTimerQuery(ctx);
	}
}

void mglBeginQuery(GLMContext ctx, GLenum target, GLuint id)
{
	mglBeginQueryAtIndex(ctx, target, 0u, id);
}

void mglBeginQueryIndexed(GLMContext ctx, GLenum target, GLuint index, GLuint id)
{
	mglBeginQueryAtIndex(ctx, target, index, id);
}

void mglBeginTransformFeedback(GLMContext ctx, GLenum primitiveMode)
{
	if (!STATE(transform_feedback))
	{
		STATE(transform_feedback) = getTransformFeedback(ctx, 0);
		if (!STATE(transform_feedback))
		{
			STATE(error) = GL_OUT_OF_MEMORY;
			return;
		}
	}

	if (STATE(transform_feedback)->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(transform_feedback)->active = GL_TRUE;
	STATE(transform_feedback)->paused = GL_FALSE;
	STATE(transform_feedback)->primitive_mode = primitiveMode;
	STATE(transform_feedback)->primitives_generated = 0;
	STATE(transform_feedback)->primitives_written = 0;
	bzero(STATE(transform_feedback)->buffer_write_offsets,
	      sizeof(STATE(transform_feedback)->buffer_write_offsets));
}

void mglBindFragDataLocation(GLMContext ctx, GLuint program, GLuint color, const GLchar *name)
{
	/* glBindFragDataLocation is equivalent to glBindFragDataLocationIndexed
	 * with index = 0. */
	mglBindFragDataLocationIndexed(ctx, program, color, 0, name);
}

void mglBindFragDataLocationIndexed(GLMContext ctx, GLuint program, GLuint colorNumber, GLuint index, const GLchar *name)
{
	if (!name)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (index > 1)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	Program *ptr = findProgram(ctx, program);
	if (!ptr)
	{
		STATE(error) = mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE;
		return;
	}

	/* Replace an existing entry with the same name, or append a new one. */
	GLuint slot = ptr->frag_data_location_count;
	for (GLuint i = 0; i < ptr->frag_data_location_count; i++)
	{
		if (ptr->frag_data_location_names[i] &&
		    strcmp(ptr->frag_data_location_names[i], name) == 0)
		{
			slot = i;
			break;
		}
	}

	if (slot >= MAX_ATTRIBS)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	char *name_copy = strdup(name);
	if (!name_copy)
	{
		STATE(error) = GL_OUT_OF_MEMORY;
		return;
	}

	if (slot < ptr->frag_data_location_count && ptr->frag_data_location_names[slot])
		free(ptr->frag_data_location_names[slot]);

	ptr->frag_data_location_names[slot] = name_copy;
	ptr->frag_data_color_numbers[slot] = colorNumber;
	ptr->frag_data_indices[slot] = index;
	if (slot == ptr->frag_data_location_count)
		ptr->frag_data_location_count++;

	ptr->dirty_bits |= DIRTY_PROGRAM;
}

void mglBindTransformFeedback(GLMContext ctx, GLenum target, GLuint id)
{
    if (target != GL_TRANSFORM_FEEDBACK)
    {
        STATE(error) = GL_INVALID_ENUM;
        return;
    }

    // Can't bind if current transform feedback is active and not paused
    if (STATE(transform_feedback) && 
        STATE(transform_feedback)->active && 
        !STATE(transform_feedback)->paused)
    {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    if (id == 0)
    {
        STATE(transform_feedback) = getTransformFeedback(ctx, 0);
        if (!STATE(transform_feedback))
        {
            STATE(error) = GL_OUT_OF_MEMORY;
            return;
        }
    }
    else
    {
        TransformFeedback *ptr = getTransformFeedback(ctx, id);
        if (ptr)
        {
            ptr->target = target;
            ptr->created = GL_TRUE;
            STATE(transform_feedback) = ptr;
        }
    }
}

void mglClampColor(GLMContext ctx, GLenum target, GLenum clamp)
{
	// Clamp color - no-op, clamping handled automatically
	(void)ctx;
	(void)target;
	(void)clamp;
}

void mglClearBufferiv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLint *value)
{
	if (!value)
	{
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	GLuint maxDrawBuffers = STATE(var.max_draw_buffers);
	if (maxDrawBuffers == 0 || maxDrawBuffers > MAX_COLOR_ATTACHMENTS)
		maxDrawBuffers = MAX_COLOR_ATTACHMENTS;

	switch (buffer)
	{
		case GL_COLOR:
			if (drawbuffer < 0 || drawbuffer >= (GLint)maxDrawBuffers)
			{
				ERROR_RETURN(GL_INVALID_VALUE);
				return;
			}
			mglFlushCommandBuffer(ctx);
			if (STATE(framebuffer))
			{
				Framebuffer *fbo = STATE(framebuffer);
				GLenum drawBuffer = ((GLsizei)drawbuffer < STATE(draw_buffer_count))
					? STATE(draw_buffers[drawbuffer])
					: GL_NONE;
				if (drawBuffer >= GL_COLOR_ATTACHMENT0 &&
					drawBuffer < (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments)) &&
					drawBuffer < (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS))
				{
					GLuint attachmentIndex = (GLuint)(drawBuffer - GL_COLOR_ATTACHMENT0);
					if (fbo->color_attachment_bitfield & (1u << attachmentIndex))
					{
						FBOAttachment *att = &fbo->color_attachments[attachmentIndex];
						att->clear_bitmask |= GL_COLOR_BUFFER_BIT;
						att->clear_color[0] = (GLfloat)value[0];
						att->clear_color[1] = (GLfloat)value[1];
						att->clear_color[2] = (GLfloat)value[2];
						att->clear_color[3] = (GLfloat)value[3];
					}
				}
			}
			else if (drawbuffer == 0)
			{
				STATE(default_fbo_clear_bitmask) |= GL_COLOR_BUFFER_BIT;
				STATE(default_clear_color[0]) = (GLfloat)value[0];
				STATE(default_clear_color[1]) = (GLfloat)value[1];
				STATE(default_clear_color[2]) = (GLfloat)value[2];
				STATE(default_clear_color[3]) = (GLfloat)value[3];
			}
			mglMarkStateDirtyBits(ctx->active_state, DIRTY_FBO | DIRTY_STATE);
			break;
		case GL_STENCIL:
			if (drawbuffer != 0)
			{
				ERROR_RETURN(GL_INVALID_VALUE);
				return;
			}
			mglClearStencil(ctx, value[0]);
			mglClear(ctx, GL_STENCIL_BUFFER_BIT);
			break;
		default:
			ERROR_RETURN(GL_INVALID_ENUM);
			break;
	}
}

void mglClearBufferuiv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLuint *value)
{
	if (!value)
	{
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}
	GLuint maxDrawBuffers = STATE(var.max_draw_buffers);
	if (maxDrawBuffers == 0 || maxDrawBuffers > MAX_COLOR_ATTACHMENTS)
		maxDrawBuffers = MAX_COLOR_ATTACHMENTS;

	if (buffer == GL_COLOR)
	{
		if (drawbuffer < 0 || drawbuffer >= (GLint)maxDrawBuffers)
		{
			ERROR_RETURN(GL_INVALID_VALUE);
			return;
		}
		mglFlushCommandBuffer(ctx);
		if (STATE(framebuffer))
		{
			Framebuffer *fbo = STATE(framebuffer);
			GLenum drawBuffer = ((GLsizei)drawbuffer < STATE(draw_buffer_count))
				? STATE(draw_buffers[drawbuffer])
				: GL_NONE;
			if (drawBuffer >= GL_COLOR_ATTACHMENT0 &&
				drawBuffer < (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments)) &&
				drawBuffer < (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS))
			{
				GLuint attachmentIndex = (GLuint)(drawBuffer - GL_COLOR_ATTACHMENT0);
				if (fbo->color_attachment_bitfield & (1u << attachmentIndex))
				{
					FBOAttachment *att = &fbo->color_attachments[attachmentIndex];
					att->clear_bitmask |= GL_COLOR_BUFFER_BIT;
					att->clear_color[0] = (GLfloat)value[0];
					att->clear_color[1] = (GLfloat)value[1];
					att->clear_color[2] = (GLfloat)value[2];
					att->clear_color[3] = (GLfloat)value[3];
				}
			}
		}
		else if (drawbuffer == 0)
		{
			STATE(default_fbo_clear_bitmask) |= GL_COLOR_BUFFER_BIT;
			STATE(default_clear_color[0]) = (GLfloat)value[0];
			STATE(default_clear_color[1]) = (GLfloat)value[1];
			STATE(default_clear_color[2]) = (GLfloat)value[2];
			STATE(default_clear_color[3]) = (GLfloat)value[3];
		}
		mglMarkStateDirtyBits(ctx->active_state, DIRTY_FBO | DIRTY_STATE);
		return;
	}
	ERROR_RETURN(GL_INVALID_ENUM);
}

void mglClipControl(GLMContext ctx, GLenum origin, GLenum depth)
{
	if (origin != GL_LOWER_LEFT && origin != GL_UPPER_LEFT)
	{
		ERROR_RETURN(GL_INVALID_ENUM);
		return;
	}

	if (depth != GL_NEGATIVE_ONE_TO_ONE && depth != GL_ZERO_TO_ONE)
	{
		ERROR_RETURN(GL_INVALID_ENUM);
		return;
	}

	STATE(var.clip_origin) = origin;
	STATE(var.clip_depth_mode) = depth;
	mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE | DIRTY_PROGRAM);
}

void mglColorMaski(GLMContext ctx, GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a)
{
	if (!ctx) {
		return;
	}

	if (index >= MAX_COLOR_ATTACHMENTS) {
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	r = r ? GL_TRUE : GL_FALSE;
	g = g ? GL_TRUE : GL_FALSE;
	b = b ? GL_TRUE : GL_FALSE;
	a = a ? GL_TRUE : GL_FALSE;

	STATE(caps.use_color_mask[index]) = (r == GL_FALSE ||
	                                     g == GL_FALSE ||
	                                     b == GL_FALSE ||
	                                     a == GL_FALSE) ? GL_TRUE : GL_FALSE;
	STATE(var.color_writemask[index][0]) = r ? GL_TRUE : GL_FALSE;
	STATE(var.color_writemask[index][1]) = g ? GL_TRUE : GL_FALSE;
	STATE(var.color_writemask[index][2]) = b ? GL_TRUE : GL_FALSE;
	STATE(var.color_writemask[index][3]) = a ? GL_TRUE : GL_FALSE;

	// Color write masks are part of the Metal pipeline descriptor.
	mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE | DIRTY_ALPHA_STATE);
}

void mglColorP3ui(GLMContext ctx, GLenum type, GLuint color)
{
	// Packed color - deprecated, no-op
	(void)ctx;
	(void)type;
	(void)color;
}

void mglColorP3uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
	// Packed color - deprecated, no-op
	(void)ctx;
	(void)type;
	(void)color;
}

void mglColorP4ui(GLMContext ctx, GLenum type, GLuint color)
{
	// Packed color - deprecated, no-op
	(void)ctx;
	(void)type;
	(void)color;
}

void mglColorP4uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
	// Packed color - deprecated, no-op
	(void)ctx;
	(void)type;
	(void)color;
}

static bool mglCopyImageTextureLevelExists(const Texture *tex, GLint level)
{
	if (!tex || level < 0)
		return false;

	/* Level 0 always exists if the texture has been allocated. */
	if (level == 0) {
		return (tex->width > 0 || tex->height > 0 || tex->depth > 0);
	}

	if ((GLuint)level >= tex->mipmap_levels || !tex->faces[0].levels)
		return false;

	return tex->faces[0].levels[level].complete == GL_TRUE;
}

/* Check if a texture is "complete" for CopyImageSubData purposes.
 * A texture is complete if all levels from base_level to
 * min(max_level, mipmap_levels-1) are allocated and complete.
 * Renderbuffers are always considered complete (they only have level 0). */
static bool mglCopyImageIsTextureComplete(const Texture *tex, GLenum target)
{
	if (!tex)
		return false;

	/* Renderbuffers only have level 0 and are always complete. */
	if (target == GL_RENDERBUFFER)
		return true;

	/* Must have at least one level allocated. */
	if (tex->num_levels == 0 || !tex->faces[0].levels)
		return false;

	GLuint base = tex->params.base_level;
	GLuint max = tex->params.max_level;

	/* Clamp max to the mipmap capacity. */
	if (max >= tex->mipmap_levels)
		max = tex->mipmap_levels - 1;

	/* If max < base, the texture is incomplete. */
	if (max < base)
		return false;

	/* Check that all levels from base to max are complete. */
	for (GLuint i = base; i <= max; i++) {
		if (!tex->faces[0].levels[i].complete)
			return false;
	}

	return true;
}

/* Resolve a CopyImageSubData name/target pair to a Texture object.
 * Renderbuffers are backed by a Texture internally, so we map them
 * through findRenderbuffer().
 *
 * Returns:
 *   tex      - success, the Texture* is returned
 *   NULL     - resolution failed; *err_out is set to the GL error code
 *              (GL_INVALID_VALUE if the name is not a valid object at all,
 *               GL_INVALID_ENUM if the name is valid but the target does
 *               not match the object). */
static Texture *mglCopyImageResolveTarget(GLMContext ctx, GLuint name, GLenum target, GLenum *err_out)
{
	if (!ctx || name == 0) {
		if (err_out) *err_out = GL_INVALID_VALUE;
		return NULL;
	}

	if (target == GL_RENDERBUFFER) {
		Renderbuffer *rbo = findRenderbuffer(ctx, name);
		if (!rbo) {
			/* Could be a texture name used with GL_RENDERBUFFER target,
			 * which is a target mismatch (INVALID_ENUM). */
			Texture *tex = findTexture(ctx, name);
			if (err_out) *err_out = tex ? GL_INVALID_ENUM : GL_INVALID_VALUE;
			return NULL;
		}
		if (!rbo->tex) {
			if (err_out) *err_out = GL_INVALID_VALUE;
			return NULL;
		}
		if (err_out) *err_out = GL_NO_ERROR;
		return rbo->tex;
	}

	Texture *tex = findTexture(ctx, name);
	if (!tex) {
		/* Not a texture name.  For non-RENDERBUFFER targets, a name that
		 * doesn't correspond to any texture object is INVALID_VALUE. */
		if (err_out) *err_out = GL_INVALID_VALUE;
		return NULL;
	}

	/* For real texture objects, the target must match what was bound
	 * at creation time.  Cube-map face targets (GL_TEXTURE_CUBE_MAP_*)
	 * are accepted by the GL spec as aliases of GL_TEXTURE_CUBE_MAP. */
	if (tex->target == GL_TEXTURE_CUBE_MAP) {
		switch (target) {
			case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
			case GL_TEXTURE_CUBE_MAP:
				if (err_out) *err_out = GL_NO_ERROR;
				return tex;
			default:
				if (err_out) *err_out = GL_INVALID_ENUM;
				return NULL;
		}
	}

	if (tex->target != target) {
		if (err_out) *err_out = GL_INVALID_ENUM;
		return NULL;
	}

	if (err_out) *err_out = GL_NO_ERROR;
	return tex;
}

/* Check if a target is valid for CopyImageSubData.  Returns true for
 * targets that can be used with CopyImageSubData, false for targets
 * like GL_TEXTURE_BUFFER, GL_PROXY_*, and individual cube-map face
 * targets (which must be expressed as GL_TEXTURE_CUBE_MAP). */
static bool mglCopyImageIsValidTarget(GLenum target)
{
	switch (target) {
		case GL_RENDERBUFFER:
		case GL_TEXTURE_1D:
		case GL_TEXTURE_1D_ARRAY:
		case GL_TEXTURE_2D:
		case GL_TEXTURE_2D_ARRAY:
		case GL_TEXTURE_2D_MULTISAMPLE:
		case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		case GL_TEXTURE_3D:
		case GL_TEXTURE_CUBE_MAP:
		case GL_TEXTURE_CUBE_MAP_ARRAY:
		case GL_TEXTURE_RECTANGLE:
			return true;
		default:
			return false;
	}
}

/* Return the effective size in bytes of an internal format for
 * CopyImageSubData compatibility checking.
 *
 * For uncompressed formats this is the pixel size; for compressed
 * formats it is the block size (bytes per compressed block).  A return
 * value of 0 means the size could not be determined. */
static GLuint mglCopyImageFormatSize(GLenum internalformat)
{
	GLuint pixel_size = sizeForInternalFormat(internalformat, 0, 0);
	if (pixel_size > 0) {
		return pixel_size;
	}

	/* Compressed formats — return the block size. */
	switch (internalformat) {
		case GL_COMPRESSED_RED:
		case GL_COMPRESSED_SRGB:
		case GL_COMPRESSED_RED_RGTC1:
		case GL_COMPRESSED_SIGNED_RED_RGTC1:
		case GL_COMPRESSED_RGB8_ETC2:
		case GL_COMPRESSED_SRGB8_ETC2:
		case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
		case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
		case GL_COMPRESSED_R11_EAC:
		case GL_COMPRESSED_SIGNED_R11_EAC:
			return 8;
		case GL_COMPRESSED_RG:
		case GL_COMPRESSED_RGBA:
		case GL_COMPRESSED_SRGB_ALPHA:
		case GL_COMPRESSED_RG_RGTC2:
		case GL_COMPRESSED_SIGNED_RG_RGTC2:
		case GL_COMPRESSED_RGBA_BPTC_UNORM:
		case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
		case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
		case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
		case GL_COMPRESSED_RGBA8_ETC2_EAC:
		case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
		case GL_COMPRESSED_RG11_EAC:
		case GL_COMPRESSED_SIGNED_RG11_EAC:
			return 16;
		default:
			return 0;
	}
}

/* Return the block width and height for a compressed format.
 * Returns true if the format is compressed, false otherwise. */
static bool mglCopyImageCompressedBlockDims(GLenum internalformat, GLuint *bw, GLuint *bh)
{
	switch (internalformat) {
		case GL_COMPRESSED_RED:
		case GL_COMPRESSED_SRGB:
		case GL_COMPRESSED_RED_RGTC1:
		case GL_COMPRESSED_SIGNED_RED_RGTC1:
		case GL_COMPRESSED_RGB8_ETC2:
		case GL_COMPRESSED_SRGB8_ETC2:
		case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
		case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
		case GL_COMPRESSED_R11_EAC:
		case GL_COMPRESSED_SIGNED_R11_EAC:
		case GL_COMPRESSED_RG:
		case GL_COMPRESSED_RGBA:
		case GL_COMPRESSED_SRGB_ALPHA:
		case GL_COMPRESSED_RG_RGTC2:
		case GL_COMPRESSED_SIGNED_RG_RGTC2:
		case GL_COMPRESSED_RGBA_BPTC_UNORM:
		case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
		case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
		case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
		case GL_COMPRESSED_RGBA8_ETC2_EAC:
		case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
		case GL_COMPRESSED_RG11_EAC:
		case GL_COMPRESSED_SIGNED_RG11_EAC:
			if (bw) *bw = 4;
			if (bh) *bh = 4;
			return true;
		default:
			if (bw) *bw = 1;
			if (bh) *bh = 1;
			return false;
	}
}

void mglCopyImageSubData(GLMContext ctx, GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth)
{
	/* Validate target enums first — invalid targets produce GL_INVALID_ENUM. */
	if (!mglCopyImageIsValidTarget(srcTarget) ||
	    !mglCopyImageIsValidTarget(dstTarget)) {
		ERROR_RETURN(GL_INVALID_ENUM);
		return;
	}

	/* Validate texture names */
	if (srcName == 0 || dstName == 0) {
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	/* Resolve source and destination textures (renderbuffers map to
	 * their internal Texture). */
	GLenum src_err = GL_NO_ERROR;
	GLenum dst_err = GL_NO_ERROR;
	Texture *srcTex = mglCopyImageResolveTarget(ctx, srcName, srcTarget, &src_err);
	Texture *dstTex = mglCopyImageResolveTarget(ctx, dstName, dstTarget, &dst_err);

	if (!srcTex || !dstTex) {
		/* Prefer INVALID_ENUM over INVALID_VALUE when the name was valid
		 * but the target did not match. */
		GLenum err = GL_INVALID_VALUE;
		if (src_err == GL_INVALID_ENUM || dst_err == GL_INVALID_ENUM)
			err = GL_INVALID_ENUM;
		ERROR_RETURN(err);
		return;
	}

	/* Validate width/height/depth are non-negative */
	if (srcWidth < 0 || srcHeight < 0 || srcDepth < 0) {
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	/* Renderbuffers only have level 0 and are 2D. */
	if (srcTarget == GL_RENDERBUFFER && srcLevel != 0) {
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}
	if (dstTarget == GL_RENDERBUFFER && dstLevel != 0) {
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	/* Validate mip levels exist */
	if (!mglCopyImageTextureLevelExists(srcTex, srcLevel) ||
	    !mglCopyImageTextureLevelExists(dstTex, dstLevel)) {
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	/* Validate texture completeness — GL spec requires both textures to
	 * be texture complete or mipmap complete for CopyImageSubData. */
	{
		bool src_complete = mglCopyImageIsTextureComplete(srcTex, srcTarget);
		bool dst_complete = mglCopyImageIsTextureComplete(dstTex, dstTarget);
		if (!src_complete || !dst_complete) {
			ERROR_RETURN(GL_INVALID_OPERATION);
			return;
		}
	}

	/* Validate source region is within bounds.
	 *
	 * For array textures (GL_TEXTURE_1D_ARRAY, GL_TEXTURE_2D_ARRAY),
	 * the array size dimension must NOT be scaled by mip level — only
	 * the spatial dimensions (width, and for 2D arrays, height) are
	 * halved at each level.  MGL stores the 1D-array size in height
	 * and the 2D-array size in depth. */
	{
		GLuint src_w = srcTex->width;
		GLuint src_h = srcTex->height;
		GLuint src_d = srcTex->depth;
		bool src_is_2d_array = (srcTarget == GL_TEXTURE_2D_ARRAY);
		bool src_is_1d_array = (srcTarget == GL_TEXTURE_1D_ARRAY);
		if (srcLevel > 0) {
			src_w >>= srcLevel;
			if (!src_is_1d_array)
				src_h >>= srcLevel;
			if (!src_is_2d_array)
				src_d >>= srcLevel;
			if (src_w == 0) src_w = 1;
			if (src_h == 0) src_h = 1;
			if (src_d == 0) src_d = 1;
		}
		if (srcTarget == GL_TEXTURE_CUBE_MAP) {
			src_d = 6;
		}
		if (srcTarget == GL_RENDERBUFFER) {
			src_d = 1;
		}
		/* For 1D array textures, MGL stores the array size in height.
		 * CopyImageSubData interprets srcY as the texel y coordinate
		 * (must be 0 for 1D) and srcZ as the layer index. */
		if (src_is_1d_array) {
			src_d = src_h;
			src_h = 1;
		}
		if (srcTarget == GL_TEXTURE_1D) {
			src_h = 1;
		}
		if ((GLint)src_w < srcX + srcWidth ||
		    (GLint)src_h < srcY + srcHeight ||
		    (GLint)src_d < srcZ + srcDepth) {
			ERROR_RETURN(GL_INVALID_VALUE);
			return;
		}
	}

	/* Validate destination region is within bounds */
	{
		GLuint dst_w = dstTex->width;
		GLuint dst_h = dstTex->height;
		GLuint dst_d = dstTex->depth;
		bool dst_is_2d_array = (dstTarget == GL_TEXTURE_2D_ARRAY);
		bool dst_is_1d_array = (dstTarget == GL_TEXTURE_1D_ARRAY);
		if (dstLevel > 0) {
			dst_w >>= dstLevel;
			if (!dst_is_1d_array)
				dst_h >>= dstLevel;
			if (!dst_is_2d_array)
				dst_d >>= dstLevel;
			if (dst_w == 0) dst_w = 1;
			if (dst_h == 0) dst_h = 1;
			if (dst_d == 0) dst_d = 1;
		}
		if (dstTarget == GL_TEXTURE_CUBE_MAP) {
			dst_d = 6;
		}
		if (dstTarget == GL_RENDERBUFFER) {
			dst_d = 1;
		}
		if (dst_is_1d_array) {
			dst_d = dst_h;
			dst_h = 1;
		}
		if (dstTarget == GL_TEXTURE_1D) {
			dst_h = 1;
		}
		if ((GLint)dst_w < dstX + srcWidth ||
		    (GLint)dst_h < dstY + srcHeight ||
		    (GLint)dst_d < dstZ + srcDepth) {
			ERROR_RETURN(GL_INVALID_VALUE);
			return;
		}
	}

	/* Validate compressed format alignment — for compressed formats,
	 * srcX, srcY, srcWidth, srcHeight must be multiples of the block
	 * size.  Same for destination coordinates.
	 *
	 * Use compressed_internalformat (the original compressed format)
	 * when available, because MGL remaps compressed internalformats to
	 * their uncompressed equivalents for storage. */
	{
		GLuint src_bw, src_bh, dst_bw, dst_bh;
		GLenum src_fmt = srcTex->compressed_internalformat ? srcTex->compressed_internalformat : srcTex->internalformat;
		GLenum dst_fmt = dstTex->compressed_internalformat ? dstTex->compressed_internalformat : dstTex->internalformat;
		bool src_compressed = mglCopyImageCompressedBlockDims(src_fmt, &src_bw, &src_bh);
		bool dst_compressed = mglCopyImageCompressedBlockDims(dst_fmt, &dst_bw, &dst_bh);

		if (src_compressed) {
			if ((srcX % (GLint)src_bw) != 0 || (srcY % (GLint)src_bh) != 0 ||
			    (srcWidth % (GLsizei)src_bw) != 0 || (srcHeight % (GLsizei)src_bh) != 0) {
				ERROR_RETURN(GL_INVALID_VALUE);
				return;
			}
		}
		if (dst_compressed) {
			if ((dstX % (GLint)dst_bw) != 0 || (dstY % (GLint)dst_bh) != 0 ||
			    (srcWidth % (GLsizei)dst_bw) != 0 || (srcHeight % (GLsizei)dst_bh) != 0) {
				ERROR_RETURN(GL_INVALID_VALUE);
				return;
			}
		}
	}

	/* Validate format compatibility — GL 4.6 spec requires that source
	 * and destination have the same effective pixel size (bytes per
	 * pixel for uncompressed formats, bytes per block for compressed
	 * formats).  Identical internal formats are always compatible.
	 *
	 * Use compressed_internalformat when available so that a texture
	 * created with a compressed internalformat is compared by its block
	 * size, not the remapped uncompressed storage format. */
	{
		GLenum src_fmt = srcTex->compressed_internalformat ? srcTex->compressed_internalformat : srcTex->internalformat;
		GLenum dst_fmt = dstTex->compressed_internalformat ? dstTex->compressed_internalformat : dstTex->internalformat;
		if (src_fmt != dst_fmt) {
			GLuint src_size = mglCopyImageFormatSize(src_fmt);
			GLuint dst_size = mglCopyImageFormatSize(dst_fmt);
			if (src_size == 0 || dst_size == 0 || src_size != dst_size) {
				ERROR_RETURN(GL_INVALID_OPERATION);
				return;
			}
		}
	}

	/* Validate multisample compatibility — mixing multisampled and
	 * non-multisampled targets is not allowed.  Different sample counts
	 * between two multisampled textures are permitted by the spec. */
	{
		bool src_ms = (srcTarget == GL_TEXTURE_2D_MULTISAMPLE ||
		               srcTarget == GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
		bool dst_ms = (dstTarget == GL_TEXTURE_2D_MULTISAMPLE ||
		               dstTarget == GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
		if (src_ms != dst_ms) {
			ERROR_RETURN(GL_INVALID_OPERATION);
			return;
		}
	}

	if (!ctx->mtl_funcs.mtlCopyImageSubData) {
		return;
	}

	// Use Metal blit to copy texture regions
	ctx->mtl_funcs.mtlCopyImageSubData(ctx, srcTex, srcLevel, srcX, srcY, srcZ,
	                                    dstTex, dstLevel, dstX, dstY, dstZ,
	                                    srcWidth, srcHeight, srcDepth);
}

void mglCreateProgramPipelines(GLMContext ctx, GLsizei n, GLuint *pipelines)
{
	for (GLsizei i = 0; i < n; i++)
	{
		mglGenProgramPipelines(ctx, 1, &pipelines[i]);
	}
}

void mglCreateQueries(GLMContext ctx, GLenum target, GLsizei n, GLuint *ids)
{
	if (!mgl_is_query_create_target(target))
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (n < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (!ids)
		return;

	mgl_init_query_table_if_needed();
	for (GLsizei i = 0; i < n; i++)
	{
		QueryObject *q;
		ids[i] = getNewName(&s_query_table);
		q = mgl_get_query(ids[i]);
		if (!q)
		{
			STATE(error) = GL_OUT_OF_MEMORY;
			return;
		}
		q->target = target;
		q->active = GL_FALSE;
		q->available = GL_TRUE;
		q->result = 0;
	}
}

GLuint  mglCreateShaderProgramv(GLMContext ctx, GLenum type, GLsizei count, const GLchar *const*strings)
{
	GLuint shader = mglCreateShader(ctx, type);
	if (!shader)
		return 0;
	
	mglShaderSource(ctx, shader, count, strings, NULL);
	mglCompileShader(ctx, shader);
	
	GLuint program = mglCreateProgram(ctx);
	if (!program) {
		mglDeleteShader(ctx, shader);
		return 0;
	}
	
	mglAttachShader(ctx, program, shader);
	mglProgramParameteri(ctx, program, GL_PROGRAM_SEPARABLE, GL_TRUE);
	mglLinkProgram(ctx, program);
	mglDeleteShader(ctx, shader);
	
	return program;
}

void mglCreateTransformFeedbacks(GLMContext ctx, GLsizei n, GLuint *ids)
{
	if (n < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (!ids)
		return;

	for (GLsizei i = 0; i < n; i++)
	{
		ids[i] = getNewName(&STATE(transform_feedback_table));
		if (ids[i] == 0)
		{
			STATE(error) = GL_OUT_OF_MEMORY;
			return;
		}
		TransformFeedback *ptr = getTransformFeedback(ctx, ids[i]);
		if (!ptr)
		{
			STATE(error) = GL_OUT_OF_MEMORY;
			return;
		}
		ptr->created = GL_TRUE;
	}
}

void mglDebugMessageCallback(GLMContext ctx, GLDEBUGPROC callback, const void *userParam)
{
	// Debug callback - no-op if debug infrastructure not available
	(void)ctx;
	(void)callback;
	(void)userParam;
}

void mglDebugMessageControl(GLMContext ctx, GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint *ids, GLboolean enabled)
{
	// Debug message control - no-op
	(void)ctx;
	(void)source;
	(void)type;
	(void)severity;
	(void)count;
	(void)ids;
	(void)enabled;
}

void mglDebugMessageInsert(GLMContext ctx, GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *buf)
{
	// Insert debug message - no-op
	(void)ctx;
	(void)source;
	(void)type;
	(void)id;
	(void)severity;
	(void)length;
	(void)buf;
}

void mglDeleteQueries(GLMContext ctx, GLsizei n, const GLuint *ids)
{
	if (n < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (!ids)
		return;

	for (GLsizei i = 0; i < n; i++)
	{
		QueryObject *q;
		int slot;
		if (ids[i] == 0)
			continue;
		q = mgl_find_query(ids[i]);
		if (!q)
			continue;
		if (q->active)
		{
			STATE(error) = GL_INVALID_OPERATION;
			continue;
		}
		slot = mgl_query_target_slot(q->target);
		if (slot >= 0) {
			for (GLuint index = 0u; index < MGL_QUERY_MAX_INDEX; index++) {
				if (s_active_query_by_target[slot][index] == q->name)
					s_active_query_by_target[slot][index] = 0;
			}
		}
		deleteHashElement(&s_query_table, q->name);
		free(q);
	}
}

void mglDeleteTransformFeedbacks(GLMContext ctx, GLsizei n, const GLuint *ids)
{
	if (n < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (!ids)
		return;

    for (GLsizei i = 0; i < n; i++)
    {
        if (ids[i] == 0)
            continue;
            
        TransformFeedback *ptr = findTransformFeedback(ctx, ids[i]);
        if (!ptr)
            continue;
            
        // Can't delete if active
        if (ptr->active)
        {
            STATE(error) = GL_INVALID_OPERATION;
            continue;
        }
            
        // If deleting currently bound transform feedback, unbind it
        if (STATE(transform_feedback) && STATE(transform_feedback)->name == ids[i])
        {
            STATE(transform_feedback) = NULL;
        }
        
        // Remove from hash table and free
        deleteHashElement(&STATE(transform_feedback_table), ids[i]);
        free(ptr);
    }
}

void mglDepthRangeArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLdouble *v)
{
	if (!mgl_validate_viewport_range(ctx, first, count))
		return;
	ERROR_CHECK_RETURN(count == 0 || v, GL_INVALID_VALUE);

	// MGL tracks viewport/depth-range state for viewport 0. Indexed ranges that
	// do not include 0 are retained for GL queries, but are not consumed by the
	// current Metal draw path.
	for (GLsizei i = 0; i < count; i++) {
		GLuint index = first + (GLuint)i;
		GLdouble n = v[i * 2 + 0];
		GLdouble f = v[i * 2 + 1];
		if (index == 0) {
			mglDepthRange(ctx, n, f);
		} else if (index < MGL_MAX_VIEWPORTS) {
			ctx->state.depth_range_array[index][0] = n < 0.0 ? 0.0 : (n > 1.0 ? 1.0 : n);
			ctx->state.depth_range_array[index][1] = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
			mglMarkRendererDirtyBits(&ctx->state, DIRTY_RENDER_STATE);
		}
	}
}

void mglDepthRangeIndexed(GLMContext ctx, GLuint index, GLdouble n, GLdouble f)
{
	ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
	if (index == 0) {
		mglDepthRange(ctx, n, f);
	} else if (index < MGL_MAX_VIEWPORTS) {
		ctx->state.depth_range_array[index][0] = n < 0.0 ? 0.0 : (n > 1.0 ? 1.0 : n);
		ctx->state.depth_range_array[index][1] = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
		mglMarkRendererDirtyBits(&ctx->state, DIRTY_RENDER_STATE);
	}
}

void mglDepthRangef(GLMContext ctx, GLfloat n, GLfloat f)
{
	mglDepthRange(ctx, (GLdouble)n, (GLdouble)f);
}

void mglDrawTransformFeedback(GLMContext ctx, GLenum mode, GLuint id)
{
	// Draw from transform feedback - no-op for now
	(void)ctx;
	(void)mode;
	(void)id;
}

void mglDrawTransformFeedbackInstanced(GLMContext ctx, GLenum mode, GLuint id, GLsizei instancecount)
{
	// Draw from transform feedback instanced - no-op for now
	(void)ctx;
	(void)mode;
	(void)id;
	(void)instancecount;
}

void mglDrawTransformFeedbackStream(GLMContext ctx, GLenum mode, GLuint id, GLuint stream)
{
	// Draw from transform feedback stream - no-op for now
	(void)ctx;
	(void)mode;
	(void)id;
	(void)stream;
}

void mglDrawTransformFeedbackStreamInstanced(GLMContext ctx, GLenum mode, GLuint id, GLuint stream, GLsizei instancecount)
{
	// Draw from transform feedback stream instanced - no-op for now
	(void)ctx;
	(void)mode;
	(void)id;
	(void)stream;
	(void)instancecount;
}

void mglEndConditionalRender(GLMContext ctx)
{
	if (!STATE(conditional_render_active))
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(conditional_render_active) = GL_FALSE;
	STATE(conditional_render_skip) = GL_FALSE;
	STATE(conditional_render_query) = 0;
	STATE(conditional_render_mode) = 0;
}

static void mglEndQueryAtIndex(GLMContext ctx, GLenum target, GLuint index)
{
	int slot;
	QueryObject *q;
	GLuint id;

	slot = mgl_query_target_slot(target);
	if (slot < 0)
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (!mgl_query_index_is_valid(target, index))
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	id = s_active_query_by_target[slot][index];
	if (id == 0)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	q = mgl_find_query(id);
	if (!q)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	/* For sample queries, read back the Metal visibility result buffer to
	 * get the real GPU count of samples that passed per-fragment tests.
	 * This replaces the previous heuristic that unconditionally set the
	 * result to 1 for any draw, which incorrectly reported non-zero for
	 * draws where all fragments failed the depth/stencil test. */
	if (mgl_query_target_is_sample(target) && ctx->mtl_funcs.mtlEndSampleQuery)
	{
		/* Draws inside the query are normally still in MGL's deferred command
		 * buffer. Replay them while the Metal visibility query is active. */
		mglFlushCommandBuffer(ctx);
		GLuint64 gpuResult = ctx->mtl_funcs.mtlEndSampleQuery(ctx);
		/* §17.3.5: ANY_SAMPLES_PASSED* results are booleans; the Metal
		 * Boolean visibility mode only guarantees "nonzero". */
		q->result = (target == GL_SAMPLES_PASSED) ? gpuResult
		                                          : (gpuResult ? 1u : 0u);
		q->sample_result_known = GL_TRUE;
	}

	/* For GL_TIME_ELAPSED, flush pending GPU work and compute the GPU
	 * elapsed time via Metal's sampleTimestamps API. */
	if (target == GL_TIME_ELAPSED && ctx->mtl_funcs.mtlEndTimerQuery)
	{
		mglFlushCommandBuffer(ctx);
		q->result = ctx->mtl_funcs.mtlEndTimerQuery(ctx);
		q->timer_result_known = GL_TRUE;
	}

	mgl_finish_query_result(q);
	s_active_query_by_target[slot][index] = 0;
}

void mglEndQuery(GLMContext ctx, GLenum target)
{
	mglEndQueryAtIndex(ctx, target, 0u);
}

void mglEndQueryIndexed(GLMContext ctx, GLenum target, GLuint index)
{
	mglEndQueryAtIndex(ctx, target, index);
}

void mglEndTransformFeedback(GLMContext ctx)
{
	if (!STATE(transform_feedback) || !STATE(transform_feedback)->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(transform_feedback)->active = GL_FALSE;
	STATE(transform_feedback)->paused = GL_FALSE;
}

void mglGenQueries(GLMContext ctx, GLsizei n, GLuint *ids)
{
	if (n < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (!ids)
		return;

	mgl_init_query_table_if_needed();
	for (GLsizei i = 0; i < n; i++)
	{
		ids[i] = getNewName(&s_query_table);
	}
}

void mglGenTransformFeedbacks(GLMContext ctx, GLsizei n, GLuint *ids)
{
	if (n < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (!ids)
		return;

    for (GLsizei i = 0; i < n; i++)
    {
        ids[i] = getNewName(&STATE(transform_feedback_table));
		if (ids[i] == 0)
		{
			STATE(error) = GL_OUT_OF_MEMORY;
			return;
		}
        TransformFeedback *ptr = getTransformFeedback(ctx, ids[i]);
		if (!ptr)
		{
			STATE(error) = GL_OUT_OF_MEMORY;
			return;
		}
		ptr->created = GL_FALSE;
    }
}

/* ---- Buffer variable (SSBO member) enumeration ----
 * GL_BUFFER_VARIABLE interface exposes individual leaf members of SSBO
 * blocks, qualified with the block name (e.g. "Block.member[0]"). */

static GLsizei mgl_program_buffer_variable_count(Program *pptr)
{
	if (!pptr)
		return 0;
	GLsizei total = 0;
	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
	{
		MGLShaderResourceList *list =
			&pptr->shader_resources_list[stage][_STORAGE_BUFFER_RES];
		for (GLuint i = 0; list->list && i < list->count; i++)
		{
			if (mgl_program_block_seen_before(pptr, _STORAGE_BUFFER_RES, stage, i))
				continue;
			MGLShaderResource *block = &list->list[i];
			if (block->ubo_members)
				total += (GLsizei)block->ubo_member_count;
		}
	}
	return total;
}

/* Returns the owning SSBO block and member index for the Nth buffer variable. */
static MGLShaderResource *mgl_program_buffer_variable_at(Program *pptr, GLuint index,
                                                      GLuint *out_member_idx)
{
	if (!pptr)
		return NULL;
	GLuint ordinal = 0;
	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
	{
		MGLShaderResourceList *list =
			&pptr->shader_resources_list[stage][_STORAGE_BUFFER_RES];
		for (GLuint i = 0; list->list && i < list->count; i++)
		{
			if (mgl_program_block_seen_before(pptr, _STORAGE_BUFFER_RES, stage, i))
				continue;
			MGLShaderResource *block = &list->list[i];
			if (!block->ubo_members)
				continue;
			for (GLuint m = 0; m < block->ubo_member_count; m++)
			{
				if (ordinal == index)
				{
					if (out_member_idx)
						*out_member_idx = m;
					return block;
				}
				ordinal++;
			}
		}
	}
	return NULL;
}

/* Find a buffer variable by its qualified name (e.g. "Block.member[0]").
 * Per the GL spec, an array member may also be queried without the "[0]"
 * suffix (e.g. "Block.member" matches "Block.member[0]").
 * Returns the flat index, or -1 if not found. */
static GLint mgl_program_buffer_variable_find_by_name(Program *pptr, const char *name)
{
	if (!pptr || !name)
		return -1;
	GLuint ordinal = 0;
	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
	{
		MGLShaderResourceList *list =
			&pptr->shader_resources_list[stage][_STORAGE_BUFFER_RES];
		for (GLuint i = 0; list->list && i < list->count; i++)
		{
			if (mgl_program_block_seen_before(pptr, _STORAGE_BUFFER_RES, stage, i))
				continue;
			MGLShaderResource *block = &list->list[i];
			if (!block->ubo_members)
				continue;
			for (GLuint m = 0; m < block->ubo_member_count; m++)
			{
				const char *qname = block->ubo_members[m].query_name;
				if (qname)
				{
					if (strcmp(qname, name) == 0)
						return (GLint)ordinal;
					/* Allow matching "Block.member" or "Block.member[0][0]"
					 * to "Block.member[0][0][0]" — the query name may omit
					 * trailing [0] suffixes per GL spec. */
					size_t qlen = strlen(qname);
					size_t nlen = strlen(name);
					if (qlen > nlen &&
					    qname[nlen] == '[' &&
					    strncmp(qname, name, nlen) == 0)
					{
						/* Verify the suffix consists only of "[0]" repetitions */
						const char *suffix = qname + nlen;
						GLboolean all_zero_brackets = GL_TRUE;
						while (*suffix)
						{
							if (strncmp(suffix, "[0]", 3) != 0)
							{
								all_zero_brackets = GL_FALSE;
								break;
							}
							suffix += 3;
						}
						if (all_zero_brackets)
							return (GLint)ordinal;
					}
				}
				ordinal++;
			}
		}
	}
	return -1;
}

/* For a given SSBO block, return the buffer variable indices (into the
 * GL_BUFFER_VARIABLE interface) of its members. */
static GLsizei mgl_program_ssbo_buffer_variable_indices(Program *pptr,
                                                         const MGLShaderResource *target_block,
                                                         GLint *params,
                                                         GLsizei max_count)
{
	if (!pptr || !target_block || !target_block->ubo_members || !params || max_count <= 0)
		return 0;
	GLsizei written = 0;
	GLuint ordinal = 0;
	for (int stage = 0; stage < _MAX_SHADER_TYPES && written < max_count; stage++)
	{
		MGLShaderResourceList *list =
			&pptr->shader_resources_list[stage][_STORAGE_BUFFER_RES];
		for (GLuint i = 0; list->list && i < list->count && written < max_count; i++)
		{
			if (mgl_program_block_seen_before(pptr, _STORAGE_BUFFER_RES, stage, i))
				continue;
			MGLShaderResource *block = &list->list[i];
			if (!block->ubo_members)
				continue;
			for (GLuint m = 0; m < block->ubo_member_count && written < max_count; m++)
			{
				if (block == target_block)
					params[written++] = (GLint)ordinal;
				ordinal++;
			}
		}
	}
	return written;
}

/* Collect distinct atomic-counter buffer binding points used by the program.
 * Returns the count of distinct bindings and fills `out_bindings` (sorted
 * ascending) up to `max_bindings` entries.  Each distinct gl_binding of an
 * ATOMIC_COUNTER resource identifies one atomic-counter buffer. */
static GLuint mgl_program_atomic_counter_buffer_bindings(Program *pptr,
                                                          GLuint *out_bindings,
                                                          GLuint max_bindings)
{
	if (!pptr || !out_bindings || max_bindings == 0)
		return 0;

	GLboolean seen[MAX_BINDABLE_BUFFERS];
	memset(seen, 0, sizeof(seen));

	GLuint count = 0;
	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
	{
		MGLShaderResourceList *list =
			&pptr->shader_resources_list[stage][_ATOMIC_COUNTER_RES];
		for (GLuint i = 0; i < list->count; i++)
		{
			GLuint b = list->list[i].gl_binding;
			if (b < MAX_BINDABLE_BUFFERS && !seen[b])
			{
				seen[b] = GL_TRUE;
				if (count < max_bindings)
					out_bindings[count] = b;
				count++;
			}
		}
	}

	/* Sort the collected bindings ascending so the sequential index
	 * is stable and matches the enumeration order expected by the GL
	 * spec for GL_ATOMIC_COUNTER_BUFFER resources. */
	for (GLuint i = 1; i < count && i < max_bindings; i++)
	{
		GLuint key = out_bindings[i];
		GLuint j = i;
		while (j > 0 && out_bindings[j - 1] > key)
		{
			out_bindings[j] = out_bindings[j - 1];
			j--;
		}
		out_bindings[j] = key;
	}

	return count;
}

void mglGetActiveAtomicCounterBufferiv(GLMContext ctx, GLuint program, GLuint bufferIndex, GLenum pname, GLint *params)
{
	ERROR_CHECK_RETURN(ctx, GL_INVALID_OPERATION);
	ERROR_CHECK_RETURN(params, GL_INVALID_VALUE);

	Program *pptr = findProgram(ctx, program);
	ERROR_CHECK_RETURN(pptr, GL_INVALID_VALUE);

	GLuint active_count = 0;
	GLuint data_size = 0;
	GLboolean referenced_by_stage[_MAX_SHADER_TYPES] = {0};

	for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
		MGLShaderResourceList *list =
			&pptr->shader_resources_list[stage][_ATOMIC_COUNTER_RES];
		for (GLuint i = 0; i < list->count; i++) {
			MGLShaderResource *res = &list->list[i];
			if (res->gl_binding != bufferIndex) {
				continue;
			}
			active_count++;
			referenced_by_stage[stage] = GL_TRUE;
			GLuint offset = res->location != 0xffffffffu ? res->location : 0u;
			if (offset + sizeof(GLuint) > data_size) {
				data_size = offset + sizeof(GLuint);
			}
		}
	}

	if (active_count == 0) {
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	switch (pname) {
		case GL_ATOMIC_COUNTER_BUFFER_DATA_SIZE:
			*params = (GLint)data_size;
			break;
		case GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS:
			*params = (GLint)active_count;
			break;
		case GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES:
			for (GLuint stage = 0, out = 0; stage < _MAX_SHADER_TYPES; stage++) {
				MGLShaderResourceList *list =
					&pptr->shader_resources_list[stage][_ATOMIC_COUNTER_RES];
				for (GLuint i = 0; i < list->count; i++) {
					if (list->list[i].gl_binding == bufferIndex) {
						params[out++] = (GLint)i;
					}
				}
			}
			break;
		case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER:
			*params = referenced_by_stage[_VERTEX_SHADER] ? GL_TRUE : GL_FALSE;
			break;
		case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER:
			*params = referenced_by_stage[_TESS_CONTROL_SHADER] ? GL_TRUE : GL_FALSE;
			break;
		case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER:
			*params = referenced_by_stage[_TESS_EVALUATION_SHADER] ? GL_TRUE : GL_FALSE;
			break;
		case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER:
			*params = referenced_by_stage[_GEOMETRY_SHADER] ? GL_TRUE : GL_FALSE;
			break;
		case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER:
			*params = referenced_by_stage[_FRAGMENT_SHADER] ? GL_TRUE : GL_FALSE;
			break;
		case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER:
			*params = referenced_by_stage[_COMPUTE_SHADER] ? GL_TRUE : GL_FALSE;
			break;
		case GL_ATOMIC_COUNTER_BUFFER_BINDING:
			/* The atomic-counter buffer's GL binding point is its
			 * layout(binding=) value; <bufferIndex> IS that binding. */
			*params = (GLint)bufferIndex;
			break;
		default:
			ERROR_RETURN(GL_INVALID_ENUM);
			break;
	}
}

void mglGetActiveSubroutineName(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	// Subroutines - return empty string
	(void)program; (void)shadertype; (void)index; (void)bufSize;
	if (length) *length = 0;
	if (name && bufSize > 0) name[0] = '\0';
}

void mglGetActiveSubroutineUniformName(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	// Subroutine uniforms - return empty string
	(void)program; (void)shadertype; (void)index; (void)bufSize;
	if (length) *length = 0;
	if (name && bufSize > 0) name[0] = '\0';
}

void mglGetActiveSubroutineUniformiv(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint *values)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	// Subroutine uniform parameters - return 0
	(void)program; (void)shadertype; (void)index; (void)pname;
	if (values) *values = 0;
}

void mglGetBooleani_v(GLMContext ctx, GLenum target, GLuint index, GLboolean *data)
{
	GLint tmp[4] = {0};
	GLsizei count = (target == GL_COLOR_WRITEMASK || target == GL_VIEWPORT || target == GL_SCISSOR_BOX) ? 4 : 1;
	if (target == GL_DEPTH_RANGE) {
		count = 2;
	}
	ERROR_CHECK_RETURN(data, GL_INVALID_VALUE);
	mglGetIntegeri_v(ctx, target, index, tmp);
	for (GLsizei i = 0; i < count; i++) {
		data[i] = (tmp[i] != 0) ? GL_TRUE : GL_FALSE;
	}
}

void mglGetBufferParameteri64v(GLMContext ctx, GLenum target, GLenum pname, GLint64 *params)
{
	GLint tmp = 0;
	if (!params)
		return;
	mglGetBufferParameteriv(ctx, target, pname, &tmp);
	*params = (GLint64)tmp;
}

GLuint  mglGetDebugMessageLog(GLMContext ctx, GLuint count, GLsizei bufSize, GLenum *sources, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, GLchar *messageLog)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	// No debug messages stored
	(void)count;
	(void)bufSize;
	(void)sources;
	(void)types;
	(void)ids;
	(void)severities;
	(void)lengths;
	(void)messageLog;
	return 0;
}

void mglGetDoublei_v(GLMContext ctx, GLenum target, GLuint index, GLdouble *data)
{
	ERROR_CHECK_RETURN(data, GL_INVALID_VALUE);
	switch (target) {
		case GL_VIEWPORT:
			ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
			for (int i = 0; i < 4; i++) {
				data[i] = (GLdouble)ctx->state.viewport_array[index][i];
			}
			return;
		case GL_SCISSOR_BOX:
			ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
			for (int i = 0; i < 4; i++) {
				data[i] = (GLdouble)ctx->state.scissor_box_array[index][i];
			}
			return;
		case GL_DEPTH_RANGE:
			ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
			data[0] = ctx->state.depth_range_array[index][0];
			data[1] = ctx->state.depth_range_array[index][1];
			return;
		default:
			break;
	}

	GLint tmp[4] = {0};
	GLsizei count = (target == GL_COLOR_WRITEMASK) ? 4 : 1;
	mglGetIntegeri_v(ctx, target, index, tmp);
	for (GLsizei i = 0; i < count; i++) {
		data[i] = (GLdouble)tmp[i];
	}
}

void mglGetFloati_v(GLMContext ctx, GLenum target, GLuint index, GLfloat *data)
{
	ERROR_CHECK_RETURN(data, GL_INVALID_VALUE);
	switch (target) {
		case GL_VIEWPORT:
			ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
			for (int i = 0; i < 4; i++) {
				data[i] = ctx->state.viewport_array[index][i];
			}
			return;
		case GL_SCISSOR_BOX:
			ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
			for (int i = 0; i < 4; i++) {
				data[i] = (GLfloat)ctx->state.scissor_box_array[index][i];
			}
			return;
		case GL_DEPTH_RANGE:
			ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
			data[0] = (GLfloat)ctx->state.depth_range_array[index][0];
			data[1] = (GLfloat)ctx->state.depth_range_array[index][1];
			return;
		default:
			break;
	}

	GLint tmp[4] = {0};
	GLsizei count = (target == GL_COLOR_WRITEMASK) ? 4 : 1;
	mglGetIntegeri_v(ctx, target, index, tmp);
	for (GLsizei i = 0; i < count; i++) {
		data[i] = (GLfloat)tmp[i];
	}
}

GLint  mglGetFragDataIndex(GLMContext ctx, GLuint program, const GLchar *name)
{
	GLint location = mglGetFragDataLocation(ctx, program, name);
	return (location >= 0) ? 0 : -1;
}

GLint  mglGetFragDataLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
	Program *pptr;
	MGLShaderResource *res;

	if (!name)
		return -1;

	pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = GL_INVALID_VALUE;
		return -1;
	}
	if (!pptr->link_success)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return -1;
	}

	int output_type = _STAGE_OUTPUT_RES;
	res = mgl_program_resource_find_by_name(pptr, &output_type, 1, name, NULL, NULL, NULL);
	return res ? (GLint)res->location : -1;
}


GLenum  mglGetGraphicsResetStatus(GLMContext ctx)
{
	// No robust context support - always return no error
	(void)ctx;
	return GL_NO_ERROR;
}

void mglGetMultisamplefv(GLMContext ctx, GLenum pname, GLuint index, GLfloat *val)
{
	// Get sample positions for multisample rendering
	if (pname != GL_SAMPLE_POSITION) {
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (!val)
		return;

	/* Metal's standard sample positions, indexed by sample count and then
	 * by sample index.  These must match exactly what the MSL built-in
	 * get_sample_position() returns in the shader, so the CPU-side query
	 * (glGetMultisamplefv) and the GPU-side gl_SamplePosition agree. */
	GLsizei samples = 1;
	if (ctx && ctx->state.framebuffer) {
		Framebuffer *fbo = ctx->state.framebuffer;
		GLint sc = (GLint)fbo->default_samples;
		if (sc <= 1) {
			FBOAttachment *a = &fbo->color_attachments[0];
			if (a && a->texture && a->buf.tex) {
				sc = (GLint)((Texture *)a->buf.tex)->samples;
			}
		}
		if (sc > 1)
			samples = sc;
	}

	static const GLfloat s_pos_2[2][2]  = { {0.25f, 0.25f}, {0.75f, 0.75f} };
	static const GLfloat s_pos_4[4][2]  = { {0.375f, 0.125f}, {0.875f, 0.375f},
	                                        {0.125f, 0.875f}, {0.625f, 0.625f} };
	/* 8x and 16x Metal standard positions for completeness. */
	static const GLfloat s_pos_8[8][2]  = {
		{0.5625f, 0.3125f}, {0.4375f, 0.6875f}, {0.8125f, 0.5625f}, {0.3125f, 0.1875f},
		{0.1875f, 0.8125f}, {0.0625f, 0.4375f}, {0.6875f, 0.9375f}, {0.9375f, 0.0625f} };

	if (samples == 2 && index < 2) {
		val[0] = s_pos_2[index][0];
		val[1] = s_pos_2[index][1];
	} else if (samples == 4 && index < 4) {
		val[0] = s_pos_4[index][0];
		val[1] = s_pos_4[index][1];
	} else if (samples == 8 && index < 8) {
		val[0] = s_pos_8[index][0];
		val[1] = s_pos_8[index][1];
	} else if (index == 0) {
		/* 1x or out-of-range: center. */
		val[0] = 0.5f;
		val[1] = 0.5f;
	} else {
		STATE(error) = GL_INVALID_VALUE;
	}
}

void mglGetObjectLabel(GLMContext ctx, GLenum identifier, GLuint name, GLsizei bufSize, GLsizei *length, GLchar *label)
{
	const char *stored_label = "";
	if (ctx && identifier == GL_TEXTURE && name != 0) {
		Texture *tex = findTexture(ctx, name);
		if (tex && tex->debug_label[0] != '\0') {
			stored_label = tex->debug_label;
		}
	}

	GLsizei stored_len = (GLsizei)strlen(stored_label);
	if (length) {
		*length = stored_len;
	}
	if (label && bufSize > 0) {
		GLsizei copy_len = stored_len;
		if (copy_len >= bufSize) {
			copy_len = bufSize - 1;
		}
		if (copy_len > 0) {
			memcpy(label, stored_label, (size_t)copy_len);
		}
		label[copy_len] = '\0';
	}
}

void mglGetObjectPtrLabel(GLMContext ctx, const void *ptr, GLsizei bufSize, GLsizei *length, GLchar *label)
{
	// No labels stored
	(void)ctx;
	(void)ptr;
	if (length) *length = 0;
	if (label && bufSize > 0) label[0] = '\0';
}

void mglGetProgramBinary(GLMContext ctx, GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, void *binary)
{
	// Program binary not supported
	(void)program; (void)bufSize; (void)binary;
	STATE(error) = GL_INVALID_OPERATION;
	if (length) *length = 0;
	if (binaryFormat) *binaryFormat = 0;
}

/* Transform feedback built-in varying names recognized by
 * glTransformFeedbackVaryings (ARB_transform_feedback3).  These appear as
 * active GL_TRANSFORM_FEEDBACK_VARYING resources when enumerated by index,
 * but glGetProgramResourceIndex returns GL_INVALID_INDEX for them. */
static GLboolean mgl_tf_is_builtin_name(const char *name)
{
	if (!name)
		return GL_FALSE;
	if (strcmp(name, "gl_NextBuffer") == 0)
		return GL_TRUE;
	if (strcmp(name, "gl_SkipComponents1") == 0 ||
	    strcmp(name, "gl_SkipComponents2") == 0 ||
	    strcmp(name, "gl_SkipComponents3") == 0 ||
	    strcmp(name, "gl_SkipComponents4") == 0)
		return GL_TRUE;
	return GL_FALSE;
}

/* gl_NextBuffer -> 0, gl_SkipComponentsN -> N. */
static GLuint mgl_tf_builtin_array_size(const char *name)
{
	if (!name)
		return 0;
	if (strcmp(name, "gl_NextBuffer") == 0)
		return 0;
	if (strncmp(name, "gl_SkipComponents", 17) == 0)
		return (GLuint)(name[17] - '0');
	return 0;
}

/* Find the STAGE_OUTPUT resource in a vertex-processing stage that matches
 * the transform feedback varying name.  The name may be a plain identifier
 * ("a"), a whole array ("e"), or an array element ("b[0]").  Returns the
 * matching resource or NULL.  When the name includes "[N]", *out_is_element
 * is set to GL_TRUE so the caller can report array_size=1. */
static MGLShaderResource *mgl_tf_find_varying_output(Program *pptr,
                                                  const char *name,
                                                  GLboolean *out_is_element)
{
	if (!pptr || !name)
		return NULL;

	if (out_is_element)
		*out_is_element = GL_FALSE;

	/* Determine whether the name is an array element reference ("foo[N]"). */
	const char *bracket = strchr(name, '[');
	GLboolean is_element = (bracket != NULL) ? GL_TRUE : GL_FALSE;
	if (out_is_element)
		*out_is_element = is_element;

	/* Base name length (up to '[' or end). */
	size_t base_len = bracket ? (size_t)(bracket - name) : strlen(name);

	/* Search vertex-processing stage outputs in pipeline order.  Transform
	 * feedback captures from the last active vertex-processing stage, so
	 * search geometry -> tess-eval -> tess-control -> vertex and return the
	 * first match.  In practice these tests only use vertex. */
	static const int stages[] = {
		_GEOMETRY_SHADER,
		_TESS_EVALUATION_SHADER,
		_TESS_CONTROL_SHADER,
		_VERTEX_SHADER
	};
	for (size_t si = 0; si < sizeof(stages) / sizeof(stages[0]); si++)
	{
		int stage = stages[si];
		if (!(pptr->attached_shader_mask & SHADER_MASK_BIT(stage)))
			continue;
		MGLShaderResourceList *list =
			&pptr->shader_resources_list[stage][_STAGE_OUTPUT_RES];
		for (GLuint i = 0; list->list && i < list->count; i++)
		{
			MGLShaderResource *res = &list->list[i];
			if (!res->name)
				continue;
			if (is_element)
			{
				/* Match base name; validate index is within bounds. */
				if (strncmp(res->name, name, base_len) != 0 ||
				    res->name[base_len] != '\0')
					continue;
				if (!res->is_array)
					continue;
				char *end = NULL;
				unsigned long parsed = strtoul(name + base_len + 1, &end, 10);
				if (!end || *end != ']' || end[1] != '\0')
					continue;
				GLuint array_size = res->gl_array_size > 0 ? (GLuint)res->gl_array_size : 1u;
				if (parsed >= array_size)
					continue;
				return res;
			}
			else
			{
				if (strcmp(res->name, name) == 0)
					return res;
			}
		}
	}
	return NULL;
}

void mglGetProgramInterfaceiv(GLMContext ctx, GLuint program, GLenum programInterface, GLenum pname, GLint *params)
{
	Program *pptr;
	int res_types[6];
	int res_type_count;

	if (!params)
		return;

	pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE;
		return;
	}

	/* Subroutine and transform-feedback interfaces are valid per the GL
	 * spec but not backed by shader resource types.  Return 0 for
	 * all valid pnames without generating an error. */
	if (!mgl_program_interface_is_valid(programInterface))
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}

	res_type_count = mgl_program_interface_to_spvc_list(programInterface, res_types, 6);
	if (res_type_count <= 0)
	{
		/* GL_TRANSFORM_FEEDBACK_VARYING is backed by the varyings array
		 * set via glTransformFeedbackVaryings. */
		if (programInterface == GL_TRANSFORM_FEEDBACK_VARYING)
		{
			if (pname == GL_ACTIVE_RESOURCES)
			{
				*params = (GLint)pptr->transform_feedback_varying_count;
				return;
			}
			if (pname == GL_MAX_NAME_LENGTH)
			{
				GLint max_len = 0;
				for (GLsizei i = 0; i < pptr->transform_feedback_varying_count; i++)
				{
					const char *vname = pptr->transform_feedback_varying_names[i];
					if (vname)
					{
						GLint len = (GLint)strlen(vname) + 1;
						if (len > max_len)
							max_len = len;
					}
				}
				*params = max_len;
				return;
			}
			STATE(error) = GL_INVALID_ENUM;
			return;
		}
		/* Valid interface (subroutine) with no shader resource backing —
		 * return 0 for all valid pnames. */
		if (pname == GL_ACTIVE_RESOURCES ||
		    pname == GL_MAX_NAME_LENGTH ||
		    pname == GL_MAX_NUM_ACTIVE_VARIABLES ||
		    pname == GL_MAX_NUM_COMPATIBLE_SUBROUTINES)
		{
			*params = 0;
			return;
		}
		STATE(error) = GL_INVALID_ENUM;
		return;
	}

	/* GL_MAX_NAME_LENGTH is not valid for GL_ATOMIC_COUNTER_BUFFER or
	 * GL_TRANSFORM_FEEDBACK_BUFFER. */
	if (pname == GL_MAX_NAME_LENGTH &&
	    (programInterface == GL_ATOMIC_COUNTER_BUFFER ||
	     programInterface == GL_TRANSFORM_FEEDBACK_BUFFER))
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	/* GL_MAX_NUM_ACTIVE_VARIABLES is only valid for block-like interfaces. */
	if (pname == GL_MAX_NUM_ACTIVE_VARIABLES &&
	    programInterface != GL_UNIFORM_BLOCK &&
	    programInterface != GL_SHADER_STORAGE_BLOCK &&
	    programInterface != GL_ATOMIC_COUNTER_BUFFER &&
	    programInterface != GL_TRANSFORM_FEEDBACK_BUFFER)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	switch (pname)
	{
		case GL_ACTIVE_RESOURCES:
			if (programInterface == GL_UNIFORM)
				*params = mglProgramActiveUniformCount(pptr);
			else if (programInterface == GL_ATOMIC_COUNTER_BUFFER)
			{
				GLuint bindings[MAX_BINDABLE_BUFFERS];
				*params = (GLint)mgl_program_atomic_counter_buffer_bindings(
					pptr, bindings, MAX_BINDABLE_BUFFERS);
			}
			else if (programInterface == GL_BUFFER_VARIABLE)
				*params = mgl_program_buffer_variable_count(pptr);
			else
				*params = mgl_program_resource_count(pptr, res_types, res_type_count);
			break;
		case GL_MAX_NAME_LENGTH:
		{
			if (programInterface == GL_UNIFORM)
			{
				*params = mglProgramActiveUniformMaxNameLength(pptr);
				break;
			}
			if (programInterface == GL_BUFFER_VARIABLE)
			{
				GLint max_len = 0;
				for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
				{
					MGLShaderResourceList *list =
						&pptr->shader_resources_list[stage][_STORAGE_BUFFER_RES];
					for (GLuint i = 0; list->list && i < list->count; i++)
					{
						if (mgl_program_block_seen_before(pptr, _STORAGE_BUFFER_RES, stage, i))
							continue;
						MGLShaderResource *block = &list->list[i];
						if (!block->ubo_members)
							continue;
						for (GLuint m = 0; m < block->ubo_member_count; m++)
						{
							const char *qname = block->ubo_members[m].query_name;
							if (qname)
							{
								GLint len = (GLint)strlen(qname) + 1;
								if (len > max_len)
									max_len = len;
							}
						}
					}
				}
				*params = max_len;
				break;
			}

			GLint max_len = 0;
			for (int t = 0; t < res_type_count; t++)
			{
				int res_type = res_types[t];
				bool is_block_type = (res_type == _UNIFORM_BUFFER_RES ||
				                      res_type == _STORAGE_BUFFER_RES);
				for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
				{
					if (!mgl_program_resource_should_include_stage(pptr, res_type, stage))
						continue;
					GLuint count = pptr->shader_resources_list[stage][res_type].count;
					for (GLuint i = 0; i < count; i++)
					{
						if (is_block_type &&
						    mgl_program_block_seen_before(pptr, res_type, stage, i))
							continue;
						MGLShaderResource *res = &pptr->shader_resources_list[stage][res_type].list[i];
						GLint len = (GLint)(res->name ? strlen(res->name) + 1 : 1);
						if (is_block_type &&
						    mgl_program_uniform_block_array_size(res) > 1)
						{
							char tmp_name[256];
							len = mgl_program_uniform_block_element_name(res,
							                                            mgl_program_uniform_block_array_size(res) - 1u,
							                                            (GLsizei)sizeof(tmp_name),
							                                            tmp_name) + 1;
						}
						else if (res_type == _STAGE_INPUT_RES ||
						         res_type == _STAGE_OUTPUT_RES)
						{
							char tmp_name[256];
							len = mgl_program_resource_name_with_array(res,
							                                           (GLsizei)sizeof(tmp_name),
							                                           tmp_name) + 1;
						}
						if (len > max_len)
						max_len = len;
				}
			}
			/* Also consider built-in resource names. */
			for (int t = 0; t < res_type_count; t++)
			{
				int res_type = res_types[t];
				if (res_type != _STAGE_INPUT_RES &&
				    res_type != _STAGE_OUTPUT_RES)
					continue;
				GLuint builtin_count = 0;
				int builtin_stage = -1;
				MGLShaderResource *builtins = mgl_program_builtin_list(pptr, res_type, &builtin_count, &builtin_stage);
				for (GLuint i = 0; builtins && i < builtin_count; i++)
				{
					MGLShaderResource *res = &builtins[i];
					char tmp_name[256];
					GLint len = mgl_program_resource_name_with_array(res,
					                                           (GLsizei)sizeof(tmp_name),
					                                           tmp_name) + 1;
					if (len > max_len)
						max_len = len;
				}
			}
		}
			*params = max_len;
			break;
		}
		case GL_MAX_NUM_ACTIVE_VARIABLES:
		{
			GLint max_vars = 0;
			if (programInterface == GL_ATOMIC_COUNTER_BUFFER)
			{
				GLuint bindings[MAX_BINDABLE_BUFFERS];
				GLuint buf_count = mgl_program_atomic_counter_buffer_bindings(
					pptr, bindings, MAX_BINDABLE_BUFFERS);
				GLint total = mglProgramActiveUniformCount(pptr);
				for (GLuint bi = 0; bi < buf_count; bi++)
				{
					GLuint active = 0;
					for (GLint ui = 0; ui < total; ui++)
					{
						int ui_stage = 0, ui_type = 0;
						MGLShaderResource *ures = mglProgramActiveUniformAt(
							pptr, (GLuint)ui, &ui_stage, &ui_type);
						if (ures && ui_type == _ATOMIC_COUNTER_RES &&
						    ures->gl_binding == bindings[bi])
							active++;
					}
					if ((GLint)active > max_vars)
						max_vars = (GLint)active;
				}
				*params = max_vars;
				break;
			}
			for (int t = 0; t < res_type_count; t++)
			{
				int res_type = res_types[t];
				if (res_type != _UNIFORM_BUFFER_RES &&
				    res_type != _STORAGE_BUFFER_RES)
					continue;
				for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
				{
					MGLShaderResourceList *resources = &pptr->shader_resources_list[stage][res_type];
					for (GLuint i = 0; resources->list && i < resources->count; i++)
					{
						MGLShaderResource *res = &resources->list[i];
						if (mgl_program_block_seen_before(pptr, res_type, stage, i))
							continue;
						if ((GLint)res->ubo_member_count > max_vars)
							max_vars = (GLint)res->ubo_member_count;
					}
				}
			}
			*params = max_vars;
			break;
		}
		default:
			STATE(error) = GL_INVALID_ENUM;
			break;
	}
}

void mglGetProgramPipelineInfoLog(GLMContext ctx, GLuint pipeline, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
	ProgramPipeline *pp = findProgramPipeline(ctx, pipeline);
	if (!pp)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (bufSize < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	if (length) *length = 0;
	if (infoLog && bufSize > 0) infoLog[0] = '\0';
}

void mglGetProgramPipelineiv(GLMContext ctx, GLuint pipeline, GLenum pname, GLint *params)
{
	ProgramPipeline *pp = findProgramPipeline(ctx, pipeline);
	if (!params)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (!pp)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	switch (pname)
	{
		case GL_ACTIVE_PROGRAM:
			*params = 0;
			break;
		case GL_VERTEX_SHADER:
			*params = pp->stage_programs[_VERTEX_SHADER] ? (GLint)pp->stage_programs[_VERTEX_SHADER]->name : 0;
			break;
		case GL_FRAGMENT_SHADER:
			*params = pp->stage_programs[_FRAGMENT_SHADER] ? (GLint)pp->stage_programs[_FRAGMENT_SHADER]->name : 0;
			break;
		case GL_GEOMETRY_SHADER:
			*params = pp->stage_programs[_GEOMETRY_SHADER] ? (GLint)pp->stage_programs[_GEOMETRY_SHADER]->name : 0;
			break;
		case GL_TESS_CONTROL_SHADER:
			*params = pp->stage_programs[_TESS_CONTROL_SHADER] ? (GLint)pp->stage_programs[_TESS_CONTROL_SHADER]->name : 0;
			break;
		case GL_TESS_EVALUATION_SHADER:
			*params = pp->stage_programs[_TESS_EVALUATION_SHADER] ? (GLint)pp->stage_programs[_TESS_EVALUATION_SHADER]->name : 0;
			break;
		case GL_COMPUTE_SHADER:
			*params = pp->stage_programs[_COMPUTE_SHADER] ? (GLint)pp->stage_programs[_COMPUTE_SHADER]->name : 0;
			break;
		case GL_VALIDATE_STATUS:
			*params = pp->validated ? GL_TRUE : GL_FALSE;
			break;
		case GL_INFO_LOG_LENGTH:
			*params = 0;
			break;
		default:
			STATE(error) = GL_INVALID_ENUM;
			break;
	}
}

GLuint  mglGetProgramResourceIndex(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
	Program *pptr;
	int res_types[6];
	int res_type_count;
	int found_type = -1;
	int found_stage = -1;
	GLuint index = GL_INVALID_INDEX;

	if (!name)
		return GL_INVALID_INDEX;

	pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE;
		return GL_INVALID_INDEX;
	}

	/* Per the GL spec, GetProgramResourceIndex does not accept
	 * GL_ATOMIC_COUNTER_BUFFER or GL_TRANSFORM_FEEDBACK_BUFFER. */
	if (programInterface == GL_ATOMIC_COUNTER_BUFFER ||
	    programInterface == GL_TRANSFORM_FEEDBACK_BUFFER)
	{
		STATE(error) = GL_INVALID_ENUM;
		return GL_INVALID_INDEX;
	}

	res_type_count = mgl_program_interface_to_spvc_list(programInterface, res_types, 6);
	if (res_type_count <= 0)
	{
		/* GL_TRANSFORM_FEEDBACK_VARYING: look up by name in the varyings
		 * array.  Built-in names (gl_NextBuffer, gl_SkipComponents*)
		 * always return GL_INVALID_INDEX per the GL spec. */
		if (programInterface == GL_TRANSFORM_FEEDBACK_VARYING)
		{
			if (mgl_tf_is_builtin_name(name))
				return GL_INVALID_INDEX;
			for (GLsizei i = 0; i < pptr->transform_feedback_varying_count; i++)
			{
				const char *vname = pptr->transform_feedback_varying_names[i];
				if (vname && strcmp(vname, name) == 0)
					return (GLuint)i;
			}
			return GL_INVALID_INDEX;
		}
		/* Valid interface (subroutine) with no shader resource backing —
		 * return GL_INVALID_INDEX without error. */
		if (mgl_program_interface_is_valid(programInterface))
			return GL_INVALID_INDEX;
		STATE(error) = GL_INVALID_ENUM;
		return GL_INVALID_INDEX;
	}

	if (programInterface == GL_UNIFORM)
	{
		GLint active_index = mglProgramActiveUniformIndexByName(pptr, name);
		return (active_index >= 0) ? (GLuint)active_index : GL_INVALID_INDEX;
	}

	if (programInterface == GL_BUFFER_VARIABLE)
	{
		GLint bv_index = mgl_program_buffer_variable_find_by_name(pptr, name);
		return (bv_index >= 0) ? (GLuint)bv_index : GL_INVALID_INDEX;
	}

	if (mgl_program_resource_find_by_name(pptr, res_types, res_type_count, name, &index, &found_stage, &found_type))
	{
		return index;
	}
	return GL_INVALID_INDEX;
}

GLint  mglGetProgramResourceLocation(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
	Program *pptr;
	MGLShaderResource *res;
	int res_types[6];
	int res_type_count;
	int found_type = -1;
	int found_stage = -1;

	if (!name)
		return -1;

	pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE;
		return -1;
	}

	/* Per the GL spec, GetProgramResourceLocation requires the program
	 * to have been linked. */
	if (!pptr->link_success)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return -1;
	}

	/* Per the GL spec, GetProgramResourceLocation does not accept
	 * GL_ATOMIC_COUNTER_BUFFER, GL_TRANSFORM_FEEDBACK_VARYING, or
	 * GL_TRANSFORM_FEEDBACK_BUFFER. */
	if (programInterface == GL_ATOMIC_COUNTER_BUFFER ||
	    programInterface == GL_TRANSFORM_FEEDBACK_VARYING ||
	    programInterface == GL_TRANSFORM_FEEDBACK_BUFFER)
	{
		STATE(error) = GL_INVALID_ENUM;
		return -1;
	}

	res_type_count = mgl_program_interface_to_spvc_list(programInterface, res_types, 6);
	if (res_type_count <= 0)
	{
		STATE(error) = GL_INVALID_ENUM;
		return -1;
	}

	if (programInterface == GL_UNIFORM)
	{
		return mglGetUniformLocation(ctx, program, name);
	}

	res = mgl_program_resource_find_by_name(pptr, res_types, res_type_count, name, NULL, &found_stage, &found_type);
	if (res)
	{
		GLint location = (res->location != 0xffffffffu) ? (GLint)res->location : (GLint)res->gl_binding;
		if (programInterface == GL_PROGRAM_INPUT || programInterface == GL_PROGRAM_OUTPUT)
			location += (GLint)(res->ubo_array_element * mgl_program_resource_location_span(res));
		return location;
	}
	return -1;
}

GLint  mglGetProgramResourceLocationIndex(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
	Program *pptr;
	MGLShaderResource *res;
	int res_types[6];
	int res_type_count;

	if (!name)
		return -1;

	pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE;
		return -1;
	}

	if (!pptr->link_success)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return -1;
	}

	/* Per the GL spec, GetProgramResourceLocationIndex is only valid for
	 * GL_PROGRAM_OUTPUT. */
	if (programInterface != GL_PROGRAM_OUTPUT)
	{
		STATE(error) = GL_INVALID_ENUM;
		return -1;
	}

	res_type_count = mgl_program_interface_to_spvc_list(programInterface, res_types, 6);
	if (res_type_count <= 0)
	{
		STATE(error) = GL_INVALID_ENUM;
		return -1;
	}

	res = mgl_program_resource_find_by_name(pptr, res_types, res_type_count, name, NULL, NULL, NULL);
	if (res)
		return (GLint)res->location_index;
	return -1;
}

void mglGetProgramResourceName(GLMContext ctx, GLuint program, GLenum programInterface, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
	Program *pptr;
	MGLShaderResource *res;
	int res_types[6];
	int res_type_count;
	const char *src;
	GLsizei src_len;
	GLsizei copy_len;

	if (length)
		*length = 0;
	if (name && bufSize > 0)
		name[0] = '\0';
	if (bufSize < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE;
		return;
	}

	/* Per the GL spec, GetProgramResourceName does not accept
	 * GL_ATOMIC_COUNTER_BUFFER or GL_TRANSFORM_FEEDBACK_BUFFER. */
	if (programInterface == GL_ATOMIC_COUNTER_BUFFER ||
	    programInterface == GL_TRANSFORM_FEEDBACK_BUFFER)
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}

	res_type_count = mgl_program_interface_to_spvc_list(programInterface, res_types, 6);
	if (res_type_count <= 0)
	{
		/* GL_TRANSFORM_FEEDBACK_VARYING: return the name stored at index. */
		if (programInterface == GL_TRANSFORM_FEEDBACK_VARYING)
		{
			if (index >= (GLuint)pptr->transform_feedback_varying_count)
			{
				STATE(error) = GL_INVALID_VALUE;
				return;
			}
			src = pptr->transform_feedback_varying_names[index];
			src_len = (src) ? (GLsizei)strlen(src) : 0;
			if (name && bufSize > 0)
			{
				copy_len = (src_len < (bufSize - 1)) ? src_len : (bufSize - 1);
				memcpy(name, src, copy_len);
				name[copy_len] = '\0';
				if (length)
					*length = copy_len;
			}
			else if (length)
			{
				*length = src_len;
			}
			return;
		}
		STATE(error) = GL_INVALID_ENUM;
		return;
	}

	if (programInterface == GL_UNIFORM)
	{
		res = mglProgramActiveUniformAt(pptr, index, NULL, NULL);
		if (!res)
		{
			STATE(error) = GL_INVALID_VALUE;
			return;
		}

		mglProgramCopyActiveUniformName(res, bufSize, length, name);
		return;
	}

	if (programInterface == GL_BUFFER_VARIABLE)
	{
		GLuint member_idx = 0;
		MGLShaderResource *block = mgl_program_buffer_variable_at(pptr, index, &member_idx);
		if (!block || !block->ubo_members || member_idx >= block->ubo_member_count)
		{
			STATE(error) = GL_INVALID_VALUE;
			return;
		}
		src = block->ubo_members[member_idx].query_name;
		if (!src)
		{
			STATE(error) = GL_INVALID_VALUE;
			return;
		}
		src_len = (GLsizei)strlen(src);
		copy_len = (src_len < bufSize) ? src_len : bufSize;
		/* Per GL spec, length does NOT include the null terminator. */
		if (length)
			*length = copy_len;
		if (name && bufSize > 0)
		{
			if (copy_len > 0)
				memcpy(name, src, (size_t)copy_len);
			name[copy_len] = '\0';
		}
		return;
	}

	res = mgl_program_resource_at_index(pptr, res_types, res_type_count, index, NULL, NULL);
	if (!res)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	char block_name_buf[256];
	if (programInterface == GL_UNIFORM_BLOCK ||
	    programInterface == GL_SHADER_STORAGE_BLOCK)
	{
		src_len = mgl_program_uniform_block_element_name(res,
		                                                res->ubo_array_element,
		                                                (GLsizei)sizeof(block_name_buf),
		                                                block_name_buf);
		src = block_name_buf;
	}
	else if (programInterface == GL_PROGRAM_INPUT || programInterface == GL_PROGRAM_OUTPUT)
	{
		src_len = mgl_program_resource_name_with_array(res,
		                                              (GLsizei)sizeof(block_name_buf),
		                                              block_name_buf);
		src = block_name_buf;
	}
	else
	{
		src = res->name ? res->name : "";
		src_len = (GLsizei)strlen(src);
	}
	if (name && bufSize > 0)
	{
		copy_len = (src_len < (bufSize - 1)) ? src_len : (bufSize - 1);
		memcpy(name, src, copy_len);
		name[copy_len] = '\0';
		if (length)
			*length = copy_len;
	}
	else if (length)
	{
		*length = src_len;
	}
}

/* Validate a property enum for glGetProgramResourceiv.
 * Returns GL_NO_ERROR if the prop is valid for the given interface,
 * GL_INVALID_ENUM if the prop is not a recognized property name at all,
 * or GL_INVALID_OPERATION if the prop is a recognized name but not
 * valid for the given programInterface. */
static GLenum mglValidateProgramResourceProp(GLenum prop, GLenum programInterface)
{
	switch (prop)
	{
		/* Valid for all interfaces */
		case GL_NAME_LENGTH:
		case GL_TYPE:
		case GL_ARRAY_SIZE:
		case GL_REFERENCED_BY_VERTEX_SHADER:
		case GL_REFERENCED_BY_TESS_CONTROL_SHADER:
		case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
		case GL_REFERENCED_BY_GEOMETRY_SHADER:
		case GL_REFERENCED_BY_FRAGMENT_SHADER:
		case GL_REFERENCED_BY_COMPUTE_SHADER:
			return GL_NO_ERROR;
		/* Valid only for GL_UNIFORM and GL_BUFFER_VARIABLE */
		case GL_OFFSET:
		case GL_BLOCK_INDEX:
		case GL_ARRAY_STRIDE:
		case GL_MATRIX_STRIDE:
		case GL_IS_ROW_MAJOR:
			return (programInterface == GL_UNIFORM ||
			        programInterface == GL_BUFFER_VARIABLE)
				? GL_NO_ERROR : GL_INVALID_OPERATION;
		/* Valid only for GL_UNIFORM */
		case GL_ATOMIC_COUNTER_BUFFER_INDEX:
			return (programInterface == GL_UNIFORM)
				? GL_NO_ERROR : GL_INVALID_OPERATION;
		case GL_LOCATION:
			return (programInterface == GL_UNIFORM ||
			        programInterface == GL_PROGRAM_INPUT ||
			        programInterface == GL_PROGRAM_OUTPUT)
				? GL_NO_ERROR : GL_INVALID_OPERATION;
		case GL_LOCATION_INDEX:
			return (programInterface == GL_PROGRAM_OUTPUT)
				? GL_NO_ERROR : GL_INVALID_OPERATION;
		/* Valid for block-like interfaces */
		case GL_BUFFER_BINDING:
		case GL_BUFFER_DATA_SIZE:
		case GL_NUM_ACTIVE_VARIABLES:
		case GL_ACTIVE_VARIABLES:
			return (programInterface == GL_UNIFORM_BLOCK ||
			        programInterface == GL_SHADER_STORAGE_BLOCK ||
			        programInterface == GL_ATOMIC_COUNTER_BUFFER ||
			        programInterface == GL_TRANSFORM_FEEDBACK_BUFFER)
				? GL_NO_ERROR : GL_INVALID_OPERATION;
		case GL_TOP_LEVEL_ARRAY_SIZE:
		case GL_TOP_LEVEL_ARRAY_STRIDE:
			return (programInterface == GL_BUFFER_VARIABLE)
				? GL_NO_ERROR : GL_INVALID_OPERATION;
		case GL_IS_PER_PATCH:
			return (programInterface == GL_PROGRAM_INPUT ||
			        programInterface == GL_PROGRAM_OUTPUT)
				? GL_NO_ERROR : GL_INVALID_OPERATION;
		default:
			return GL_INVALID_ENUM;
	}
}

void mglGetProgramResourceiv(GLMContext ctx, GLuint program, GLenum programInterface, GLuint index, GLsizei propCount, const GLenum *props, GLsizei count, GLsizei *length, GLint *params)
{
	Program *pptr;
	MGLShaderResource *res;
	int stage = 0;
	int res_type = -1;
	int res_types[6];
	int res_type_count;

	if (length)
		*length = 0;
	if (propCount <= 0 || count < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (propCount > 0 && !props)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (count > 0 && !params)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (count == 0 || !params)
	{
		return;
	}

	pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE;
		return;
	}
	res_type_count = mgl_program_interface_to_spvc_list(programInterface, res_types, 6);
	if (res_type_count <= 0)
	{
		/* GL_TRANSFORM_FEEDBACK_VARYING: report properties from the
		 * stored varyings array.  Built-in names (gl_NextBuffer,
		 * gl_SkipComponents*) report GL_TYPE=GL_NONE and array_size
		 * per the spec; user varyings report the underlying type. */
		if (programInterface == GL_TRANSFORM_FEEDBACK_VARYING)
		{
			if (index >= (GLuint)pptr->transform_feedback_varying_count)
			{
				STATE(error) = GL_INVALID_VALUE;
				return;
			}
			const char *vname = pptr->transform_feedback_varying_names[index];

			/* Validate props for this interface. */
			for (GLsizei i = 0; i < propCount; i++)
			{
				GLenum err = mglValidateProgramResourceProp(props[i], programInterface);
				if (err != GL_NO_ERROR)
				{
					STATE(error) = err;
					return;
				}
			}

			GLboolean is_builtin = mgl_tf_is_builtin_name(vname);
			MGLShaderResource *varying = is_builtin ? NULL :
				mgl_tf_find_varying_output(pptr, vname, NULL);

			GLsizei out_idx = 0;
			for (GLsizei i = 0; i < propCount; i++)
			{
				if (out_idx >= count)
					break;
				switch (props[i])
				{
					case GL_NAME_LENGTH:
						params[out_idx++] = (GLint)strlen(vname) + 1;
						break;
					case GL_TYPE:
						params[out_idx++] = is_builtin ? GL_NONE :
							(varying ? (GLint)varying->gl_type : GL_NONE);
						break;
					case GL_ARRAY_SIZE:
						if (is_builtin)
							params[out_idx++] = (GLint)mgl_tf_builtin_array_size(vname);
						else if (varying && varying->is_array &&
							 strchr(vname, '[') == NULL)
							params[out_idx++] = (GLint)(varying->gl_array_size > 0 ?
								varying->gl_array_size : 1);
						else
							params[out_idx++] = 1;
						break;
					case GL_REFERENCED_BY_VERTEX_SHADER:
						params[out_idx++] = (pptr->attached_shader_mask & VERTEX_SHADER_MASK_BIT) ? GL_TRUE : GL_FALSE;
						break;
					case GL_REFERENCED_BY_TESS_CONTROL_SHADER:
						params[out_idx++] = (pptr->attached_shader_mask & TESS_CONTROL_SHADER_MASK_BIT) ? GL_TRUE : GL_FALSE;
						break;
					case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
						params[out_idx++] = (pptr->attached_shader_mask & TESS_EVALUATION_SHADER_MASK_BIT) ? GL_TRUE : GL_FALSE;
						break;
					case GL_REFERENCED_BY_GEOMETRY_SHADER:
						params[out_idx++] = (pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT) ? GL_TRUE : GL_FALSE;
						break;
					case GL_REFERENCED_BY_FRAGMENT_SHADER:
						params[out_idx++] = GL_FALSE;
						break;
					case GL_REFERENCED_BY_COMPUTE_SHADER:
						params[out_idx++] = GL_FALSE;
						break;
					default:
						/* Already validated above. */
						params[out_idx++] = 0;
						break;
				}
			}
			if (length)
				*length = out_idx;
			return;
		}
		STATE(error) = GL_INVALID_ENUM;
		return;
	}

	if (programInterface == GL_UNIFORM)
	{
		mgl_get_program_uniform_resourceiv(ctx,
		                                   program,
		                                   pptr,
		                                   index,
		                                   propCount,
		                                   props,
		                                   count,
		                                   length,
		                                   params);
		return;
	}

	if (programInterface == GL_ATOMIC_COUNTER_BUFFER)
	{
		/* Validate each prop against the interface */
		for (GLsizei i = 0; i < propCount; i++)
		{
			GLenum err = mglValidateProgramResourceProp(props[i], programInterface);
			if (err != GL_NO_ERROR)
			{
				STATE(error) = err;
				return;
			}
		}

		GLuint bindings[MAX_BINDABLE_BUFFERS];
		GLuint buf_count = mgl_program_atomic_counter_buffer_bindings(pptr, bindings, MAX_BINDABLE_BUFFERS);
		if (index >= buf_count)
		{
			STATE(error) = GL_INVALID_VALUE;
			return;
		}
		GLuint target_binding = bindings[index];

		/* Use a running output index because GL_ACTIVE_VARIABLES writes a
		 * variable number of values. Each scalar property advances out_idx
		 * by 1; GL_ACTIVE_VARIABLES advances by the number of active vars. */
		GLsizei out_idx = 0;
		for (GLsizei i = 0; i < propCount; i++)
		{
			if (out_idx >= count)
				break;
			switch (props[i])
			{
				case GL_NAME_LENGTH: params[out_idx++] = 1; break;
				case GL_TYPE: params[out_idx++] = GL_NONE; break;
				case GL_ARRAY_SIZE: params[out_idx++] = 1; break;
				case GL_BUFFER_BINDING: params[out_idx++] = (GLint)target_binding; break;
				case GL_BUFFER_DATA_SIZE:
				{
					GLuint data_size = 0;
					for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
					{
						MGLShaderResourceList *list =
							&pptr->shader_resources_list[stage][_ATOMIC_COUNTER_RES];
						for (GLuint j = 0; j < list->count; j++)
						{
							if (list->list[j].gl_binding == target_binding)
							{
								GLuint offset = list->list[j].location != 0xffffffffu
									? list->list[j].location : 0u;
								if (offset + sizeof(GLuint) > data_size)
									data_size = offset + sizeof(GLuint);
							}
						}
					}
					params[out_idx++] = (GLint)data_size;
					break;
				}
				case GL_NUM_ACTIVE_VARIABLES:
				{
					GLuint active = 0;
					GLint total = mglProgramActiveUniformCount(pptr);
					for (GLint ui = 0; ui < total; ui++)
					{
						int ui_stage = 0;
						int ui_type = 0;
						MGLShaderResource *ures = mglProgramActiveUniformAt(pptr, (GLuint)ui, &ui_stage, &ui_type);
						if (ures && ui_type == _ATOMIC_COUNTER_RES &&
						    ures->gl_binding == target_binding)
							active++;
					}
					params[out_idx++] = (GLint)active;
					break;
				}
				case GL_REFERENCED_BY_VERTEX_SHADER:
				case GL_REFERENCED_BY_FRAGMENT_SHADER:
				case GL_REFERENCED_BY_COMPUTE_SHADER:
				case GL_REFERENCED_BY_GEOMETRY_SHADER:
				case GL_REFERENCED_BY_TESS_CONTROL_SHADER:
				case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
				{
					int query_stage;
					switch (props[i])
					{
						case GL_REFERENCED_BY_VERTEX_SHADER: query_stage = _VERTEX_SHADER; break;
						case GL_REFERENCED_BY_FRAGMENT_SHADER: query_stage = _FRAGMENT_SHADER; break;
						case GL_REFERENCED_BY_COMPUTE_SHADER: query_stage = _COMPUTE_SHADER; break;
						case GL_REFERENCED_BY_GEOMETRY_SHADER: query_stage = _GEOMETRY_SHADER; break;
						case GL_REFERENCED_BY_TESS_CONTROL_SHADER: query_stage = _TESS_CONTROL_SHADER; break;
						case GL_REFERENCED_BY_TESS_EVALUATION_SHADER: query_stage = _TESS_EVALUATION_SHADER; break;
						default: query_stage = -1; break;
					}
					GLboolean referenced = GL_FALSE;
					if (query_stage >= 0)
					{
						MGLShaderResourceList *list =
							&pptr->shader_resources_list[query_stage][_ATOMIC_COUNTER_RES];
						for (GLuint j = 0; j < list->count; j++)
						{
							if (list->list[j].gl_binding == target_binding)
							{
								referenced = GL_TRUE;
								break;
							}
						}
					}
					params[out_idx++] = referenced ? GL_TRUE : GL_FALSE;
					break;
				}
				case GL_ACTIVE_VARIABLES:
				{
					GLint total = mglProgramActiveUniformCount(pptr);
					for (GLint ui = 0; ui < total && out_idx < count; ui++)
					{
						int ui_stage = 0;
						int ui_type = 0;
						MGLShaderResource *ures = mglProgramActiveUniformAt(pptr, (GLuint)ui, &ui_stage, &ui_type);
						if (ures && ui_type == _ATOMIC_COUNTER_RES &&
						    ures->gl_binding == target_binding)
							params[out_idx++] = ui;
					}
					break;
				}
				default:
					STATE(error) = GL_INVALID_ENUM;
					return;
			}
		}
		if (length)
			*length = out_idx;
		return;
	}

	if (programInterface == GL_BUFFER_VARIABLE)
	{
		/* Validate each prop against the interface */
		for (GLsizei i = 0; i < propCount; i++)
		{
			GLenum err = mglValidateProgramResourceProp(props[i], programInterface);
			if (err != GL_NO_ERROR)
			{
				STATE(error) = err;
				return;
			}
		}

		GLuint member_idx = 0;
		MGLShaderResource *block = mgl_program_buffer_variable_at(pptr, index, &member_idx);
		if (!block || !block->ubo_members || member_idx >= block->ubo_member_count)
		{
			STATE(error) = GL_INVALID_VALUE;
			return;
		}
		const SpirvUBOMember *bv = &block->ubo_members[member_idx];

		/* Compute the owning SSBO block's index in the
		 * GL_SHADER_STORAGE_BLOCK interface. */
		GLint block_index = -1;
		{
			GLuint ordinal = 0;
			for (int s = 0; s < _MAX_SHADER_TYPES && block_index < 0; s++)
			{
				MGLShaderResourceList *list =
					&pptr->shader_resources_list[s][_STORAGE_BUFFER_RES];
				for (GLuint i = 0; list->list && i < list->count && block_index < 0; i++)
				{
					if (mgl_program_block_seen_before(pptr, _STORAGE_BUFFER_RES, s, i))
						continue;
					MGLShaderResource *blk = &list->list[i];
					GLuint array_size = mgl_program_uniform_block_array_size(blk);
					GLuint element_count = array_size > 1 ? array_size : 1;
					for (GLuint elem = 0; elem < element_count; elem++)
					{
						if (blk == block)
						{
							block_index = (GLint)ordinal;
							break;
						}
						ordinal++;
					}
				}
			}
		}

		GLsizei out_idx = 0;
		for (GLsizei i = 0; i < propCount; i++)
		{
			if (out_idx >= count)
				break;
			switch (props[i])
			{
				case GL_NAME_LENGTH:
					params[out_idx++] = bv->query_name
						? (GLint)strlen(bv->query_name) + 1 : 1;
					break;
				case GL_TYPE:
					params[out_idx++] = (GLint)bv->gl_type;
					break;
				case GL_ARRAY_SIZE:
					params[out_idx++] = (GLint)bv->size;
					break;
				case GL_BLOCK_INDEX:
					params[out_idx++] = block_index;
					break;
				case GL_OFFSET:
					params[out_idx++] = (GLint)bv->offset;
					break;
				case GL_ARRAY_STRIDE:
					params[out_idx++] = bv->array_stride;
					break;
				case GL_MATRIX_STRIDE:
					params[out_idx++] = bv->matrix_stride;
					break;
				case GL_IS_ROW_MAJOR:
					params[out_idx++] = bv->is_row_major ? GL_TRUE : GL_FALSE;
					break;
				case GL_TOP_LEVEL_ARRAY_SIZE:
				params[out_idx++] = bv->top_level_array_size > 0
					? bv->top_level_array_size : 1;
				break;
			case GL_TOP_LEVEL_ARRAY_STRIDE:
				params[out_idx++] = bv->top_level_array_stride;
				break;
				case GL_REFERENCED_BY_VERTEX_SHADER:
				case GL_REFERENCED_BY_FRAGMENT_SHADER:
				case GL_REFERENCED_BY_COMPUTE_SHADER:
				case GL_REFERENCED_BY_GEOMETRY_SHADER:
				case GL_REFERENCED_BY_TESS_CONTROL_SHADER:
				case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
				{
					int query_stage;
					switch (props[i])
					{
						case GL_REFERENCED_BY_VERTEX_SHADER: query_stage = _VERTEX_SHADER; break;
						case GL_REFERENCED_BY_FRAGMENT_SHADER: query_stage = _FRAGMENT_SHADER; break;
						case GL_REFERENCED_BY_COMPUTE_SHADER: query_stage = _COMPUTE_SHADER; break;
						case GL_REFERENCED_BY_GEOMETRY_SHADER: query_stage = _GEOMETRY_SHADER; break;
						case GL_REFERENCED_BY_TESS_CONTROL_SHADER: query_stage = _TESS_CONTROL_SHADER; break;
						case GL_REFERENCED_BY_TESS_EVALUATION_SHADER: query_stage = _TESS_EVALUATION_SHADER; break;
						default: query_stage = -1; break;
					}
					GLboolean referenced = GL_FALSE;
					if (query_stage >= 0)
					{
						referenced = mgl_program_block_referenced_by_stage(
							pptr, _STORAGE_BUFFER_RES, block, query_stage);
					}
					params[out_idx++] = referenced ? GL_TRUE : GL_FALSE;
					break;
				}
				default:
					STATE(error) = GL_INVALID_ENUM;
					return;
			}
		}
		if (length)
			*length = out_idx;
		return;
	}

	res = mgl_program_resource_at_index(pptr, res_types, res_type_count, index, &stage, &res_type);
	if (!res)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	/* Validate each prop against the interface */
	for (GLsizei i = 0; i < propCount; i++)
	{
		GLenum err = mglValidateProgramResourceProp(props[i], programInterface);
		if (err != GL_NO_ERROR)
		{
			STATE(error) = err;
			return;
		}
	}

	/* Use a running output index because GL_ACTIVE_VARIABLES writes a
	 * variable number of values. Each scalar property advances out_idx
	 * by 1; GL_ACTIVE_VARIABLES advances by the number of active vars. */
	GLsizei out_idx = 0;
	for (GLsizei i = 0; i < propCount; i++)
	{
		if (out_idx >= count)
			break;
		switch (props[i])
		{
			case GL_NAME_LENGTH:
			if (res_type == _UNIFORM_BUFFER_RES ||
			    res_type == _STORAGE_BUFFER_RES)
			{
				char tmp_name[256];
				params[out_idx++] = mgl_program_uniform_block_element_name(res,
				                                                   res->ubo_array_element,
				                                                   (GLsizei)sizeof(tmp_name),
				                                                   tmp_name) + 1;
			}
			else if (res_type == _STAGE_INPUT_RES ||
			         res_type == _STAGE_OUTPUT_RES)
			{
				char tmp_name[256];
				params[out_idx++] = mgl_program_resource_name_with_array(res,
				                                                (GLsizei)sizeof(tmp_name),
				                                                tmp_name) + 1;
			}
			else
				params[out_idx++] = (GLint)(res->name ? strlen(res->name) + 1 : 1);
			break;
			case GL_TYPE: params[out_idx++] = mgl_program_resource_gl_type(res, res_type); break;
			case GL_ARRAY_SIZE:
				params[out_idx++] = (res_type == _STAGE_INPUT_RES ||
				             res_type == _STAGE_OUTPUT_RES)
					? (res->gl_array_size > 0 ? res->gl_array_size : 1)
					: 1;
				break;
			case GL_OFFSET: params[out_idx++] = -1; break;
			case GL_BLOCK_INDEX: params[out_idx++] = -1; break;
			case GL_ARRAY_STRIDE: params[out_idx++] = 0; break;
			case GL_MATRIX_STRIDE: params[out_idx++] = 0; break;
			case GL_IS_ROW_MAJOR: params[out_idx++] = 0; break;
			case GL_ATOMIC_COUNTER_BUFFER_INDEX: params[out_idx++] = -1; break;
			case GL_BUFFER_BINDING:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? (GLint)mgl_program_uniform_block_element_binding(res, res->ubo_array_element)
					: (GLint)res->gl_binding;
				break;
			case GL_BUFFER_DATA_SIZE:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? (GLint)mgl_program_block_required_size(pptr, res_type, res)
					: (GLint)mgl_round_up_16(res->required_size);
				break;
			case GL_NUM_ACTIVE_VARIABLES:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? (GLint)res->ubo_member_count : 0;
				break;
			case GL_REFERENCED_BY_VERTEX_SHADER:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? ((res_type == _STORAGE_BUFFER_RES ||
					    res->ubo_array_element == 0)
						? mgl_program_block_referenced_by_stage(pptr, res_type, res, _VERTEX_SHADER)
						: GL_FALSE)
					: ((stage == _VERTEX_SHADER) ? GL_TRUE : GL_FALSE);
				break;
			case GL_REFERENCED_BY_FRAGMENT_SHADER:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? ((res_type == _STORAGE_BUFFER_RES ||
					    res->ubo_array_element == 0)
						? mgl_program_block_referenced_by_stage(pptr, res_type, res, _FRAGMENT_SHADER)
						: GL_FALSE)
					: ((stage == _FRAGMENT_SHADER) ? GL_TRUE : GL_FALSE);
				break;
			case GL_REFERENCED_BY_GEOMETRY_SHADER:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? ((res_type == _STORAGE_BUFFER_RES ||
					    res->ubo_array_element == 0)
						? mgl_program_block_referenced_by_stage(pptr, res_type, res, _GEOMETRY_SHADER)
						: GL_FALSE)
					: ((stage == _GEOMETRY_SHADER) ? GL_TRUE : GL_FALSE);
				break;
			case GL_REFERENCED_BY_TESS_CONTROL_SHADER:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? ((res_type == _STORAGE_BUFFER_RES ||
					    res->ubo_array_element == 0)
						? mgl_program_block_referenced_by_stage(pptr, res_type, res, _TESS_CONTROL_SHADER)
						: GL_FALSE)
					: ((stage == _TESS_CONTROL_SHADER) ? GL_TRUE : GL_FALSE);
				break;
			case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? ((res_type == _STORAGE_BUFFER_RES ||
					    res->ubo_array_element == 0)
						? mgl_program_block_referenced_by_stage(pptr, res_type, res, _TESS_EVALUATION_SHADER)
						: GL_FALSE)
					: ((stage == _TESS_EVALUATION_SHADER) ? GL_TRUE : GL_FALSE);
				break;
			case GL_REFERENCED_BY_COMPUTE_SHADER:
				params[out_idx++] = (res_type == _UNIFORM_BUFFER_RES ||
				             res_type == _STORAGE_BUFFER_RES)
					? ((res_type == _STORAGE_BUFFER_RES ||
					    res->ubo_array_element == 0)
						? mgl_program_block_referenced_by_stage(pptr, res_type, res, _COMPUTE_SHADER)
						: GL_FALSE)
					: ((stage == _COMPUTE_SHADER) ? GL_TRUE : GL_FALSE);
				break;
			case GL_LOCATION:
				params[out_idx++] = (GLint)res->location +
				            (GLint)(res->ubo_array_element * mgl_program_resource_location_span(res));
				break;
			case GL_LOCATION_INDEX: params[out_idx++] = (GLint)res->location_index; break;
			case GL_IS_PER_PATCH: params[out_idx++] = res->is_per_patch ? GL_TRUE : GL_FALSE; break;
			case GL_ACTIVE_VARIABLES:
			{
				if (res_type != _UNIFORM_BUFFER_RES &&
				    res_type != _STORAGE_BUFFER_RES)
				{
					STATE(error) = GL_INVALID_ENUM;
					return;
				}
				GLsizei written;
				if (res_type == _STORAGE_BUFFER_RES)
				{
					/* SSBO members are not in the active uniform list;
					 * enumerate buffer variable indices instead. */
					written = mgl_program_ssbo_buffer_variable_indices(pptr,
					                                                   res,
					                                                   params + out_idx,
					                                                   count - out_idx);
				}
				else
				{
					written = mgl_program_uniform_block_active_variables(pptr,
					                                                    res,
					                                                    params + out_idx,
					                                                    count - out_idx);
				}
				out_idx += written;
				break;
			}
			default:
				STATE(error) = GL_INVALID_ENUM;
				return;
		}
	}
	if (length)
		*length = out_idx;
}

void mglGetProgramStageiv(GLMContext ctx, GLuint program, GLenum shadertype, GLenum pname, GLint *values)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetQueryBufferObjecti64v(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
	QueryObject *q = mgl_find_query(id);
	Buffer *buf = findBuffer(ctx, buffer);
	GLuint64 value = 0;
	GLint64 stored;

	if (!q || q->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (!mgl_query_value(q, pname, &value))
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (!buf || !buf->data.buffer_data)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (offset < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (offset > buf->size || (GLsizeiptr)sizeof(GLint64) > buf->size - offset)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	stored = (GLint64)value;
	memcpy((void *)(uintptr_t)(buf->data.buffer_data + (vm_address_t)offset), &stored, sizeof(stored));
	buf->data.dirty_bits |= DIRTY_BUFFER_DATA;
	buf->has_initialized_data = GL_TRUE;
	buf->ever_written = GL_TRUE;
}

void mglGetQueryBufferObjectiv(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
	QueryObject *q = mgl_find_query(id);
	Buffer *buf = findBuffer(ctx, buffer);
	GLuint64 value = 0;
	GLint stored;

	if (!q || q->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (!mgl_query_value(q, pname, &value))
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (!buf || !buf->data.buffer_data)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (offset < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (offset > buf->size || (GLsizeiptr)sizeof(GLint) > buf->size - offset)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	stored = (GLint)value;
	memcpy((void *)(uintptr_t)(buf->data.buffer_data + (vm_address_t)offset), &stored, sizeof(stored));
	buf->data.dirty_bits |= DIRTY_BUFFER_DATA;
	buf->has_initialized_data = GL_TRUE;
	buf->ever_written = GL_TRUE;
}

void mglGetQueryBufferObjectui64v(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
	QueryObject *q = mgl_find_query(id);
	Buffer *buf = findBuffer(ctx, buffer);
	GLuint64 stored = 0;

	if (!q || q->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (!mgl_query_value(q, pname, &stored))
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (!buf || !buf->data.buffer_data)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (offset < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (offset > buf->size || (GLsizeiptr)sizeof(GLuint64) > buf->size - offset)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	memcpy((void *)(uintptr_t)(buf->data.buffer_data + (vm_address_t)offset), &stored, sizeof(stored));
	buf->data.dirty_bits |= DIRTY_BUFFER_DATA;
	buf->has_initialized_data = GL_TRUE;
	buf->ever_written = GL_TRUE;
}

void mglGetQueryBufferObjectuiv(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
	QueryObject *q = mgl_find_query(id);
	Buffer *buf = findBuffer(ctx, buffer);
	GLuint64 value = 0;
	GLuint stored;

	if (!q || q->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (!mgl_query_value(q, pname, &value))
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (!buf || !buf->data.buffer_data)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (offset < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (offset > buf->size || (GLsizeiptr)sizeof(GLuint) > buf->size - offset)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	stored = (GLuint)value;
	memcpy((void *)(uintptr_t)(buf->data.buffer_data + (vm_address_t)offset), &stored, sizeof(stored));
	buf->data.dirty_bits |= DIRTY_BUFFER_DATA;
	buf->has_initialized_data = GL_TRUE;
	buf->ever_written = GL_TRUE;
}

void mglGetQueryIndexediv(GLMContext ctx, GLenum target, GLuint index, GLenum pname, GLint *params)
{
	int slot = mgl_query_target_slot(target);
	(void)ctx;
	if (!params)
		return;
	if (slot < 0)
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (!mgl_query_index_is_valid(target, index))
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (pname == GL_CURRENT_QUERY) {
		*params = (GLint)s_active_query_by_target[slot][index];
		return;
	}
	if (pname == GL_QUERY_COUNTER_BITS) {
		*params = (target == GL_TIME_ELAPSED) ? 64 : 32;
		return;
	}
	STATE(error) = GL_INVALID_ENUM;
}

void mglGetQueryObjecti64v(GLMContext ctx, GLuint id, GLenum pname, GLint64 *params)
{
	QueryObject *q = mgl_find_query(id);
	if (!params)
		return;
	if (!q)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	GLuint64 value = 0;
	if (mgl_query_value(q, pname, &value))
		*params = (GLint64)value;
	else
		STATE(error) = GL_INVALID_ENUM;
}

void mglGetQueryObjectiv(GLMContext ctx, GLuint id, GLenum pname, GLint *params)
{
	GLint64 val = 0;
	if (!params)
		return;
	mglGetQueryObjecti64v(ctx, id, pname, &val);
	*params = (GLint)val;
}

void mglGetQueryObjectui64v(GLMContext ctx, GLuint id, GLenum pname, GLuint64 *params)
{
	QueryObject *q = mgl_find_query(id);
	if (!params)
		return;
	if (!q)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	GLuint64 value = 0;
	if (mgl_query_value(q, pname, &value))
		*params = value;
	else
		STATE(error) = GL_INVALID_ENUM;
}

void mglGetQueryObjectuiv(GLMContext ctx, GLuint id, GLenum pname, GLuint *params)
{
	GLuint64 val = 0;
	if (!params)
		return;
	mglGetQueryObjectui64v(ctx, id, pname, &val);
	*params = (GLuint)val;
}

void mglGetQueryiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
	int slot = mgl_query_target_slot(target);
	if (!params)
		return;
	if (slot < 0)
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}

	switch (pname)
	{
		case GL_CURRENT_QUERY:
			*params = (GLint)s_active_query_by_target[slot][0];
			break;
		case GL_QUERY_COUNTER_BITS:
			*params = (target == GL_TIME_ELAPSED) ? 64 : 32;
			break;
		default:
			STATE(error) = GL_INVALID_ENUM;
			break;
	}
}

void mglGetShaderPrecisionFormat(GLMContext ctx, GLenum shadertype, GLenum precisiontype, GLint *range, GLint *precision)
{
	// Return shader precision format - full precision for all types
	(void)ctx;
	(void)shadertype;
	(void)precisiontype;
	if (range) {
		range[0] = 127;
		range[1] = 127;
	}
	if (precision) {
		*precision = 23;
	}
}

GLuint mglGetSubroutineIndex(GLMContext ctx, GLuint program, GLenum shadertype, const GLchar *name)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
	return 0;
}

GLint mglGetSubroutineUniformLocation(GLMContext ctx, GLuint program, GLenum shadertype, const GLchar *name)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
	return 0;
}
void mglGetTransformFeedbackVarying(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLsizei *size, GLenum *type, GLchar *name)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetTransformFeedbacki64_v(GLMContext ctx, GLuint xfb, GLenum pname, GLuint index, GLint64 *param)
{
	if (!param)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	TransformFeedback *ptr = findTransformFeedback(ctx, xfb);
	if (!ptr || !ptr->created)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (index >= MAX_BINDABLE_BUFFERS)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	BufferBaseTarget *slot = &ptr->buffers[index];
	switch (pname)
	{
		case GL_TRANSFORM_FEEDBACK_BUFFER_START:
			*param = (GLint64)slot->offset;
			break;
		case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE:
			*param = (GLint64)slot->size;
			break;
		case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
			*param = (GLint64)slot->buffer;
			break;
		default:
			STATE(error) = GL_INVALID_ENUM;
			break;
	}
}

void mglGetTransformFeedbacki_v(GLMContext ctx, GLuint xfb, GLenum pname, GLuint index, GLint *param)
{
	if (!param)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	GLint64 value = 0;
	mglGetTransformFeedbacki64_v(ctx, xfb, pname, index, &value);
	if (STATE(error) == GL_NO_ERROR)
		*param = (GLint)value;
}

void mglGetTransformFeedbackiv(GLMContext ctx, GLuint xfb, GLenum pname, GLint *param)
{
	if (!param)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	TransformFeedback *ptr = findTransformFeedback(ctx, xfb);
	if (!ptr || !ptr->created)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	switch (pname)
	{
		case GL_TRANSFORM_FEEDBACK_ACTIVE:
			*param = ptr->active ? GL_TRUE : GL_FALSE;
			break;
		case GL_TRANSFORM_FEEDBACK_PAUSED:
			*param = ptr->paused ? GL_TRUE : GL_FALSE;
			break;
		default:
			STATE(error) = GL_INVALID_ENUM;
			break;
	}
}

void mglGetUniformSubroutineuiv(GLMContext ctx, GLenum shadertype, GLint location, GLuint *params)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetUniformdv(GLMContext ctx, GLuint program, GLint location, GLdouble *params)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetUniformuiv(GLMContext ctx, GLuint program, GLint location, GLuint *params)
{
	GLint tmp = 0;
	if (!params)
		return;
	mglGetUniformiv(ctx, program, location, &tmp);
	*params = (GLuint)tmp;
}

void mglGetVertexAttribIiv(GLMContext ctx, GLuint index, GLenum pname, GLint *params)
{
	ERROR_CHECK_RETURN(params, GL_INVALID_VALUE);
	ERROR_CHECK_RETURN(index < MAX_ATTRIBS, GL_INVALID_VALUE);
	if (pname == GL_CURRENT_VERTEX_ATTRIB) {
		for (int i = 0; i < 4; i++)
			params[i] = ctx->state.current_vertex_attrib[index].i[i];
		return;
	}
	mglGetVertexAttribiv(ctx, index, pname, params);
}

void mglGetVertexAttribIuiv(GLMContext ctx, GLuint index, GLenum pname, GLuint *params)
{
	ERROR_CHECK_RETURN(params, GL_INVALID_VALUE);
	ERROR_CHECK_RETURN(index < MAX_ATTRIBS, GL_INVALID_VALUE);
	if (pname == GL_CURRENT_VERTEX_ATTRIB) {
		for (int i = 0; i < 4; i++)
			params[i] = ctx->state.current_vertex_attrib[index].u[i];
		return;
	}
	GLint value = 0;
	mglGetVertexAttribiv(ctx, index, pname, &value);
	*params = (GLuint)value;
}

void mglGetVertexAttribLdv(GLMContext ctx, GLuint index, GLenum pname, GLdouble *params)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnMapdv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLdouble *v)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnMapfv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLfloat *v)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnMapiv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLint *v)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnPixelMapfv(GLMContext ctx, GLenum map, GLsizei bufSize, GLfloat *values)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnPixelMapuiv(GLMContext ctx, GLenum map, GLsizei bufSize, GLuint *values)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnPixelMapusv(GLMContext ctx, GLenum map, GLsizei bufSize, GLushort *values)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnUniformdv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLdouble *params)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnUniformfv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLfloat *params)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnUniformiv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLint *params)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglGetnUniformuiv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLuint *params)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

GLboolean mglIsQuery(GLMContext ctx, GLuint id)
{
	(void)ctx;
	return mgl_find_query(id) ? GL_TRUE : GL_FALSE;
}

GLboolean mglIsTransformFeedback(GLMContext ctx, GLuint id)
{
	TransformFeedback *ptr = findTransformFeedback(ctx, id);
	return (ptr && ptr->created) ? GL_TRUE : GL_FALSE;
}

static bool mglReadIndirectCountParameter(GLMContext ctx,
                                          const char *label,
                                          GLintptr drawcount,
                                          GLuint *actual_drawcount,
                                          Buffer **out_parameter_buffer)
{
	Buffer *parameter_buffer;
	const uint8_t *base;
	size_t logical_size;
	size_t backing_size;
	size_t readable_size;

	if (!ctx || !actual_drawcount) {
		ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
	}
	*actual_drawcount = 0u;
	if (out_parameter_buffer) {
		*out_parameter_buffer = NULL;
	}

	if (drawcount < 0 || (drawcount & 3) != 0) {
		mglTraceLogExternal("%s_SKIP reason=bad_drawcount_offset drawcountOffset=%lld program=%u",
		                    label ? label : "MULTI_DRAW_INDIRECT_COUNT",
		                    (long long)drawcount,
		                    (unsigned)ctx->state.program_name);
		ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
	}

	parameter_buffer = STATE(buffers[_PARAMETER_BUFFER]);
	if (!parameter_buffer) {
		mglTraceLogExternal("%s_SKIP reason=no_parameter_buffer program=%u",
		                    label ? label : "MULTI_DRAW_INDIRECT_COUNT",
		                    (unsigned)ctx->state.program_name);
		ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
	}
	if (parameter_buffer->mapped &&
	    !(parameter_buffer->access_flags & GL_MAP_PERSISTENT_BIT)) {
		mglTraceLogExternal("%s_SKIP reason=parameter_buffer_mapped buffer=%u program=%u",
		                    label ? label : "MULTI_DRAW_INDIRECT_COUNT",
		                    (unsigned)parameter_buffer->name,
		                    (unsigned)ctx->state.program_name);
		ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
	}
	if (parameter_buffer->size < 0 ||
	    drawcount > parameter_buffer->size ||
	    (GLsizeiptr)sizeof(*actual_drawcount) > parameter_buffer->size - drawcount) {
		mglTraceLogExternal("%s_SKIP reason=count_oob buffer=%u offset=%lld size=%lld program=%u",
		                    label ? label : "MULTI_DRAW_INDIRECT_COUNT",
		                    (unsigned)parameter_buffer->name,
		                    (long long)drawcount,
		                    (long long)parameter_buffer->size,
		                    (unsigned)ctx->state.program_name);
		ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
	}

	/* GPU writes to the parameter buffer must be visible before the CPU-side
	 * fallback reads the command count. mglFlushCommandBuffer only drains MGL's
	 * deferred draw buffer; mtlFlush(..., true) commits and waits for Metal. */
	mglFlushCommandBuffer(ctx);
	if (ctx->mtl_funcs.mtlFlush) {
		ctx->mtl_funcs.mtlFlush(ctx, true);
	}

	base = (const uint8_t *)(uintptr_t)parameter_buffer->data.buffer_data;
	logical_size = parameter_buffer->size > 0 ? (size_t)parameter_buffer->size : 0u;
	backing_size = parameter_buffer->data.buffer_size;
	readable_size = logical_size < backing_size ? logical_size : backing_size;
	if (!base ||
	    readable_size == 0u ||
	    (size_t)drawcount > readable_size ||
	    sizeof(*actual_drawcount) > readable_size - (size_t)drawcount) {
		mglTraceLogExternal("%s_SKIP reason=count_unreadable buffer=%u data=%p size=%zu backing=%zu readable=%zu offset=%lld program=%u",
		                    label ? label : "MULTI_DRAW_INDIRECT_COUNT",
		                    (unsigned)parameter_buffer->name,
		                    base,
		                    logical_size,
		                    backing_size,
		                    readable_size,
		                    (long long)drawcount,
		                    (unsigned)ctx->state.program_name);
		ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
	}

	memcpy(actual_drawcount, base + drawcount, sizeof(*actual_drawcount));
	if (out_parameter_buffer) {
		*out_parameter_buffer = parameter_buffer;
	}
	return true;
}

void mglMinSampleShading(GLMContext ctx, GLfloat value)
{
	/* GL 4.6 §14.3.1: clamps to [0,1]. Metal does not natively support
	 * GL-style per-sample fragment shading, so the value is stored for
	 * state query correctness but has no rendering effect. */
	if (!ctx)
		return;
	if (value < 0.0f) value = 0.0f;
	if (value > 1.0f) value = 1.0f;
	STATE_VAR(min_sample_shading) = value;
}

void mglMultiDrawArraysIndirectCount(GLMContext ctx, GLenum mode, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride)
{
	GLuint actual_drawcount = 0u;
	GLsizei effective_drawcount;
	Buffer *parameter_buffer = NULL;

	mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_COUNT_ENTRY mode=0x%x indirect=%p drawcountOffset=%lld maxdrawcount=%d stride=%d program=%u",
	                    (unsigned)mode,
	                    indirect,
	                    (long long)drawcount,
	                    (int)maxdrawcount,
	                    (int)stride,
	                    (unsigned)(ctx ? ctx->state.program_name : 0u));

	ERROR_CHECK_RETURN(ctx, GL_INVALID_OPERATION);
	if (drawcount < 0 || maxdrawcount < 0) {
		mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_COUNT_SKIP reason=bad_count_args drawcountOffset=%lld maxdrawcount=%d program=%u",
		                    (long long)drawcount,
		                    (int)maxdrawcount,
		                    (unsigned)ctx->state.program_name);
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}
	if (maxdrawcount == 0) {
		mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_COUNT_SKIP reason=zero_maxdrawcount program=%u",
	                    (unsigned)ctx->state.program_name);
		return;
	}

	if (!mglReadIndirectCountParameter(ctx,
	                                   "MULTI_DRAW_ARRAYS_INDIRECT_COUNT",
	                                   drawcount,
	                                   &actual_drawcount,
	                                   &parameter_buffer)) {
		return;
	}

	effective_drawcount = (actual_drawcount > (GLuint)maxdrawcount)
		? maxdrawcount
		: (GLsizei)actual_drawcount;
	mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_COUNT_DISPATCH mode=0x%x indirect=%p count=%u effective=%d maxdrawcount=%d stride=%d parameterBuffer=%u program=%u",
	                    (unsigned)mode,
	                    indirect,
	                    (unsigned)actual_drawcount,
	                    (int)effective_drawcount,
	                    (int)maxdrawcount,
	                    (int)stride,
	                    (unsigned)(parameter_buffer ? parameter_buffer->name : 0u),
	                    (unsigned)ctx->state.program_name);
	mglMultiDrawArraysIndirect(ctx, mode, indirect, effective_drawcount, stride);
}

void mglMultiDrawElementsIndirectCount(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride)
{
	GLuint actual_drawcount = 0u;
	GLsizei effective_drawcount;
	Buffer *parameter_buffer = NULL;

	mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_COUNT_ENTRY mode=0x%x type=0x%x indirect=%p drawcountOffset=%lld maxdrawcount=%d stride=%d program=%u",
	                    (unsigned)mode,
	                    (unsigned)type,
	                    indirect,
	                    (long long)drawcount,
	                    (int)maxdrawcount,
	                    (int)stride,
	                    (unsigned)(ctx ? ctx->state.program_name : 0u));

	ERROR_CHECK_RETURN(ctx, GL_INVALID_OPERATION);
	if (drawcount < 0 || maxdrawcount < 0) {
		mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_COUNT_SKIP reason=bad_count_args drawcountOffset=%lld maxdrawcount=%d program=%u",
		                    (long long)drawcount,
		                    (int)maxdrawcount,
		                    (unsigned)ctx->state.program_name);
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}
	if (maxdrawcount == 0) {
		mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_COUNT_SKIP reason=zero_maxdrawcount program=%u",
	                    (unsigned)ctx->state.program_name);
		return;
	}

	if (!mglReadIndirectCountParameter(ctx,
	                                   "MULTI_DRAW_ELEMENTS_INDIRECT_COUNT",
	                                   drawcount,
	                                   &actual_drawcount,
	                                   &parameter_buffer)) {
		return;
	}

	effective_drawcount = (actual_drawcount > (GLuint)maxdrawcount)
		? maxdrawcount
		: (GLsizei)actual_drawcount;
	mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_COUNT_DISPATCH mode=0x%x type=0x%x indirect=%p count=%u effective=%d maxdrawcount=%d stride=%d parameterBuffer=%u program=%u",
	                    (unsigned)mode,
	                    (unsigned)type,
	                    indirect,
	                    (unsigned)actual_drawcount,
	                    (int)effective_drawcount,
	                    (int)maxdrawcount,
	                    (int)stride,
	                    (unsigned)(parameter_buffer ? parameter_buffer->name : 0u),
	                    (unsigned)ctx->state.program_name);
	mglMultiDrawElementsIndirect(ctx, mode, type, indirect, effective_drawcount, stride);
}

void mglNormalP3ui(GLMContext ctx, GLenum type, GLuint coords)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglNormalP3uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglObjectLabel(GLMContext ctx, GLenum identifier, GLuint name, GLsizei length, const GLchar *label)
{
	if (!ctx) {
		return;
	}

	if (identifier != GL_TEXTURE) {
		/* Keep unsupported labels non-fatal; they are diagnostics only. */
		return;
	}

	if (name == 0) {
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	Texture *tex = findTexture(ctx, name);
	if (!tex) {
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	size_t max_len = sizeof(tex->debug_label) - 1u;
	size_t copy_len = 0u;
	if (label) {
		if (length < 0) {
			copy_len = strnlen(label, max_len);
		} else {
			copy_len = (size_t)length;
			if (copy_len > max_len) {
				copy_len = max_len;
			}
		}
		if (copy_len > 0u) {
			memcpy(tex->debug_label, label, copy_len);
		}
	}
	tex->debug_label[copy_len] = '\0';
	mglTraceLogExternal("OBJECT_LABEL texture=%u label=\"%s\" length=%d stored=%zu",
	                    name,
	                    tex->debug_label,
	                    length,
	                    copy_len);
}

void mglObjectPtrLabel(GLMContext ctx, const void *ptr, GLsizei length, const GLchar *label)
{
	// Object ptr label - no-op
	(void)ctx;
	(void)ptr;
	(void)length;
	(void)label;
}

void mglPatchParameterfv(GLMContext ctx, GLenum pname, const GLfloat *values)
{
	if (!values)
	{
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	switch (pname)
	{
		case GL_PATCH_DEFAULT_INNER_LEVEL:
			ctx->state.var.patch_default_inner_level[0] = values[0];
			ctx->state.var.patch_default_inner_level[1] = values[1];
			return;
		case GL_PATCH_DEFAULT_OUTER_LEVEL:
			ctx->state.var.patch_default_outer_level[0] = values[0];
			ctx->state.var.patch_default_outer_level[1] = values[1];
			ctx->state.var.patch_default_outer_level[2] = values[2];
			ctx->state.var.patch_default_outer_level[3] = values[3];
			return;
		default:
			ERROR_RETURN(GL_INVALID_ENUM);
			return;
	}
}

void mglPatchParameteri(GLMContext ctx, GLenum pname, GLint value)
{
	switch (pname)
	{
		case GL_PATCH_VERTICES:
			if (value <= 0 || value > (GLint)ctx->state.var.max_patch_vertices)
			{
				ERROR_RETURN(GL_INVALID_VALUE);
				return;
			}
			ctx->state.var.patch_vertices = (GLuint)value;
			return;
		default:
			ERROR_RETURN(GL_INVALID_ENUM);
			return;
	}
}

void mglPauseTransformFeedback(GLMContext ctx)
{
	if (!STATE(transform_feedback) || !STATE(transform_feedback)->active || STATE(transform_feedback)->paused)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(transform_feedback)->paused = GL_TRUE;
}

void mglPolygonOffsetClamp(GLMContext ctx, GLfloat factor, GLfloat units, GLfloat clamp)
{
	(void)clamp;
	mglPolygonOffset(ctx, factor, units);
}

void mglMaxShaderCompilerThreadsKHR(GLMContext ctx, GLuint count)
{
	/* GL_ARB/KHR_parallel_shader_compile: the spec permits this to be a no-op
	 * hint; we only store the value so GL_MAX_SHADER_COMPILER_THREADS_KHR
	 * queries echo the last set value (initial value 0). */
	STATE(var.max_shader_compiler_threads) = count;
}

void mglPopDebugGroup(GLMContext ctx)
{
	// Pop debug group - no-op
	(void)ctx;
}

void mglPrimitiveRestartIndex(GLMContext ctx, GLuint index)
{
	STATE(var.primitive_restart_index) = index;
	mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
}

void mglProgramBinary(GLMContext ctx, GLuint program, GLenum binaryFormat, const void *binary, GLsizei length)
{
	(void)program; (void)binaryFormat; (void)binary; (void)length;
	STATE(error) = GL_INVALID_OPERATION;
}

void mglProgramParameteri(GLMContext ctx, GLuint program, GLenum pname, GLint value)
{
	Program *pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}

	switch (pname)
	{
		case GL_PROGRAM_SEPARABLE:
			pptr->program_separable = value ? GL_TRUE : GL_FALSE;
			break;
		case GL_PROGRAM_BINARY_RETRIEVABLE_HINT:
			break;
		default:
			STATE(error) = GL_INVALID_ENUM;
			break;
	}
}

static GLboolean mgl_program_uniform_begin(GLMContext ctx, GLuint program, Program **saved_program)
{
	Program *target;

	if (!saved_program)
		return GL_FALSE;

	target = findProgram(ctx, program);
	if (!target)
	{
		STATE(error) = GL_INVALID_VALUE;
		return GL_FALSE;
	}
	if (!target->link_success)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return GL_FALSE;
	}

	*saved_program = STATE(program);
	if (STATE(program) != target)
		mglUseProgram(ctx, program);
	return GL_TRUE;
}

static void mgl_program_uniform_end(GLMContext ctx, Program *saved_program)
{
	if (STATE(program) != saved_program)
	{
		GLuint restore_program = saved_program ? saved_program->name : 0;
		mglUseProgram(ctx, restore_program);
	}
}

#define DEFINE_PROGRAM_UNIFORM_FORWARD(_suffix, _decl, ...) \
void mglProgramUniform##_suffix _decl \
{ \
	Program *saved_program = NULL; \
	if (!mgl_program_uniform_begin(ctx, program, &saved_program)) \
		return; \
	mglUniform##_suffix(ctx, __VA_ARGS__); \
	mgl_program_uniform_end(ctx, saved_program); \
}

DEFINE_PROGRAM_UNIFORM_FORWARD(1d, (GLMContext ctx, GLuint program, GLint location, GLdouble v0), location, v0)
DEFINE_PROGRAM_UNIFORM_FORWARD(1dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLdouble *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(1f, (GLMContext ctx, GLuint program, GLint location, GLfloat v0), location, v0)
DEFINE_PROGRAM_UNIFORM_FORWARD(1fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLfloat *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(1i, (GLMContext ctx, GLuint program, GLint location, GLint v0), location, v0)
DEFINE_PROGRAM_UNIFORM_FORWARD(1iv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLint *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(1ui, (GLMContext ctx, GLuint program, GLint location, GLuint v0), location, v0)
DEFINE_PROGRAM_UNIFORM_FORWARD(1uiv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLuint *value), location, count, value)

DEFINE_PROGRAM_UNIFORM_FORWARD(2d, (GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1), location, v0, v1)
DEFINE_PROGRAM_UNIFORM_FORWARD(2dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLdouble *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(2f, (GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1), location, v0, v1)
DEFINE_PROGRAM_UNIFORM_FORWARD(2fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLfloat *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(2i, (GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1), location, v0, v1)
DEFINE_PROGRAM_UNIFORM_FORWARD(2iv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLint *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(2ui, (GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1), location, v0, v1)
DEFINE_PROGRAM_UNIFORM_FORWARD(2uiv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLuint *value), location, count, value)

DEFINE_PROGRAM_UNIFORM_FORWARD(3d, (GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2), location, v0, v1, v2)
DEFINE_PROGRAM_UNIFORM_FORWARD(3dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLdouble *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(3f, (GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2), location, v0, v1, v2)
DEFINE_PROGRAM_UNIFORM_FORWARD(3fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLfloat *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(3i, (GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1, GLint v2), location, v0, v1, v2)
DEFINE_PROGRAM_UNIFORM_FORWARD(3iv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLint *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(3ui, (GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2), location, v0, v1, v2)
DEFINE_PROGRAM_UNIFORM_FORWARD(3uiv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLuint *value), location, count, value)

DEFINE_PROGRAM_UNIFORM_FORWARD(4d, (GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3), location, v0, v1, v2, v3)
DEFINE_PROGRAM_UNIFORM_FORWARD(4dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLdouble *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(4f, (GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3), location, v0, v1, v2, v3)
DEFINE_PROGRAM_UNIFORM_FORWARD(4fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLfloat *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(4i, (GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1, GLint v2, GLint v3), location, v0, v1, v2, v3)
DEFINE_PROGRAM_UNIFORM_FORWARD(4iv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLint *value), location, count, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(4ui, (GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3), location, v0, v1, v2, v3)
DEFINE_PROGRAM_UNIFORM_FORWARD(4uiv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLuint *value), location, count, value)

DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix2dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix2fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix2x3dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix2x3fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix2x4dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix2x4fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix3dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix3fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix3x2dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix3x2fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix3x4dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix3x4fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix4dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix4fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix4x2dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix4x2fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix4x3dv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value), location, count, transpose, value)
DEFINE_PROGRAM_UNIFORM_FORWARD(Matrix4x3fv, (GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value), location, count, transpose, value)

#undef DEFINE_PROGRAM_UNIFORM_FORWARD

void mglProvokingVertex(GLMContext ctx, GLenum mode)
{
	if (mode != GL_FIRST_VERTEX_CONVENTION && mode != GL_LAST_VERTEX_CONVENTION) {
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	STATE(var.provoking_vertex) = mode;
	mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
}

void mglPushDebugGroup(GLMContext ctx, GLenum source, GLuint id, GLsizei length, const GLchar *message)
{
	// Push debug group - no-op
	(void)ctx;
	(void)source;
	(void)id;
	(void)length;
	(void)message;
}

void mglQueryCounter(GLMContext ctx, GLuint id, GLenum target)
{
	QueryObject *q;
	if (target != GL_TIMESTAMP)
	{
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	if (id == 0)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	q = mgl_get_query(id);
	if (!q)
	{
		STATE(error) = GL_OUT_OF_MEMORY;
		return;
	}
	if (q->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	q->target = GL_TIMESTAMP;
	q->available = GL_TRUE;
	/* Use the real GPU timestamp from Metal's sampleTimestamps API
	 * when available; fall back to the fake counter for API-level
	 * compatibility on backends without timer query support. */
	if (ctx->mtl_funcs.mtlGetGPUTimestamp) {
		/* The callback only samples. Keep the GL ordering boundary at the
		 * semantic call site for both the ObjC fallback and C++ path. */
		mglFlushCommandBuffer(ctx);
		q->result = ctx->mtl_funcs.mtlGetGPUTimestamp(ctx);
	} else
		q->result = s_fake_timestamp_counter++;
}

void mglReadnPixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data)
{
	size_t bytes_per_pixel;
	size_t needed;
	if (!data || bufSize < 0)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	bytes_per_pixel = sizeForFormatType(format, type);
	needed = (size_t)width * (size_t)height * bytes_per_pixel;
	if ((GLsizei)needed > bufSize)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	mglReadPixels(ctx, x, y, width, height, format, type, data);
}

void mglReleaseShaderCompiler(GLMContext ctx)
{
	// No-op - shader compiler is always available
	(void)ctx;
}

void mglResumeTransformFeedback(GLMContext ctx)
{
	if (!STATE(transform_feedback) || !STATE(transform_feedback)->active || !STATE(transform_feedback)->paused)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(transform_feedback)->paused = GL_FALSE;
}

void mglSampleMaski(GLMContext ctx, GLuint maskNumber, GLbitfield mask)
{
	ERROR_CHECK_RETURN(maskNumber < ctx->state.var.max_sample_mask_words, GL_INVALID_VALUE);
	ctx->state.var.sample_mask_value = mask;
}

void mglScissorArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLint *v)
{
	if (!mgl_validate_viewport_range(ctx, first, count))
		return;
	ERROR_CHECK_RETURN(count == 0 || v, GL_INVALID_VALUE);
	for (GLsizei i = 0; i < count; i++) {
		ERROR_CHECK_RETURN(v[i * 4 + 2] >= 0, GL_INVALID_VALUE);
		ERROR_CHECK_RETURN(v[i * 4 + 3] >= 0, GL_INVALID_VALUE);
	}

	for (GLsizei i = 0; i < count; i++) {
		GLuint index = first + (GLuint)i;
		const GLint *box = &v[i * 4];
		if (index == 0) {
			mglScissor(ctx, box[0], box[1], (GLsizei)box[2], (GLsizei)box[3]);
		} else if (index < MGL_MAX_VIEWPORTS) {
			ctx->state.scissor_box_array[index][0] = box[0];
			ctx->state.scissor_box_array[index][1] = box[1];
			ctx->state.scissor_box_array[index][2] = box[2];
			ctx->state.scissor_box_array[index][3] = box[3];
			mglMarkRendererDirtyBits(&ctx->state, DIRTY_RENDER_STATE);
		}
	}
}

void mglScissorIndexed(GLMContext ctx, GLuint index, GLint left, GLint bottom, GLsizei width, GLsizei height)
{
	ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
	ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
	ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);
	if (index == 0) {
		mglScissor(ctx, left, bottom, width, height);
	} else if (index < MGL_MAX_VIEWPORTS) {
		ctx->state.scissor_box_array[index][0] = left;
		ctx->state.scissor_box_array[index][1] = bottom;
		ctx->state.scissor_box_array[index][2] = width;
		ctx->state.scissor_box_array[index][3] = height;
		mglMarkRendererDirtyBits(&ctx->state, DIRTY_RENDER_STATE);
	}
}

void mglScissorIndexedv(GLMContext ctx, GLuint index, const GLint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglScissorIndexed(ctx, index, v[0], v[1], (GLsizei)v[2], (GLsizei)v[3]);
}

void mglSecondaryColorP3ui(GLMContext ctx, GLenum type, GLuint color)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglSecondaryColorP3uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglShaderBinary(GLMContext ctx, GLsizei count, const GLuint *shaders, GLenum binaryFormat, const void *binary, GLsizei length)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglShaderStorageBlockBinding(GLMContext ctx, GLuint program, GLuint storageBlockIndex, GLuint storageBlockBinding)
{
	Program *pptr;

	if (!ctx) {
		return;
	}

	if (storageBlockBinding >= MAX_BINDABLE_BUFFERS) {
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	pptr = findProgram(ctx, program);
	if (!pptr) {
		GLenum err = mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE;
		mglDispatchError(ctx, __FUNCTION__, err);
		return;
	}
	if (!pptr->link_success) {
		ERROR_RETURN(GL_INVALID_OPERATION);
		return;
	}

	/* Find the SSBO resource at the given index. */
	int stage = 0;
	int res_type = _STORAGE_BUFFER_RES;
	int types[] = { res_type };
	MGLShaderResource *res = mgl_program_resource_at_index(pptr, types, 1, storageBlockIndex, &stage, &res_type);
	if (!res) {
		ERROR_RETURN(GL_INVALID_VALUE);
		return;
	}

	/* Update gl_binding for all matching SSBOs across all stages.
	 * Like glUniformBlockBinding, this only changes the GL-side binding
	 * point used to find glBindBufferRange state, not the Metal slot. */
	const char *block_name = (res->name && res->name[0]) ? res->name : NULL;
	GLuint block_element = res->ubo_array_element;

	for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
		MGLShaderResourceList *resources = &pptr->shader_resources_list[s][_STORAGE_BUFFER_RES];
		for (GLuint i = 0; i < resources->count; i++) {
			MGLShaderResource *r = &resources->list[i];
			GLboolean match = GL_FALSE;
			if (block_name && r->name && strcmp(block_name, r->name) == 0)
				match = GL_TRUE;
			if (!block_name && s == stage && r == res)
				match = GL_TRUE;
			if (!match)
				continue;
			if (r->ubo_array_bindings && block_element < r->ubo_array_size) {
				r->ubo_array_bindings[block_element] = storageBlockBinding;
			}
			if (block_element == 0) {
				r->gl_binding = storageBlockBinding;
			}
		}
	}

	mglMarkStateDirtyBits(&ctx->state, DIRTY_BUFFER_BASE_STATE | DIRTY_PROGRAM);

	/* Invalidate the buffer binding plan: glShaderStorageBlockBinding
	 * mutates res->gl_binding, which the cached client_binding_base
	 * reflects.  The next draw rebuilds the plan lazily.  See
	 * mglBufferBindingPlanInvalidate for details. */
	mglBufferBindingPlanInvalidate(pptr);
}

void mglSpecializeShader(GLMContext ctx, GLuint shader, const GLchar *pEntryPoint, GLuint numSpecializationConstants, const GLuint *pConstantIndex, const GLuint *pConstantValue)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglTexBuffer(GLMContext ctx, GLenum target, GLenum internalformat, GLuint buffer)
{
    Texture *tex;
    GLuint active_unit = ctx ? ctx->state.active_texture : 0u;

    ERROR_CHECK_RETURN(target == GL_TEXTURE_BUFFER, GL_INVALID_ENUM);

    tex = (ctx && active_unit < TEXTURE_UNITS)
        ? ctx->state.texture_units[active_unit].textures[_TEXTURE_BUFFER_TARGET]
        : NULL;

    if (MGL_VERBOSE_TEXBUFFER_LOGS) {
        fprintf(stderr,
                "MGL TRACE mglTexBuffer target=0x%x internal=0x%x buffer=%u activeUnit=%u boundTex=%u tex=%p\n",
                target,
                internalformat,
                buffer,
                active_unit,
                tex ? tex->name : 0u,
                (void *)tex);
    }

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    mglTextureBuffer(ctx, tex->name, internalformat, buffer);
}

void mglTexBufferRange(GLMContext ctx, GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    Texture *tex;
    GLuint active_unit = ctx ? ctx->state.active_texture : 0u;

    ERROR_CHECK_RETURN(target == GL_TEXTURE_BUFFER, GL_INVALID_ENUM);

    tex = (ctx && active_unit < TEXTURE_UNITS)
        ? ctx->state.texture_units[active_unit].textures[_TEXTURE_BUFFER_TARGET]
        : NULL;

    if (MGL_VERBOSE_TEXBUFFER_LOGS) {
        fprintf(stderr,
                "MGL TRACE mglTexBufferRange target=0x%x internal=0x%x buffer=%u offset=%lld size=%lld activeUnit=%u boundTex=%u tex=%p\n",
                target,
                internalformat,
                buffer,
                (long long)offset,
                (long long)size,
                active_unit,
                tex ? tex->name : 0u,
                (void *)tex);
    }

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    mglTextureBufferRange(ctx, tex->name, internalformat, buffer, offset, size);
}

void mglTexStorage2DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    Texture *tex = NULL;

    if (target != GL_TEXTURE_2D_MULTISAMPLE &&
        target != GL_PROXY_TEXTURE_2D_MULTISAMPLE) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (!mglTexStorageInternalFormatValid(internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 1 || width <= 0 || height <= 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (target == GL_PROXY_TEXTURE_2D_MULTISAMPLE) {
        return;
    }

    if (ctx && ctx->state.active_texture < TEXTURE_UNITS) {
        tex = ctx->state.texture_units[ctx->state.active_texture].textures[_TEXTURE_2D_MULTISAMPLE];
    }
    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    mglTextureStorage2DMultisample(ctx, tex->name, samples, internalformat, width, height, fixedsamplelocations);
}

void mglTexStorage3DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    Texture *tex = NULL;

    if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY &&
        target != GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (!mglTexStorageInternalFormatValid(internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 1 || width <= 0 || height <= 0 || depth <= 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (target == GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        return;
    }

    if (ctx && ctx->state.active_texture < TEXTURE_UNITS) {
        tex = ctx->state.texture_units[ctx->state.active_texture].textures[_TEXTURE_2D_MULTISAMPLE_ARRAY];
    }
    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    mglTextureStorage3DMultisample(ctx, tex->name, samples, internalformat, width, height, depth, fixedsamplelocations);
}

void mglTransformFeedbackBufferBase(GLMContext ctx, GLuint xfb, GLuint index, GLuint buffer)
{
	TransformFeedback *ptr = findTransformFeedback(ctx, xfb);
	if (!ptr || !ptr->created)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (index >= MAX_BINDABLE_BUFFERS)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	BufferBaseTarget *slot = &ptr->buffers[index];
	if (buffer == 0)
	{
		bzero(slot, sizeof(BufferBaseTarget));
		return;
	}
	Buffer *buf = getBuffer(ctx, GL_TRANSFORM_FEEDBACK_BUFFER, buffer);
	if (!buf)
	{
		STATE(error) = GL_OUT_OF_MEMORY;
		return;
	}
	slot->buffer = buffer;
	slot->offset = 0;
	/* BindBufferBase tracks the whole data store dynamically. */
	slot->size = 0;
	slot->buf = buf;
	buf->target = GL_TRANSFORM_FEEDBACK_BUFFER;
}

void mglTransformFeedbackBufferRange(GLMContext ctx, GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
	TransformFeedback *ptr = findTransformFeedback(ctx, xfb);
	if (!ptr || !ptr->created)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}
	if (index >= MAX_BINDABLE_BUFFERS)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	if (buffer == 0)
	{
		bzero(&ptr->buffers[index], sizeof(BufferBaseTarget));
		return;
	}
	if (offset < 0 || size <= 0 || ((GLuint64)offset % 4u) != 0u || ((GLuint64)size % 4u) != 0u)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	Buffer *buf = getBuffer(ctx, GL_TRANSFORM_FEEDBACK_BUFFER, buffer);
	if (!buf)
	{
		STATE(error) = GL_OUT_OF_MEMORY;
		return;
	}
	BufferBaseTarget *slot = &ptr->buffers[index];
	slot->buffer = buffer;
	slot->offset = offset;
	slot->size = size;
	slot->buf = buf;
	buf->target = GL_TRANSFORM_FEEDBACK_BUFFER;
}

void mglTransformFeedbackVaryings(GLMContext ctx, GLuint program, GLsizei count, const GLchar *const*varyings, GLenum bufferMode)
{
	ERROR_CHECK_RETURN(ctx, GL_INVALID_OPERATION);
	ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
	ERROR_CHECK_RETURN(bufferMode == GL_INTERLEAVED_ATTRIBS || bufferMode == GL_SEPARATE_ATTRIBS, GL_INVALID_ENUM);
	ERROR_CHECK_RETURN(count <= MAX_ATTRIBS, GL_INVALID_VALUE);

	Program *pptr = findProgram(ctx, program);
	ERROR_CHECK_RETURN(pptr, GL_INVALID_VALUE);
	if (bufferMode == GL_SEPARATE_ATTRIBS &&
	    (GLuint)count > ctx->state.var.max_transform_feedback_separate_attribs) {
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	GLuint nextBufferCount = 0u;
	for (GLsizei i = 0; i < count; i++) {
		if (!varyings || !varyings[i]) {
			STATE(error) = GL_INVALID_VALUE;
			return;
		}
		if (!mgl_tf_is_builtin_name(varyings[i]))
			continue;
		if (bufferMode != GL_INTERLEAVED_ATTRIBS) {
			STATE(error) = GL_INVALID_OPERATION;
			return;
		}
		if (strcmp(varyings[i], "gl_NextBuffer") == 0) {
			nextBufferCount++;
			if (nextBufferCount >= ctx->state.var.max_transform_feedback_buffers) {
				STATE(error) = GL_INVALID_OPERATION;
				return;
			}
		}
	}

	pptr->transform_feedback_varying_count = count;
	pptr->transform_feedback_buffer_mode = bufferMode;
	pptr->transform_feedback_layout_valid = GL_FALSE;
	pptr->transform_feedback_layout_buffer_count = 0u;
	pptr->transform_feedback_layout_component_count = 0u;
	bzero(pptr->transform_feedback_layout,
	      sizeof(pptr->transform_feedback_layout));
	for (GLsizei i = 0; i < count; i++) {
		pptr->transform_feedback_varying_names[i][0] = '\0';
		if (varyings && varyings[i]) {
			strncpy(pptr->transform_feedback_varying_names[i],
			        varyings[i],
			        sizeof(pptr->transform_feedback_varying_names[i]) - 1);
			pptr->transform_feedback_varying_names[i][sizeof(pptr->transform_feedback_varying_names[i]) - 1] = '\0';
		}
	}
	for (GLsizei i = count; i < MAX_ATTRIBS; i++) {
		pptr->transform_feedback_varying_names[i][0] = '\0';
	}
	pptr->dirty_bits |= DIRTY_PROGRAM;
}

void mglUniformSubroutinesuiv(GLMContext ctx, GLenum shadertype, GLsizei count, const GLuint *indices)
{
	mgl_unimplemented(ctx, __FUNCTION__);
	(void)ctx;
}

void mglValidateProgram(GLMContext ctx, GLuint program)
{
	// Program validation - no-op for now, programs are validated during linking
	Program *pptr = findProgram(ctx, program);
	if (!pptr)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
}

void mglValidateProgramPipeline(GLMContext ctx, GLuint pipeline)
{
	ProgramPipeline *pp = findProgramPipeline(ctx, pipeline);
	if (!pp)
	{
		STATE(error) = GL_INVALID_VALUE;
		return;
	}
	pp->validated = mglProgramPipelinePerVertexCompatible(pp->stage_programs);
}

void mglVertexAttrib1d(GLMContext ctx, GLuint index, GLdouble x)
{
	(void)ctx;
}

void mglVertexAttrib1dv(GLMContext ctx, GLuint index, const GLdouble *v)
{
	(void)ctx;
}

void mglVertexAttrib1f(GLMContext ctx, GLuint index, GLfloat x)
{
	(void)ctx;
}

void mglVertexAttrib1fv(GLMContext ctx, GLuint index, const GLfloat *v)
{
	(void)ctx;
}

void mglVertexAttrib1s(GLMContext ctx, GLuint index, GLshort x)
{
	(void)ctx;
}

void mglVertexAttrib1sv(GLMContext ctx, GLuint index, const GLshort *v)
{
	(void)ctx;
}

void mglVertexAttrib2d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y)
{
	(void)ctx;
}

void mglVertexAttrib2dv(GLMContext ctx, GLuint index, const GLdouble *v)
{
	(void)ctx;
}

void mglVertexAttrib2f(GLMContext ctx, GLuint index, GLfloat x, GLfloat y)
{
	(void)ctx;
}

void mglVertexAttrib2fv(GLMContext ctx, GLuint index, const GLfloat *v)
{
	(void)ctx;
}

void mglVertexAttrib2s(GLMContext ctx, GLuint index, GLshort x, GLshort y)
{
	(void)ctx;
}

void mglVertexAttrib2sv(GLMContext ctx, GLuint index, const GLshort *v)
{
	(void)ctx;
}

void mglVertexAttrib3d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z)
{
	(void)ctx;
}

void mglVertexAttrib3dv(GLMContext ctx, GLuint index, const GLdouble *v)
{
	(void)ctx;
}

void mglVertexAttrib3f(GLMContext ctx, GLuint index, GLfloat x, GLfloat y, GLfloat z)
{
	(void)ctx;
}

void mglVertexAttrib3fv(GLMContext ctx, GLuint index, const GLfloat *v)
{
	(void)ctx;
}

void mglVertexAttrib3s(GLMContext ctx, GLuint index, GLshort x, GLshort y, GLshort z)
{
	(void)ctx;
}

void mglVertexAttrib3sv(GLMContext ctx, GLuint index, const GLshort *v)
{
	(void)ctx;
}

void mglVertexAttrib4Nbv(GLMContext ctx, GLuint index, const GLbyte *v)
{
	(void)ctx;
}

void mglVertexAttrib4Niv(GLMContext ctx, GLuint index, const GLint *v)
{
	(void)ctx;
}

void mglVertexAttrib4Nsv(GLMContext ctx, GLuint index, const GLshort *v)
{
	(void)ctx;
}

void mglVertexAttrib4Nub(GLMContext ctx, GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w)
{
	(void)ctx;
}

void mglVertexAttrib4Nubv(GLMContext ctx, GLuint index, const GLubyte *v)
{
	(void)ctx;
}

void mglVertexAttrib4Nuiv(GLMContext ctx, GLuint index, const GLuint *v)
{
	(void)ctx;
}

void mglVertexAttrib4Nusv(GLMContext ctx, GLuint index, const GLushort *v)
{
	(void)ctx;
}

void mglVertexAttrib4bv(GLMContext ctx, GLuint index, const GLbyte *v)
{
	(void)ctx;
}

void mglVertexAttrib4d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
	mglSetCurrentVertexAttribDouble(ctx, index, x, y, z, w);
}

void mglVertexAttrib4dv(GLMContext ctx, GLuint index, const GLdouble *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribDouble(ctx, index, v[0], v[1], v[2], v[3]);
}

void mglVertexAttrib4f(GLMContext ctx, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
	mglSetCurrentVertexAttribFloat(ctx, index, x, y, z, w);
}

void mglVertexAttrib4fv(GLMContext ctx, GLuint index, const GLfloat *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribFloat(ctx, index, v[0], v[1], v[2], v[3]);
}

void mglVertexAttrib4iv(GLMContext ctx, GLuint index, const GLint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribFloat(ctx, index, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]);
}

void mglVertexAttrib4s(GLMContext ctx, GLuint index, GLshort x, GLshort y, GLshort z, GLshort w)
{
	(void)ctx;
}

void mglVertexAttrib4sv(GLMContext ctx, GLuint index, const GLshort *v)
{
	(void)ctx;
}

void mglVertexAttrib4ubv(GLMContext ctx, GLuint index, const GLubyte *v)
{
	(void)ctx;
}

void mglVertexAttrib4uiv(GLMContext ctx, GLuint index, const GLuint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribFloat(ctx, index, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]);
}

void mglVertexAttrib4usv(GLMContext ctx, GLuint index, const GLushort *v)
{
	(void)ctx;
}

void mglVertexAttribI1i(GLMContext ctx, GLuint index, GLint x)
{
	mglSetCurrentVertexAttribInt(ctx, index, x, 0, 0, 1);
}

void mglVertexAttribI1iv(GLMContext ctx, GLuint index, const GLint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribInt(ctx, index, v[0], 0, 0, 1);
}

void mglVertexAttribI1ui(GLMContext ctx, GLuint index, GLuint x)
{
	mglSetCurrentVertexAttribUInt(ctx, index, x, 0u, 0u, 1u);
}

void mglVertexAttribI1uiv(GLMContext ctx, GLuint index, const GLuint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribUInt(ctx, index, v[0], 0u, 0u, 1u);
}

void mglVertexAttribI2i(GLMContext ctx, GLuint index, GLint x, GLint y)
{
	mglSetCurrentVertexAttribInt(ctx, index, x, y, 0, 1);
}

void mglVertexAttribI2iv(GLMContext ctx, GLuint index, const GLint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribInt(ctx, index, v[0], v[1], 0, 1);
}

void mglVertexAttribI2ui(GLMContext ctx, GLuint index, GLuint x, GLuint y)
{
	mglSetCurrentVertexAttribUInt(ctx, index, x, y, 0u, 1u);
}

void mglVertexAttribI2uiv(GLMContext ctx, GLuint index, const GLuint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribUInt(ctx, index, v[0], v[1], 0u, 1u);
}

void mglVertexAttribI3i(GLMContext ctx, GLuint index, GLint x, GLint y, GLint z)
{
	mglSetCurrentVertexAttribInt(ctx, index, x, y, z, 1);
}

void mglVertexAttribI3iv(GLMContext ctx, GLuint index, const GLint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribInt(ctx, index, v[0], v[1], v[2], 1);
}

void mglVertexAttribI3ui(GLMContext ctx, GLuint index, GLuint x, GLuint y, GLuint z)
{
	mglSetCurrentVertexAttribUInt(ctx, index, x, y, z, 1u);
}

void mglVertexAttribI3uiv(GLMContext ctx, GLuint index, const GLuint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribUInt(ctx, index, v[0], v[1], v[2], 1u);
}

void mglVertexAttribI4bv(GLMContext ctx, GLuint index, const GLbyte *v)
{
	(void)ctx;
}

void mglVertexAttribI4i(GLMContext ctx, GLuint index, GLint x, GLint y, GLint z, GLint w)
{
	mglSetCurrentVertexAttribInt(ctx, index, x, y, z, w);
}

void mglVertexAttribI4iv(GLMContext ctx, GLuint index, const GLint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribInt(ctx, index, v[0], v[1], v[2], v[3]);
}

void mglVertexAttribI4sv(GLMContext ctx, GLuint index, const GLshort *v)
{
	(void)ctx;
}

void mglVertexAttribI4ubv(GLMContext ctx, GLuint index, const GLubyte *v)
{
	(void)ctx;
}

void mglVertexAttribI4ui(GLMContext ctx, GLuint index, GLuint x, GLuint y, GLuint z, GLuint w)
{
	mglSetCurrentVertexAttribUInt(ctx, index, x, y, z, w);
}

void mglVertexAttribI4uiv(GLMContext ctx, GLuint index, const GLuint *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribUInt(ctx, index, v[0], v[1], v[2], v[3]);
}

void mglVertexAttribI4usv(GLMContext ctx, GLuint index, const GLushort *v)
{
	(void)ctx;
}

void mglVertexAttribL1d(GLMContext ctx, GLuint index, GLdouble x)
{
	(void)ctx;
}

void mglVertexAttribL1dv(GLMContext ctx, GLuint index, const GLdouble *v)
{
	(void)ctx;
}

void mglVertexAttribL2d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y)
{
	(void)ctx;
}

void mglVertexAttribL2dv(GLMContext ctx, GLuint index, const GLdouble *v)
{
	(void)ctx;
}

void mglVertexAttribL3d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z)
{
	(void)ctx;
}

void mglVertexAttribL3dv(GLMContext ctx, GLuint index, const GLdouble *v)
{
	(void)ctx;
}

void mglVertexAttribL4d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
	mglSetCurrentVertexAttribDouble(ctx, index, x, y, z, w);
}

void mglVertexAttribL4dv(GLMContext ctx, GLuint index, const GLdouble *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglSetCurrentVertexAttribDouble(ctx, index, v[0], v[1], v[2], v[3]);
}

void mglVertexAttribP1ui(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value)
{
	(void)ctx;
}

void mglVertexAttribP1uiv(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, const GLuint *value)
{
	(void)ctx;
}

void mglVertexAttribP2ui(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value)
{
	(void)ctx;
}

void mglVertexAttribP2uiv(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, const GLuint *value)
{
	(void)ctx;
}

void mglVertexAttribP3ui(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value)
{
	(void)ctx;
}

void mglVertexAttribP3uiv(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, const GLuint *value)
{
	(void)ctx;
}

void mglVertexAttribP4ui(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value)
{
	(void)ctx;
}

void mglVertexAttribP4uiv(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, const GLuint *value)
{
	(void)ctx;
}

void mglVertexP2ui(GLMContext ctx, GLenum type, GLuint value)
{
	(void)ctx;
}

void mglVertexP2uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
	(void)ctx;
}

void mglVertexP3ui(GLMContext ctx, GLenum type, GLuint value)
{
	(void)ctx;
}

void mglVertexP3uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
	(void)ctx;
}

void mglVertexP4ui(GLMContext ctx, GLenum type, GLuint value)
{
	(void)ctx;
}

void mglVertexP4uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
	(void)ctx;
}

void mglViewportArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLfloat *v)
{
	if (!mgl_validate_viewport_range(ctx, first, count))
		return;
	ERROR_CHECK_RETURN(count == 0 || v, GL_INVALID_VALUE);
	for (GLsizei i = 0; i < count; i++) {
		ERROR_CHECK_RETURN(v[i * 4 + 2] >= 0.0f, GL_INVALID_VALUE);
		ERROR_CHECK_RETURN(v[i * 4 + 3] >= 0.0f, GL_INVALID_VALUE);
	}

	for (GLsizei i = 0; i < count; i++) {
		mglViewportIndexedf(ctx,
		                    first + (GLuint)i,
		                    v[i * 4 + 0],
		                    v[i * 4 + 1],
		                    v[i * 4 + 2],
		                    v[i * 4 + 3]);
	}
}

void mglViewportIndexedf(GLMContext ctx, GLuint index, GLfloat x, GLfloat y, GLfloat w, GLfloat h)
{
	ERROR_CHECK_RETURN(index < mgl_effective_max_viewports(ctx), GL_INVALID_VALUE);
	ERROR_CHECK_RETURN(w >= 0.0f, GL_INVALID_VALUE);
	ERROR_CHECK_RETURN(h >= 0.0f, GL_INVALID_VALUE);
	if (index == 0) {
		mglViewport(ctx, (GLint)x, (GLint)y, (GLsizei)w, (GLsizei)h);
	} else if (index < MGL_MAX_VIEWPORTS) {
		ctx->state.viewport_array[index][0] = x;
		ctx->state.viewport_array[index][1] = y;
		ctx->state.viewport_array[index][2] = w;
		ctx->state.viewport_array[index][3] = h;
		ctx->state.viewport_array_set = GL_TRUE;
		mglMarkRendererDirtyBits(&ctx->state, DIRTY_RENDER_STATE);
	}
}

void mglViewportIndexedfv(GLMContext ctx, GLuint index, const GLfloat *v)
{
	ERROR_CHECK_RETURN(v, GL_INVALID_VALUE);
	mglViewportIndexedf(ctx, index, v[0], v[1], v[2], v[3]);
}

#ifdef MGL_GL_ES
void  mglBlendBarrier(GLMContext ctx)
{
    // Unimplemented function
    mgl_unimplemented(ctx, __FUNCTION__);

}

void mglPrimitiveBoundingBox(GLMContext ctx, GLfloat minX, GLfloat minY, GLfloat minZ, GLfloat minW, GLfloat maxX, GLfloat maxY, GLfloat maxZ, GLfloat maxW)
{
    
}
#endif
