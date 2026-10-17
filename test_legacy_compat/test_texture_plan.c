/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Texture creation plan harness (audit P4).
 *
 * Golden for mglTexturePlanBuild(): the Metal descriptor a GL texture turns
 * into.  The expectations were read off mglTextureCreateFromGLTexture before
 * the descriptor logic moved into the plan; reference_plan() below is that
 * sequence kept verbatim, and a sweep over targets, formats, mip state and
 * dimensions requires the plan to agree with it on every field.
 *
 * Build: see `make test-texture-plan`.
 */

#include "mgl_texture_plan.h"

#include <stdio.h>
#include <string.h>

#include "glcorearb.h"
#include "mgl_render_values.h"

static int tests_run;
static int tests_passed;

#define CHECK(cond, label)                                       \
    do {                                                         \
        tests_run++;                                             \
        if (cond) {                                              \
            tests_passed++;                                      \
        } else {                                                 \
            printf("  [FAIL] %s\n", (label));                    \
        }                                                        \
    } while (0)

static uint64_t umax(uint64_t a, uint64_t b) { return a > b ? a : b; }

/* ---- the retired inline sequence ------------------------------------------ */

typedef struct Reference {
    MGLRenderTextureDescriptorState desc;
    uint32_t tex_type;
    uint32_t pixel_format;
    uint32_t effective_levels;
    uint32_t upload_levels;
    int storage_mipmapped, is_array, b1d, b1da, ms, agx, has_data, needs_upload;
    int clear_tex_mipmapped;
    int access_ok;
} Reference;

static void reference_plan(const MGLTexturePlanInput *in, Reference *r)
{
    memset(r, 0, sizeof(*r));
    uint32_t tex_type = in->target.texture_type;
    r->is_array = in->target.is_array != 0u;
    r->b1d = in->target.texture_1d_backed_by_2d != 0u;
    r->b1da = in->target.texture_1d_array_backed_by_2d_array != 0u;
    uint32_t effective = in->effective_mipmap_levels;
    int storage_mipmapped = in->effective_mipmapped ? 1 : 0;
    int tex_mipmapped = in->tex_mipmapped ? 1 : 0;

    int converted = 0;
    uint32_t pixel_format =
        mglRenderAGXCompatiblePixelFormat(in->pixel_format, &converted);
    r->agx = converted != 0;
    if (mglRenderPromote1DArrayDepthStencil(tex_type, pixel_format)) {
        tex_type = MGLTextureType2DArray;
        r->b1da = 1;
    }
    if (tex_type == MGLTextureType2DMultisample ||
        tex_type == MGLTextureType2DMultisampleArray) {
        storage_mipmapped = 0;
        effective = 1u;
        tex_mipmapped = 0;
        r->clear_tex_mipmapped = 1;
    }
    const int mipmapped = storage_mipmapped;
    r->upload_levels =
        mglRenderUploadLevelCount(mipmapped ? 1 : 0, tex_mipmapped, effective);

    MGLRenderTextureDescriptorState *d = &r->desc;
    d->texture_type = tex_type;
    d->pixel_format = pixel_format;
    d->width = in->width;
    d->height = mglRenderTextureDescHeight(tex_type, (uint32_t)in->height);
    int ms = 0;
    {
        uint32_t out_type = tex_type, sc = 1u;
        uint64_t arr = 1u, dd = 1u;
        if (mglRenderEmulateMSAsArray(tex_type, 2u, in->depth, &out_type, &sc,
                                      &arr, &dd)) {
            ms = 1;
            tex_type = out_type;
            d->texture_type = tex_type;
            d->sample_count = sc;
            d->array_length = arr;
            d->depth = dd;
        }
    }
    d->cpu_cache_mode = in->conservative_cpu_cache ? 1u : 0u;
    r->has_data = in->first_uploadable_level < r->upload_levels;
    r->needs_upload = in->dirty_texture_data && r->has_data;
    d->storage_mode =
        mglRenderPreferSharedStorage(r->needs_upload, in->format_is_depth_or_stencil)
            ? 0u
            : 2u;
    d->sample_count = umax(d->sample_count, 1u);
    d->mipmap_level_count = umax(d->mipmap_level_count, 1u);
    d->array_length = umax(d->array_length, 1u);
    d->depth = umax(d->depth, 1u);
    {
        uint64_t arr = d->array_length, dd = d->depth;
        if (mglRenderTextureArrayDepthForType(tex_type, r->is_array, ms, in->width,
                                              in->height, in->depth, &arr, &dd)) {
            d->array_length = arr;
            d->depth = dd;
        }
    }
    if (mipmapped) {
        if (mglRenderPromoteMipmapped1D(tex_type)) {
            tex_type = MGLTextureType2D;
            r->b1d = 1;
        }
        if (mglRenderPromoteMipmapped1DArray(tex_type)) {
            tex_type = MGLTextureType2DArray;
            r->b1da = 1;
        }
        d->mipmap_level_count = umax(1u, effective);
    }
    if (r->b1d || r->b1da) {
        uint32_t t = d->texture_type;
        uint64_t arr = d->array_length;
        uint32_t h = (uint32_t)d->height;
        mglRenderApply1DBackingToDesc(r->b1d, r->b1da, in->height, &t, &arr, &h);
        d->texture_type = t;
        d->array_length = arr;
        d->height = h;
    }
    r->ms = ms;
    r->tex_type = tex_type;
    r->pixel_format = pixel_format;
    r->effective_levels = effective;
    r->storage_mipmapped = storage_mipmapped;

    uint32_t usage = 0u;
    r->access_ok = mglRenderTextureUsageForAccess(in->access, &usage);
    if (!r->access_ok) {
        return;
    }
    d->usage = usage;
    if (mglRenderPixelFormatNeedsShaderAtomic(pixel_format)) {
        d->usage |= 0x20u;
    }
    if (in->is_render_target) {
        d->usage |= 4u | 1u;
    }
    d->usage |= 16u;
}

static int same_desc(const MGLRenderTextureDescriptorState *a,
                     const MGLRenderTextureDescriptorState *b)
{
    return a->texture_type == b->texture_type &&
           a->pixel_format == b->pixel_format && a->width == b->width &&
           a->height == b->height && a->depth == b->depth &&
           a->mipmap_level_count == b->mipmap_level_count &&
           a->sample_count == b->sample_count &&
           a->array_length == b->array_length && a->usage == b->usage &&
           a->cpu_cache_mode == b->cpu_cache_mode &&
           a->storage_mode == b->storage_mode;
}

/* ---- helpers --------------------------------------------------------------- */

static MGLTexturePlanInput input_for(uint32_t gl_target, uint32_t samples,
                                     uint32_t format)
{
    MGLTexturePlanInput in;
    memset(&in, 0, sizeof(in));
    mglRenderTextureTargetPlan(gl_target, samples, &in.target);
    in.width = 64u;
    in.height = 32u;
    in.depth = 1u;
    in.access = GL_READ_WRITE;
    in.effective_mipmap_levels = 1u;
    in.pixel_format = format;
    in.first_uploadable_level = MGL_TEXTURE_PLAN_NO_LEVEL;
    return in;
}

static void test_basic_shapes(void)
{
    MGLTexturePlan plan;
    MGLTexturePlanInput in = input_for(GL_TEXTURE_2D, 0u, 70u);
    CHECK(mglTexturePlanBuild(&in, &plan) == 0 && plan.status == MGL_TEXTURE_PLAN_OK,
          "a plain 2D texture plans");
    CHECK(plan.desc.texture_type == MGLTextureType2D && plan.desc.width == 64u &&
              plan.desc.height == 32u && plan.desc.depth == 1u &&
              plan.desc.array_length == 1u && plan.desc.sample_count == 1u &&
              plan.desc.mipmap_level_count == 1u,
          "2D shape");
    CHECK(plan.desc.usage == (MGL_TEXTURE_USAGE_SHADER_READ |
                              MGL_TEXTURE_USAGE_SHADER_WRITE |
                              MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW),
          "a plain texture gets read, write and pixel-format-view usage");
    CHECK(plan.desc.storage_mode == MGL_TEXTURE_STORAGE_PRIVATE,
          "no CPU upload and no depth: private storage");

    in = input_for(GL_TEXTURE_1D, 0u, 70u);
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.desc.texture_type == MGLTextureType2D && plan.desc.height == 1u &&
              plan.texture_1d_backed_by_2d,
          "1D is a one-row 2D texture");

    in = input_for(GL_TEXTURE_1D, 0u, 70u);
    in.effective_mipmapped = 1;
    in.tex_mipmapped = 1;
    in.effective_mipmap_levels = 7u;
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.desc.mipmap_level_count == 7u && plan.upload_texture_type == MGLTextureType2D &&
              plan.upload_level_count == 7u,
          "a mipmapped 1D texture keeps its chain on the 2D backing");

    in = input_for(GL_TEXTURE_1D_ARRAY, 0u, 70u);
    in.height = 5u; /* GL keeps the layer count in height */
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.desc.texture_type == MGLTextureType2DArray &&
              plan.desc.array_length == 5u && plan.desc.height == 1u &&
              plan.texture_1d_array_backed_by_2d_array && plan.is_array,
          "1D array is a one-row 2D array with height as its layer count");

    in = input_for(GL_TEXTURE_CUBE_MAP, 0u, 70u);
    in.height = 64u;
    mglTexturePlanBuild(&in, &plan);
    CHECK(in.target.num_faces == 6u && plan.desc.texture_type == MGLTextureTypeCube &&
              plan.desc.array_length == 1u && plan.desc.depth == 1u,
          "cube map");

    in = input_for(GL_TEXTURE_CUBE_MAP_ARRAY, 0u, 70u);
    in.height = 64u;
    in.depth = 12u; /* cubes * 6 */
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.desc.texture_type == MGLTextureTypeCubeArray &&
              plan.desc.array_length == 2u && plan.desc.depth == 1u,
          "cube array depth counts cubes, not faces");

    in = input_for(GL_TEXTURE_3D, 0u, 70u);
    in.depth = 9u;
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.desc.texture_type == MGLTextureType3D && plan.desc.depth == 9u &&
              plan.desc.array_length == 1u,
          "3D keeps its depth");
}

static void test_multisample(void)
{
    MGLTexturePlan plan;
    MGLTexturePlanInput in = input_for(GL_TEXTURE_2D_MULTISAMPLE, 4u, 70u);
    in.effective_mipmapped = 1;
    in.tex_mipmapped = 1;
    in.effective_mipmap_levels = 4u;
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.ms_emulated_as_array && plan.desc.texture_type == MGLTextureType2DArray &&
              plan.desc.sample_count == 1u && plan.desc.array_length == 8u &&
              plan.desc.depth == 1u,
          "multisample becomes eight planes of a 2D array");
    CHECK(plan.clear_tex_mipmapped && plan.desc.mipmap_level_count == 1u &&
              plan.upload_level_count == 1u && !plan.storage_mipmapped,
          "multisample has one level and clears the mipmapped flag");

    in = input_for(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 4u, 70u);
    in.depth = 3u;
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.ms_emulated_as_array && plan.desc.array_length == 24u,
          "multisample array: layers * 8 planes");

    in = input_for(GL_RENDERBUFFER, 4u, 70u);
    mglTexturePlanBuild(&in, &plan);
    CHECK(in.target.texture_type == MGLTextureType2DMultisample &&
              plan.ms_emulated_as_array,
          "a multisampled renderbuffer takes the same path");
    in = input_for(GL_RENDERBUFFER, 1u, 70u);
    mglTexturePlanBuild(&in, &plan);
    CHECK(in.target.texture_type == MGLTextureType2D && !plan.ms_emulated_as_array,
          "a single-sample renderbuffer is plain 2D");
}

static void test_format_usage_and_storage(void)
{
    MGLTexturePlan plan;
    MGLTexturePlanInput in = input_for(GL_TEXTURE_2D, 0u, 40u /* B5G6R5 */);
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.agx_converted && plan.pixel_format == 70u &&
              plan.desc.pixel_format == 70u,
          "a 16-bit packed format is replaced by RGBA8 for AGX");

    in = input_for(GL_TEXTURE_2D, 0u, 53u /* R32Uint */);
    mglTexturePlanBuild(&in, &plan);
    CHECK((plan.desc.usage & MGL_TEXTURE_USAGE_SHADER_ATOMIC) != 0u,
          "R32Uint gets ShaderAtomic");

    in = input_for(GL_TEXTURE_2D, 0u, 70u);
    in.is_render_target = 1;
    mglTexturePlanBuild(&in, &plan);
    CHECK((plan.desc.usage & MGL_TEXTURE_USAGE_RENDER_TARGET) != 0u &&
              (plan.desc.usage & MGL_TEXTURE_USAGE_SHADER_READ) != 0u,
          "a render target gets RenderTarget usage");

    in = input_for(GL_TEXTURE_1D_ARRAY, 0u, 252u /* Depth32Float */);
    in.format_is_depth_or_stencil = 1;
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.upload_texture_type == MGLTextureType2DArray &&
              plan.texture_1d_array_backed_by_2d_array &&
              plan.desc.storage_mode == MGL_TEXTURE_RESOURCE_STORAGE_SHARED,
          "a depth 1D array is promoted and lives in shared storage");

    in = input_for(GL_TEXTURE_2D, 0u, 70u);
    in.dirty_texture_data = 1;
    in.first_uploadable_level = 0u;
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.needs_cpu_upload && plan.has_uploadable_cpu_data &&
              plan.desc.storage_mode == MGL_TEXTURE_RESOURCE_STORAGE_SHARED,
          "dirty data that exists takes shared storage");

    in.first_uploadable_level = 3u; /* beyond the single level walked */
    mglTexturePlanBuild(&in, &plan);
    CHECK(!plan.has_uploadable_cpu_data && !plan.needs_cpu_upload &&
              plan.desc.storage_mode == MGL_TEXTURE_STORAGE_PRIVATE,
          "data only in levels the plan does not walk does not count");

    in.first_uploadable_level = 0u;
    in.dirty_texture_data = 0;
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.has_uploadable_cpu_data && !plan.needs_cpu_upload,
          "clean data is not uploaded");

    in = input_for(GL_TEXTURE_2D, 0u, 70u);
    in.conservative_cpu_cache = 1;
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.desc.cpu_cache_mode == MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED,
          "the conservative device uses write-combined CPU caching");

    in = input_for(GL_TEXTURE_2D_MULTISAMPLE, 4u, 70u);
    in.access = 0x1234u; /* not an image access mode */
    mglTexturePlanBuild(&in, &plan);
    CHECK(plan.status == MGL_TEXTURE_PLAN_BAD_ACCESS && plan.clear_tex_mipmapped,
          "a bad access mode is reported, and the multisample side effect still shows");

    CHECK(mglTexturePlanBuild(NULL, &plan) == -1 &&
              mglTexturePlanBuild(&in, NULL) == -1,
          "NULL arguments are refused");
}

/* ---- sweep ------------------------------------------------------------------ */

static void test_sweep(void)
{
    static const uint32_t targets[] = {
        GL_TEXTURE_1D, GL_TEXTURE_1D_ARRAY, GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE,
        GL_TEXTURE_2D_ARRAY, GL_TEXTURE_2D_MULTISAMPLE,
        GL_TEXTURE_2D_MULTISAMPLE_ARRAY, GL_TEXTURE_CUBE_MAP,
        GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_3D, GL_RENDERBUFFER};
    static const uint32_t formats[] = {70u, 80u, 40u, 53u, 252u, 253u, 260u};
    static const uint32_t samples[] = {0u, 1u, 4u};
    static const uint32_t levels[] = {1u, 5u};
    static const uint64_t depths[] = {1u, 6u, 12u};
    int agree = 1;
    int cases = 0;

    for (unsigned t = 0; t < sizeof(targets) / sizeof(targets[0]); t++)
    for (unsigned f = 0; f < sizeof(formats) / sizeof(formats[0]); f++)
    for (unsigned s = 0; s < sizeof(samples) / sizeof(samples[0]); s++)
    for (unsigned l = 0; l < sizeof(levels) / sizeof(levels[0]); l++)
    for (unsigned d = 0; d < sizeof(depths) / sizeof(depths[0]); d++)
    for (unsigned flags = 0; flags < 32u; flags++) {
        MGLTexturePlanInput in = input_for(targets[t], samples[s], formats[f]);
        in.depth = depths[d];
        in.height = (flags & 1u) ? 64u : 7u;
        in.effective_mipmap_levels = levels[l];
        in.effective_mipmapped = (flags & 2u) != 0u && levels[l] > 1u;
        in.tex_mipmapped = (flags & 4u) != 0u;
        in.is_render_target = (flags & 8u) != 0u;
        in.dirty_texture_data = (flags & 16u) != 0u;
        in.first_uploadable_level = (flags & 1u) ? 0u : 2u;
        in.format_is_depth_or_stencil = formats[f] >= 250u;
        in.conservative_cpu_cache = (int)(flags & 1u);
        Reference ref;
        MGLTexturePlan plan;
        reference_plan(&in, &ref);
        mglTexturePlanBuild(&in, &plan);
        cases++;
        if (plan.status != MGL_TEXTURE_PLAN_OK || !ref.access_ok ||
            !same_desc(&plan.desc, &ref.desc) ||
            plan.upload_texture_type != ref.tex_type ||
            plan.pixel_format != ref.pixel_format ||
            plan.effective_mipmap_levels != ref.effective_levels ||
            plan.upload_level_count != ref.upload_levels ||
            plan.storage_mipmapped != ref.storage_mipmapped ||
            plan.is_array != ref.is_array ||
            plan.texture_1d_backed_by_2d != ref.b1d ||
            plan.texture_1d_array_backed_by_2d_array != ref.b1da ||
            plan.ms_emulated_as_array != ref.ms ||
            plan.agx_converted != ref.agx ||
            !!plan.has_uploadable_cpu_data != !!ref.has_data ||
            !!plan.needs_cpu_upload != !!ref.needs_upload ||
            plan.clear_tex_mipmapped != ref.clear_tex_mipmapped) {
            if (agree) {
                printf("  first disagreement: target=0x%x format=%u samples=%u "
                       "levels=%u depth=%llu flags=%u\n",
                       targets[t], formats[f], samples[s], levels[l],
                       (unsigned long long)depths[d], flags);
            }
            agree = 0;
        }
    }
    printf("  swept %d inputs\n", cases);
    CHECK(agree, "the plan matches the retired inline sequence on every swept input");
}

int main(void)
{
    printf("texture creation plan harness (audit P4)\n");
    test_basic_shapes();
    test_multisample();
    test_format_usage_and_storage();
    test_sweep();
    printf("test_texture_plan: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
