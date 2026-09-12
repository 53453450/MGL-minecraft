/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Stage reference-query harness (O7.4.4).
 *
 * Behavioural oracle for mglFrontendStageReferencesName() /
 * mglFrontendStageReferencesMember(), the TU-backed replacement for the
 * source-text scanners (mgl_program_stage_source_body /
 * mgl_program_source_name_is_referenced / mgl_program_qualified_member_referenced)
 * that used to answer the GL per-stage "is this resource referenced?" queries in
 * mgl_gl_extensions.c.
 *
 * The corpus pins the intended semantics:
 *   - only function bodies count (a declaration, comment or preprocessor line
 *     is not a use), and helpers defined before main() do count;
 *   - a member is matched on its access path, component by component, so
 *     "colors[0]" matches "colors[0].rgb" but never "colors[1]", a dynamic
 *     index is a wildcard, and the leaf of an unrelated object's member no
 *     longer fakes a reference;
 *   - a qualified query ("<instance>.<member>") must line up with the instance,
 *     including an array-of-blocks instance ("blk[1].tint").
 *
 * Build: see `make test-reference-query`.
 */

#include "mgl_frontend_session.h"
#include "mgl_glsl_parser.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"
#include "mgl_shader_abi.h" /* MGL_STAGE_* */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run;
static int tests_passed;

#define CHECK(cond, label)                                       \
    do {                                                         \
        tests_run++;                                             \
        if (cond) {                                              \
            tests_passed++;                                      \
            printf("  [PASS] %s\n", (label));                    \
        } else {                                                 \
            printf("  [FAIL] %s\n", (label));                    \
        }                                                        \
    } while (0)

/* Parse + sema exactly like the production path, then hand the TU to the
 * reference queries.  Returns NULL on a front-end failure (a broken fixture
 * must fail loudly, not silently report "not referenced"). */
static MGLTranslationUnit *analyze(const char *name, const char *src)
{
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    if (!tu || tu->error) {
        fprintf(stderr, "FATAL: %s: parse failed (%s)\n", name,
                tu && tu->error ? tu->error : "no TU");
        exit(2);
    }
    MGLIRModule mod;
    MGLSemaError *errors = NULL;
    uint32_t error_count = 0;
    memset(&mod, 0, sizeof(mod));
    mglGLSLSemanticCheck(tu, MGL_STAGE_FRAGMENT, &mod, &errors, &error_count);
    if (error_count > 0) {
        fprintf(stderr, "FATAL: %s: sema rejected the fixture (%u errors)\n",
                name, error_count);
        for (uint32_t i = 0; i < error_count; i++)
            fprintf(stderr, "  [%u] line %u: %s\n", i, errors[i].line,
                    errors[i].message ? errors[i].message : "(no message)");
        exit(2);
    }
    mglIRModuleDestroy(&mod);
    return tu;
}

static void expect_name(const char *label, const char *src, const char *name,
                        int expected)
{
    MGLTranslationUnit *tu = analyze(label, src);
    const int got = mglFrontendStageReferencesName(tu, name) ? 1 : 0;
    tests_run++;
    if (got == expected) {
        tests_passed++;
        printf("  [PASS] %s\n", label);
    } else {
        printf("  [FAIL] %s: reference(\"%s\") = %d, expected %d\n", label, name,
               got, expected);
    }
    mglGLSLTranslationUnitDestroy(tu);
}

static void expect_member(const char *label, const char *src,
                          const char *instance, const char *member, int expected)
{
    MGLTranslationUnit *tu = analyze(label, src);
    const int got =
        mglFrontendStageReferencesMember(tu, instance, member) ? 1 : 0;
    tests_run++;
    if (got == expected) {
        tests_passed++;
        printf("  [PASS] %s\n", label);
    } else {
        printf("  [FAIL] %s: reference(\"%s\", \"%s\") = %d, expected %d\n",
               label, instance, member, got, expected);
    }
    mglGLSLTranslationUnitDestroy(tu);
}

/* A block instance with a nested struct member, plus an unrelated object that
 * shares the nested member's leaf name (the old scanner's false positive). */
static const char *kNested =
    "#version 450\n"
    "struct Inner { vec4 d[2]; };\n"
    "struct Outer { Inner a[2]; };\n"
    "layout(std140, binding = 0) uniform Blk { Outer o; vec4 tint; } blk;\n"
    "layout(std140, binding = 1) uniform Other { Inner a[2]; } other;\n"
    "layout(location = 0) out vec4 color;\n"
    "void main() { color = blk.o.a[0].d[0] + other.a[1].d[0]; }\n";

int main(void)
{
    printf("reference-query harness (O7.4.4)\n");

    /* ---- bodies only ---- */
    expect_name("read in main counts",
                "#version 450\n"
                "uniform vec4 tint;\n"
                "layout(location = 0) out vec4 color;\n"
                "void main() { color = tint; }\n",
                "tint", 1);
    expect_name("declaration alone is not a use",
                "#version 450\n"
                "uniform vec4 tint;\n"
                "layout(location = 0) out vec4 color;\n"
                "void main() { color = vec4(1.0); }\n",
                "tint", 0);
    expect_name("comment is not a use",
                "#version 450\n"
                "uniform vec4 tint;\n"
                "layout(location = 0) out vec4 color;\n"
                "void main() { /* tint stays unused */ color = vec4(1.0); }\n",
                "tint", 0);
    expect_name("helper defined before main counts",
                "#version 450\n"
                "uniform vec4 tint;\n"
                "vec4 helper() { return tint; }\n"
                "layout(location = 0) out vec4 color;\n"
                "void main() { color = helper(); }\n",
                "tint", 1);
    expect_name("write counts as a use",
                "#version 450\n"
                "layout(std140) buffer B { vec4 slot; };\n"
                "void main() { slot = vec4(1.0); }\n",
                "slot", 1);
    expect_name("whole-name boundary is respected",
                "#version 450\n"
                "uniform vec4 tint;\n"
                "uniform vec4 tint_extra;\n"
                "layout(location = 0) out vec4 color;\n"
                "void main() { color = tint_extra; }\n",
                "tint", 0);
    expect_name("longer run after the name still counts",
                "#version 450\n"
                "uniform vec4 tint;\n"
                "layout(location = 0) out vec4 color;\n"
                "void main() { color = tint.xyzw; }\n",
                "tint", 1);

    /* ---- array members: index fidelity and prefix matching ---- */
    static const char *kArray =
        "#version 450\n"
        "layout(std140, binding = 0) uniform Blk { vec4 colors[4]; } blk;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = blk.colors[0]; }\n";
    expect_member("array member element 0 is a use", kArray, "blk", "colors[0]",
                  1);
    expect_member("a sibling element is not a use", kArray, "blk", "colors[1]",
                  0);
    expect_member("index-less member name covers every element", kArray, "blk",
                  "colors", 1);
    expect_member("another instance does not match", kArray, "other", "colors[0]",
                  0);

    static const char *kDeeper =
        "#version 450\n"
        "layout(std140, binding = 0) uniform Blk { vec4 colors[4]; } blk;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(blk.colors[0].xyz, 1.0); }\n";
    expect_member("deeper selection still references the element", kDeeper,
                  "blk", "colors[0]", 1);
    expect_member("deeper selection does not reference a sibling", kDeeper,
                  "blk", "colors[3]", 0);

    static const char *kDynamic =
        "#version 450\n"
        "layout(std140, binding = 0) uniform Blk { vec4 colors[4]; } blk;\n"
        "layout(location = 0) flat in int idx;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = blk.colors[idx]; }\n";
    expect_member("a dynamic index covers a literal query", kDynamic, "blk",
                  "colors[2]", 1);

    static const char *kInstanceArray =
        "#version 450\n"
        "layout(std140, binding = 0) uniform Blk { vec4 tint; } blk[2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = blk[1].tint; }\n";
    expect_member("subscripted block instance is matched", kInstanceArray, "blk",
                  "tint", 1);
    expect_member("subscripted block instance, wrong member", kInstanceArray,
                  "blk", "other", 0);

    /* ---- nested paths: the query must line up with its own object ---- */
    expect_member("nested path matches its own object", kNested, "blk",
                  "o.a[0].d[0]", 1);
    expect_member("nested path does not match a sibling index", kNested, "blk",
                  "o.a[1].d[0]", 0);
    expect_member("nested path does not match another object's member", kNested,
                  "blk", "o.a[0].d[1]", 0);
    expect_member("intermediate member is referenced", kNested, "blk", "o", 1);
    expect_member("block member is referenced", kNested, "blk", "tint", 0);

    /* Two blocks with the same member name: the qualified query must follow its
     * own instance (the retired leaf scan answered "referenced" for both). */
    static const char *kTwoBlocks =
        "#version 450\n"
        "layout(std140, binding = 0) uniform Blk { vec4 tint; } blk;\n"
        "layout(std140, binding = 1) uniform Other { vec4 tint; } other;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = other.tint; }\n";
    expect_member("another instance's member does not fake it", kTwoBlocks, "blk",
                  "tint", 0);
    expect_member("the accessed instance's member is a use", kTwoBlocks, "other",
                  "tint", 1);
    expect_name("unqualified query still sees the matching use", kTwoBlocks,
                "tint", 1);

    /* ---- degenerate inputs ---- */
    {
        MGLTranslationUnit *tu = analyze("degenerate", kNested);
        CHECK(mglFrontendStageReferencesName(tu, NULL) == 0,
              "NULL name is not a reference");
        CHECK(mglFrontendStageReferencesName(tu, "") == 0,
              "empty name is not a reference");
        CHECK(mglFrontendStageReferencesName(NULL, "tint") == 0,
              "NULL TU is not a reference");
        CHECK(mglFrontendStageReferencesMember(tu, NULL, "tint") == 0,
              "NULL instance falls back to an unqualified member match");
        CHECK(mglFrontendStageReferencesMember(tu, "blk", NULL) == 0,
              "NULL member is not a reference");
        CHECK(mglFrontendStageReferencesMember(NULL, "blk", "tint") == 0,
              "NULL TU is not a reference (member)");
        mglGLSLTranslationUnitDestroy(tu);
    }

    printf("test_reference_query: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
