/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Buffer slot layout harness (audit P4).
 *
 * mglBufferSlotLayoutBuild() is the one computation of where a stage's
 * buffers sit in the Metal argument table; reflection and the AIR backend
 * both call it.  reference_layout() is the arithmetic reflection carried
 * inline before (the backend's was meant to match it); the sweep requires the
 * plan to agree with it for every stage and every small resource count.
 *
 * Build: see `make test-buffer-layout`.
 */

#include "mgl_buffer_layout.h"

#include <stdio.h>
#include <string.h>

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

/* The retired inline layout (mglAirReflectModuleImpl). */
static void reference_layout(const MGLBufferSlotCounts *in, uint32_t *ssbo,
                             uint32_t *ubo, uint32_t *ac, uint32_t *plain_base)
{
    const int is_vs = in->stage == MGL_STAGE_VERTEX;
    const uint32_t user_buffer_base =
        in->stage == MGL_STAGE_TESS_EVALUATION ? 1u : 0u;
    const uint32_t has_plain = in->has_plain_pack ? 1u : 0u;
    *ssbo = user_buffer_base +
            (is_vs ? (has_plain + in->attr_slots)
                   : ((in->stage == MGL_STAGE_COMPUTE ||
                       in->stage == MGL_STAGE_TESS_CONTROL ||
                       in->stage == MGL_STAGE_TESS_EVALUATION ||
                       in->stage == MGL_STAGE_GEOMETRY ||
                       in->stage == MGL_STAGE_FRAGMENT)
                          ? has_plain
                          : 0u));
    *ubo = *ssbo + in->ssbo_slots;
    *ac = *ubo + in->ubo_slots;
    *plain_base = user_buffer_base;
}

static MGLBufferSlotCounts counts(int stage)
{
    MGLBufferSlotCounts c;
    memset(&c, 0, sizeof(c));
    c.stage = stage;
    c.atomic_counter_limit = 84u;
    return c;
}

static void test_sweep(void)
{
    static const int stages[] = {MGL_STAGE_VERTEX, MGL_STAGE_FRAGMENT,
                                 MGL_STAGE_COMPUTE, MGL_STAGE_TESS_CONTROL,
                                 MGL_STAGE_TESS_EVALUATION, MGL_STAGE_GEOMETRY};
    int agree = 1;
    int cases = 0;
    for (unsigned s = 0; s < sizeof(stages) / sizeof(stages[0]); s++)
    for (uint32_t attrs = 0; attrs < 4; attrs++)
    for (uint32_t plain = 0; plain < 2; plain++)
    for (uint32_t ssbo = 0; ssbo < 4; ssbo++)
    for (uint32_t ubo = 0; ubo < 4; ubo++)
    for (uint32_t ac = 0; ac < 3; ac++) {
        MGLBufferSlotCounts c = counts(stages[s]);
        c.attr_slots = attrs;
        c.has_plain_pack = (int)plain;
        c.ssbo_slots = ssbo;
        c.ubo_slots = ubo;
        c.atomic_counter_slots = ac;
        MGLBufferSlotLayout l;
        uint32_t want_ssbo, want_ubo, want_ac, want_plain_base;
        reference_layout(&c, &want_ssbo, &want_ubo, &want_ac, &want_plain_base);
        cases++;
        if (mglBufferSlotLayoutBuild(&c, &l) != MGL_BUFFER_LAYOUT_OK ||
            l.ssbo_base != want_ssbo || l.ubo_base != want_ubo ||
            l.atomic_counter_base != want_ac ||
            l.plain_base != want_plain_base ||
            l.end != want_ac + ac || l.plain_slots != plain) {
            agree = 0;
        }
    }
    printf("  swept %d layouts\n", cases);
    CHECK(agree, "the layout matches the retired reflection arithmetic");
}

static void test_shapes(void)
{
    MGLBufferSlotLayout l;
    MGLBufferSlotCounts c = counts(MGL_STAGE_VERTEX);
    c.attr_slots = 3u;
    c.has_plain_pack = 1;
    c.ssbo_slots = 2u;
    c.ubo_slots = 1u;
    c.atomic_counter_slots = 1u;
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_OK &&
              l.plain_base == 0u && l.ssbo_base == 4u && l.ubo_base == 6u &&
              l.atomic_counter_base == 7u && l.end == 8u,
          "vertex: pack, three attribute slots, then SSBOs, UBOs, counters");

    c = counts(MGL_STAGE_FRAGMENT);
    c.attr_slots = 5u; /* only a vertex stage has attributes */
    c.has_plain_pack = 1;
    c.ssbo_slots = 1u;
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_OK &&
              l.ssbo_base == 1u && l.ubo_base == 2u,
          "fragment: the pack, then SSBOs; attribute counts are ignored");

    c = counts(MGL_STAGE_TESS_EVALUATION);
    c.has_plain_pack = 1;
    c.ubo_slots = 1u;
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_OK &&
              l.plain_base == 1u && l.ssbo_base == 2u && l.ubo_base == 2u,
          "tessellation evaluation leaves slot 0 to the contract buffer");

    c = counts(MGL_STAGE_COMPUTE);
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_OK &&
              l.plain_slots == 0u && l.ssbo_base == 0u && l.end == 0u,
          "a stage with no resources has an empty layout");

    c = counts(99);
    c.has_plain_pack = 1;
    c.ssbo_slots = 1u;
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_OK &&
              l.plain_slots == 0u && l.ssbo_base == 0u,
          "an unknown stage reserves no pack");
}

static void test_limits(void)
{
    MGLBufferSlotLayout l;
    MGLBufferSlotCounts c = counts(MGL_STAGE_COMPUTE);
    c.ssbo_slots = 80u;
    c.atomic_counter_slots = 4u;
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_OK && l.end == 84u,
          "atomic counters may end exactly at the limit");
    c.atomic_counter_slots = 5u;
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_ATOMIC_EXCEEDS &&
              l.end == 0u,
          "atomic counters past the limit are refused and the output is cleared");
    c.atomic_counter_slots = 0u;
    c.ssbo_slots = 200u;
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_OK,
          "the limit only binds when there are atomic counters");

    c = counts(MGL_STAGE_COMPUTE);
    c.ssbo_slots = 0xffffffffu;
    c.ubo_slots = 1u;
    CHECK(mglBufferSlotLayoutBuild(&c, &l) == MGL_BUFFER_LAYOUT_OVERFLOW,
          "a slot number past uint32 is an overflow");

    CHECK(mglBufferSlotLayoutBuild(NULL, &l) == MGL_BUFFER_LAYOUT_BAD_ARGS &&
              mglBufferSlotLayoutBuild(&c, NULL) == MGL_BUFFER_LAYOUT_BAD_ARGS,
          "NULL arguments are refused");
}

int main(void)
{
    printf("buffer slot layout harness (audit P4)\n");
    test_sweep();
    test_shapes();
    test_limits();
    printf("test_buffer_layout: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
