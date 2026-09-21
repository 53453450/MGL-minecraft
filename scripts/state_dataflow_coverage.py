#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
#
# state_dataflow_coverage.py — G4 coverage gate (STATE_DATAFLOW_TODO §7.2)
#
# Three checks on the deferred-draw state key / GLMState dataflow:
#
#   CHECK 1  Every MGLStateKey field is written by mglComputeStateKey.
#   CHECK 2  MGLStateKey has no implicit interior padding (memcmp contract).
#   CHECK 3  Top-level GLMState fields that are neither written into the key
#            nor referenced by the four hash builders are listed as candidates
#            for human review; only findings outside the recorded baseline fail.
#
# CHECK 1/2 reuse the parsers in state_machine_invariants.py (same brace-
# anchoring and LP64 layout rules). Exit 0 when no finding is outside the
# baseline; exit 1 otherwise.
#
# Usage: python3 scripts/state_dataflow_coverage.py [--verbose]

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state_machine_invariants import (  # noqa: E402
    ROOT,
    check_key_layout,
    check_key_populated,
    field_names,
    fn_span,
    read,
    struct_body,
)

VERBOSE = "--verbose" in sys.argv

# Hash / key builders that define "covered by draw identity".
_HASH_BUILDER_PATTERNS = (
    r"^void mglComputeStateKey",
    r"^static uint64_t mglComputeTextureHash",
    r"^static uint64_t mglComputeRenderStateHash",
    r"^static uint64_t mglComputeVertexArrayStateHash",
    r"^static uint64_t mglComputeUniformBufferBindingHash",
    r"^static uint64_t mglComputeDrawBufferBindingHash",
)

# CHECK 3 baseline: fields currently outside key/hash builders.  Every entry
# was reviewed as intentional (STATE_DATAFLOW_TODO §6 false positives, plus
# clear/query/cache/table containers that are not draw-identity inputs).
# A NEW top-level GLMState field that is also uncovered fails the gate.
CHECK3_BASELINE = frozenset({
    "_hash_cache_padding",
    "active_sampled_texture_unit_mask",
    "active_sampled_texture_unit_mask_valid",
    "active_texture",
    "buffer_table",
    "buffers",
    "clear_bitmask",
    "color_clear_value",
    "compute_buffer_map_list",
    "conditional_render_active",
    "conditional_render_mode",
    "conditional_render_query",
    "conditional_render_skip",
    "default_clear_color",
    "default_draw_buffer",
    "default_draw_buffer_count",
    "default_draw_buffers",
    "default_fbo_clear_bitmask",
    "default_read_buffer",
    "default_vao_element_array_buffer",
    "depth_range_array",
    "dirty_bits",
    "error",
    "error_count",
    "error_head",
    "error_queue",
    "fragment_buffer_map_list",
    "framebuffer_table",
    "hints",
    "last_sampled_2d_textures",
    "max_color_attachments",
    "max_vertex_attribs",
    "pack",
    "program_table",
    "proxy_texture_query",
    "query_depth_known",
    "query_depth_value",
    "read_buffer",
    "readbuffer",
    "recent_sampled_2d_textures",
    "renderbuffer",
    "renderbuffer_table",
    "sampler_table",
    "scissor_box_array",
    "shader_table",
    "shaders",
    "sync_name",
    "sync_table",
    "tex",
    "texture_table",
    "transform_feedback",
    "transform_feedback_table",
    "unpack",
    "vao_table",
    "vertex_buffer_map_list",
    "viewport_array",
    "viewport_array_set",
})


def glm_state_fields(state_src):
    """Top-level GLMState field names; tolerate multi-dimensional arrays."""
    from state_machine_invariants import strip_comments

    body = strip_comments(struct_body(state_src, "GLMState") or "")
    out = []
    for stmt in body.split(";"):
        # A #define inside the struct can share a ';' chunk with the next
        # field (e.g. MGL_ERROR_QUEUE_SIZE + GLenum error). Drop preprocessor
        # lines, then parse the remaining declarator.
        lines = []
        for line in stmt.split("\n"):
            if line.lstrip().startswith("#"):
                continue
            lines.append(line)
        stmt = "\n".join(lines).strip()
        if not stmt:
            continue
        m = re.search(
            r"([A-Za-z_][A-Za-z_0-9]*)\s*(?:\[[^\]]*\])*\s*$",
            stmt,
        )
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def covered_state_fields(draw_lines):
    """GLMState member names referenced from key/hash builders."""
    covered = set()
    for pat in _HASH_BUILDER_PATTERNS:
        span = fn_span(draw_lines, pat)
        if not span:
            continue
        body = "\n".join(draw_lines[span[0] : span[1] + 1])
        for m in re.finditer(
            r"(?:active_state|STATE)\s*(?:->|\()\s*([A-Za-z_][A-Za-z0-9_]*)",
            body,
        ):
            covered.add(m.group(1))
    return covered


def identity_key_fields(identity_src):
    """MGLStateKey column of MGL_STATE_IDENTITY_ROWS (skip `_`)."""
    from state_machine_invariants import strip_comments

    body = strip_comments(identity_src)
    m = re.search(
        r"#define\s+MGL_STATE_IDENTITY_ROWS\s*\(\s*_X\s*\)\s*((?:.*\\\n)*.*)",
        body,
    )
    region = m.group(1) if m else ""
    # Truncate at the next non-continuation # directive / end of macro.
    cut = re.search(r"\n#ifndef|\n#endif|\n#if |\n#ifdef|\n#ifn", region)
    if cut:
        region = region[: cut.start()]
    keys = []
    for row in re.finditer(
        r"_X\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,",
        region,
    ):
        key = row.group(2)
        if key != "_":
            keys.append(key)
    return sorted(set(keys))


def main():
    state_h = os.path.join(ROOT, "MGL/include/mgl_types_state.h")
    key_h = os.path.join(ROOT, "MGL/include/draw_command.h")
    identity_h = os.path.join(ROOT, "MGL/include/mgl_state_identity_table.h")
    draw_c = os.path.join(ROOT, "MGL/src/draw_command.c")

    state_src = read(state_h)
    key_src = read(key_h)
    identity_src = read(identity_h)
    draw_lines = read(draw_c).split("\n")
    key_fields = field_names(struct_body(key_src, "MGLStateKey") or "")
    table_keys = identity_key_fields(identity_src)

    print("=" * 78)
    print("G4 state dataflow coverage (CHECK 1–3 + T9-1 identity table)")
    print("=" * 78)
    print(f"MGLStateKey fields: {len(key_fields)}")
    print(f"identity-table key columns: {len(table_keys)}")
    print()

    findings = []

    # ---- CHECK 1 ----------------------------------------------------------
    missing, note = check_key_populated(draw_lines, key_fields)
    print("=" * 78)
    print("CHECK 1  every MGLStateKey field written by mglComputeStateKey")
    print("=" * 78)
    if missing is None:
        print(f"  SKIP  {note}")
        findings.append(f"CHECK1: {note}")
    elif missing:
        for f in missing:
            print(f"  FAIL  {f} is never written")
        findings.extend(f"CHECK1: unwritten key field {f}" for f in missing)
    else:
        print(f"  OK    all {len(key_fields)} fields written  ({note})")
    print()

    # ---- CHECK 1b (T9-1): identity table is the authority for key fields ----
    print("=" * 78)
    print("CHECK 1b  identity table ↔ MGLStateKey bijection (T9-1)")
    print("=" * 78)
    if not table_keys:
        print("  FAIL  could not parse MGL_STATE_IDENTITY_ROWS key columns")
        findings.append("CHECK1b: identity table unparsed")
    else:
        only_struct = sorted(set(key_fields) - set(table_keys))
        only_table = sorted(set(table_keys) - set(key_fields))
        if only_struct or only_table:
            for f in only_struct:
                print(f"  FAIL  MGLStateKey has '{f}' missing from identity table")
            for f in only_table:
                print(f"  FAIL  identity table lists '{f}' not in MGLStateKey")
            findings.extend(f"CHECK1b: struct-only {f}" for f in only_struct)
            findings.extend(f"CHECK1b: table-only {f}" for f in only_table)
        else:
            print(f"  OK    {len(table_keys)} key columns match MGLStateKey exactly")
        # Table keys must also be written (same as CHECK 1, keyed off the table).
        missing_tab, _ = check_key_populated(draw_lines, table_keys)
        if missing_tab:
            for f in missing_tab:
                print(f"  FAIL  identity key '{f}' never written by mglComputeStateKey")
            findings.extend(f"CHECK1b: unwritten identity key {f}" for f in missing_tab)
        elif table_keys and not only_struct and not only_table:
            print("  OK    every identity key column is written by mglComputeStateKey")
    print()

    # ---- CHECK 2 ----------------------------------------------------------
    res, note = check_key_layout(key_src)
    print("=" * 78)
    print("CHECK 2  MGLStateKey has no implicit padding inside memcmp")
    print("=" * 78)
    if res is None:
        print(f"  SKIP  {note}")
        findings.append(f"CHECK2: {note}")
    else:
        holes, total, tail, sized = res
        if holes:
            for name, off, pad in holes:
                print(f"  FAIL  {pad} padding byte(s) before '{name}' at offset {off}")
            findings.extend(
                f"CHECK2: padding before {n} at {o}" for n, o, _ in holes
            )
        else:
            print("  OK    no interior padding hole")
        if tail:
            print(f"  WARN  {tail} tail padding byte(s) inside sizeof()")
        print(f"        ({note}; fields sum to {sized})")
    print()

    # ---- CHECK 3 ----------------------------------------------------------
    fields = glm_state_fields(state_src)
    covered = covered_state_fields(draw_lines)
    uncovered = [f for f in fields if f not in covered]
    new = sorted(f for f in uncovered if f not in CHECK3_BASELINE)
    stale = sorted(f for f in CHECK3_BASELINE if f in covered)
    # Baseline entries that disappeared from the struct (renamed/removed).
    missing_baseline = sorted(f for f in CHECK3_BASELINE if f not in fields)

    print("=" * 78)
    print("CHECK 3  GLMState fields outside key / hash builders (baseline gate)")
    print("=" * 78)
    print(f"  covered={len(covered)}  uncovered={len(uncovered)}  "
          f"baseline={len(CHECK3_BASELINE)}")
    if VERBOSE:
        for f in uncovered:
            tag = "baseline" if f in CHECK3_BASELINE else "NEW"
            print(f"  CANDIDATE [{tag}] {f}")
    if new:
        for f in new:
            print(f"  FAIL  new uncovered field '{f}' (not in §6 baseline)")
        findings.extend(f"CHECK3: new uncovered field {f}" for f in new)
    else:
        print("  OK    no uncovered fields outside the recorded baseline")
    if stale and VERBOSE:
        for f in stale:
            print(f"  NOTE  baseline '{f}' is now covered (safe to drop)")
    if missing_baseline:
        for f in missing_baseline:
            print(f"  WARN  baseline '{f}' no longer exists in GLMState")
    print()

    print("=" * 78)
    if findings:
        print(f"RESULT: FAIL ({len(findings)} finding(s))")
        for f in findings:
            print(f"  ! {f}")
        print("=" * 78)
        return 1
    print("RESULT: PASS")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
