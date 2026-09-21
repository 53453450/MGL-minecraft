#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
#
# state_machine_invariants.py — static invariant audit for the C state machine.
#
# The state machine is the `GLMState` + command-buffer + batch cluster:
#
#   MGL/include/mgl_types_state.h      GLMState, dirty bits, hot-copy regions
#   MGL/include/mgl_dirty_bits.h       authoritative DIRTY_* masks (M0)
#   MGL/include/draw_command.h         MGLStateKey, MGLDrawCommand, MGLDrawBatch
#   MGL/src/state.c                    GL API state setters
#   MGL/src/draw_command.c             recorder, key builder, flush entry
#   MGL/src/mgl_batch_*.c|.cpp         batch restore / issue / encode / replay
#   MGL/src/mgl_renderer_core_state.c  the dual-proxy (active_state) invariant
#
# It is ~12.7k lines but carries the correctness of everything else
# (docs/C_LAYER_ARCHITECTURE_REVIEW.md §5.1, docs/STATE_DATAFLOW_TODO.md).
# This script turns the invariants that were established by hand during the
# 2026-09-13 audit into a repeatable check, so a regression fails loudly in CI
# instead of silently corrupting batch merges at runtime.
#
# It is a STATIC, heuristic audit. It reports candidates for human review; it
# does not prove absence of bugs. Exit code is 0 when no finding falls outside
# the recorded baseline, 1 otherwise.
#
# Usage: python3 scripts/state_machine_invariants.py [--verbose]

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = "--verbose" in sys.argv


def rel(p):
    return os.path.relpath(p, ROOT)


def read(p):
    with open(p, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def strip_comments(text):
    """Blank out comments while PRESERVING line structure.

    Deleting a multi-line /* ... */ outright shifts every following line
    number, which made this script report line numbers that did not exist in
    the file. Replace comment characters with spaces instead.
    """
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))

    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    return re.sub(r"//[^\n]*", blank, text)


def struct_body(text, name):
    """Body of the struct whose closing brace is `} <name>;`.

    Anchored on the closing brace so structs embedding other structs
    (GLMState embeds GLMParams/GLMCaps/HashTable) extract correctly.
    """
    m = re.search(r"\}\s*" + re.escape(name) + r"\s*;", text)
    if not m:
        return None
    opens = [i for i, ch in enumerate(text[: m.start()]) if ch == "{"]
    return text[opens[-1] + 1 : m.start()] if opens else None


def field_names(body):
    out = []
    for stmt in strip_comments(body).split(";"):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("#"):
            continue
        m = re.search(r"([A-Za-z_][A-Za-z_0-9]*)\s*(\[[^\]]*\])?$", stmt)
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def fn_span(lines, pattern):
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            for j in range(i + 1, len(lines)):
                if lines[j] == "}":
                    return i, j
    return None


# --------------------------------------------------------------------------
# I1  MGLStateKey: every field must be written by mglComputeStateKey
# --------------------------------------------------------------------------
def check_key_populated(draw_lines, key_fields):
    """A key field that is never written is always zero and cannot
    discriminate state, so two different states would compare equal."""
    span = fn_span(draw_lines, r"^void mglComputeStateKey")
    if not span:
        return None, "mglComputeStateKey not found"
    body = "\n".join(draw_lines[span[0] : span[1] + 1])
    missing = [
        f
        for f in key_fields
        if not re.search(r"out->" + re.escape(f) + r"\s*(\[[^\]]*\])?\s*=", body)
    ]
    return missing, f"mglComputeStateKey spans {span[0] + 1}..{span[1] + 1}"


# --------------------------------------------------------------------------
# I2  MGLStateKey: no implicit padding inside the memcmp'd struct
# --------------------------------------------------------------------------
LP64 = {
    "uint8_t": (1, 1), "int8_t": (1, 1), "char": (1, 1),
    "uint16_t": (2, 2), "int16_t": (2, 2), "short": (2, 2),
    "uint32_t": (4, 4), "int32_t": (4, 4), "int": (4, 4), "unsigned": (4, 4),
    "float": (4, 4), "GLuint": (4, 4), "GLint": (4, 4), "GLenum": (4, 4),
    "GLsizei": (4, 4), "GLboolean": (1, 1), "GLbitfield": (4, 4),
    "uint64_t": (8, 8), "int64_t": (8, 8), "GLuint64": (8, 8),
    "double": (8, 8), "size_t": (8, 8),
}


def check_key_layout(key_src, key_name="MGLStateKey"):
    body = struct_body(key_src, key_name)
    if not body:
        return None, "could not parse MGLStateKey"
    off = 0
    maxalign = 1
    holes = []
    sized = 0
    for stmt in strip_comments(body).split(";"):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("#"):
            continue
        m = re.match(r"(.+?)\s+([A-Za-z_][A-Za-z_0-9]*)\s*(\[([^\]]*)\])?$", stmt)
        if not m:
            continue
        typ, name, dim = m.group(1).strip(), m.group(2), m.group(4)
        if typ not in LP64:
            continue
        sz, al = LP64[typ]
        if dim:
            try:
                sz *= int(dim)
            except ValueError:
                pass
        maxalign = max(maxalign, al)
        pad = (-off) % al
        if pad:
            holes.append((name, off, pad))
            off += pad
        off += sz
        sized += sz
    tail = (-off) % maxalign
    return (holes, off, tail, sized), f"{off + tail} bytes computed"


# --------------------------------------------------------------------------
# I3  hot-copy regions must not silently include the HashTable block
# --------------------------------------------------------------------------
def check_hot_copy_regions(state_src):
    """mglCopyHotStateFields identifies skipped ranges by offsetof arithmetic.
    The gap [sync_table, shaders) must be exactly the 11 embedded HashTables
    and [buffer_base, pack) exactly the cold buffer_base array."""
    findings = []
    if "offsetof(GLMState, shaders) - offsetof(GLMState, sync_table)" not in state_src:
        findings.append("hot-copy region-2 _Static_assert missing")
    if "offsetof(GLMState, pack) - offsetof(GLMState, buffer_base)" not in state_src:
        findings.append("hot-copy region-5 _Static_assert missing")
    if "kMGLSnapshotHotBufferBaseCount + kMGLSnapshotColdBufferBaseCount" not in state_src:
        findings.append("hot/cold buffer_base coverage _Static_assert missing")
    return findings


# --------------------------------------------------------------------------
# I4  DIRTY_* single source: no hand-copied numeric dirty-bit enums
# --------------------------------------------------------------------------
def check_dirty_bits_single_source(files):
    """M0 moved the authoritative masks to mgl_dirty_bits.h and deleted the
    hand copy in mgl_batch_restore.c. Any new `= 1u << <number>` dirty enum
    outside the authoritative header reintroduces the drift hazard."""
    findings = []
    pat = re.compile(r"(MGL_BATCH_DIRTY_\w+)\s*=\s*1u\s*<<\s*\d+")
    for p in files:
        src = strip_comments(read(p))
        for m in pat.finditer(src):
            line = src[: m.start()].count("\n") + 1
            findings.append(f"{rel(p)}:{line} hand-coded {m.group(1)} = 1u << N")
    return findings


# --------------------------------------------------------------------------
# I5  batch_count: every mutation site must be inside the recorder
# --------------------------------------------------------------------------
def check_batch_count_mutation(files):
    """cb->batch_count is the state machine's fundamental cursor: it gates the
    flush early-out and the batch array walk. Mutating it outside
    draw_command.c bypasses the key/snapshot/retain bookkeeping."""
    findings = []
    # Only the command-buffer cursor: `cb->batch_count`, `c->cb->batch_count`,
    # `ctx->draw_command_buffer.batch_count`. Deliberately NOT any struct that
    # merely ends in `batch_count` field of an unrelated record, and comments
    # are stripped first so prose (e.g. "the former flushDrawBufferLocked")
    # cannot match.
    # A MUTATION, not a read: require ++ / -- / += / -= / =<expr>, and for a
    # bare `=` require that it is not `==` (the earlier pattern flagged
    # `if (cb->batch_count == 0) return 0;` as a mutation).
    pat = re.compile(
        r"\b(?:cb|budget_cb|c->cb|c->ctx->draw_command_buffer"
        r"|glm_ctx->draw_command_buffer)\s*->\s*batch_count\s*"
        r"(\+\+|--|\+=|-=|=)(?!=)")
    for p in files:
        if os.path.basename(p) == "draw_command.c":
            continue
        src = strip_comments(read(p))
        for m in pat.finditer(src):
            line = src[: m.start()].count("\n") + 1
            findings.append(f"{rel(p)}:{line} mutates cb->batch_count outside the recorder")
    return findings


# --------------------------------------------------------------------------
# I6  dual proxy: replay must switch active_state through the helpers
# --------------------------------------------------------------------------
def check_dual_proxy(files):
    """mglCoreActivateReplayState / mglCoreRestoreLiveActiveState own the
    active_state transition. Assigning active_state directly anywhere else
    desynchronises MGL_STATE() from the renderer's cached pointer, which is
    the mechanism behind the replay-side error-queue loss
    (docs/STATE_DATAFLOW_TODO.md P0-1)."""
    findings = []
    allowed = {"mgl_renderer_core_state.c"}
    # Two semantically different writes share the same syntax:
    #
    #   A. REBIND TO LIVE  -> `...active_state = &ctx->state` (or `= NULL`).
    #      This is what mglCoreActivateReplayState / mglCoreRestoreLiveActiveState
    #      own. Doing it by hand anywhere else desynchronises MGL_STATE() from
    #      the renderer's cached pointer -- the P0-1 hazard. FLAGGED.
    #
    #   B. RE-SYNC TO CURRENT -> `core->activeState = ctx->active_state`.
    #      The context is already on the replay workspace; this repairs the
    #      renderer's cached copy against an existing pointer. It is NOT what
    #      the helpers do (mglCoreRestoreLiveActiveState also rebinds to live
    #      and NULLs the cache), so calling a helper here would be wrong.
    #      Recorded, not flagged.
    rebind = re.compile(r"\b(?:active_state|activeState)\s*=\s*(?:&\s*\w+->state|NULL)\b")
    # Both spellings occur in the tree: `core->activeState = ctx->active_state`
    # and the nested `areas.core->activeState = c->ctx->active_state`.
    # Keep the two sides independent: any lvalue chain ending in ->activeState
    # assigned from any chain ending in ->active_state. A single `=` (not `==`,
    # not `!=`) is required.
    resync = re.compile(
        r"->activeState\s*=(?!=)\s*[^;\n]*->active_state\b")
    for p in files:
        if os.path.basename(p) in allowed:
            continue
        src = strip_comments(read(p))
        for m in rebind.finditer(src):
            line = src[: m.start()].count("\n") + 1
            findings.append(
                f"{rel(p)}:{line} REBINDS active_state by hand (use the dual-proxy helper)")
        for m in resync.finditer(src):
            line = src[: m.start()].count("\n") + 1
            findings.append(
                f"{rel(p)}:{line} re-syncs core->activeState to the current "
                f"pointer (legitimate workspace repair, not a rebind)")
    return findings


# --------------------------------------------------------------------------
# I7  snapshot ownership: a shared snapshot must never be freed
# --------------------------------------------------------------------------
def _if_chain_branches(src, start):
    """Yield (condition, body) for an if / else-if / else chain at `start`.

    Needed because a naive "is the token near the free?" window matches across
    mutually exclusive branches: `!batch->arena_managed` and
    `batch->snapshot_shared` are exclusive, so a free inside the former is NOT
    an unguarded shared free.
    """
    i = start
    n = len(src)
    while i < n:
        m = re.compile(r"\s*(?:if\s*\(|else\s+if\s*\(|else\b|\{)").match(src, i)
        if not m:
            return
        if src.startswith("else", i) and not re.match(r"\s*else\s+if", src[i:]):
            cond = "else"
            j = src.index("{", i)
        else:
            if src.startswith("else", i):
                cond_start = src.index("(", i) + 1
            else:
                cond_start = src.index("(", i) + 1
            depth = 1
            k = cond_start
            while k < n and depth:
                if src[k] == "(":
                    depth += 1
                elif src[k] == ")":
                    depth -= 1
                k += 1
            cond = src[cond_start : k - 1].strip()
            j = src.index("{", k - 1)
        depth = 1
        e = j + 1
        while e < n and depth:
            if src[e] == "{":
                depth += 1
            elif src[e] == "}":
                depth -= 1
            e += 1
        yield cond, src[j + 1 : e - 1], j + 1
        # continue with the following else / else-if, if any
        rest = src[e:].lstrip()
        if not rest.startswith("else"):
            return
        i = e + (len(src[e:]) - len(rest))


def check_snapshot_ownership(files):
    """M2 invariants around snapshot_shared.

    (a) A snapshot under a snapshot_shared guard is the DONOR's; freeing it
        double-frees.
    (b) `commands` is NOT a snapshot.  It is the batch's own realloc'd array
        (draw_command.c ~4351) in BOTH modes, so every release path must still
        deal with it.  An earlier version of this check only looked for the
        token `snapshot_shared` near a state_snapshot free and reported OK
        while the shared branch silently leaked `commands` on the non-arena
        path.  (b) is now checked by walking the if/else chain that tests
        snapshot_shared and requiring the taken branch to handle commands.
    """
    findings = []
    for p in files:
        src = strip_comments(read(p))
        for m in re.finditer(
                r"if\s*\(\s*\w+->snapshot_shared\s*\)", src):
            line = src[: m.start()].count("\n") + 1
            shared_taken = None
            for cond, body, _off in _if_chain_branches(src, m.start()):
                if "snapshot_shared" in cond:
                    shared_taken = body
                elif cond == "else" and shared_taken is not None:
                    break
            if shared_taken is None:
                continue
            if "state_snapshot = NULL" not in shared_taken or \
               "vao_snapshot = NULL" not in shared_taken:
                findings.append(
                    f"{rel(p)}:{line} snapshot_shared branch does not null both "
                    "snapshot pointers")
            if "commands" not in shared_taken:
                findings.append(
                    f"{rel(p)}:{line} snapshot_shared branch does not handle "
                    "commands (leaks the batch's own realloc'd command array)")

        # (a) A snapshot free must sit in the innermost enclosing `if` whose
        #     condition excludes BOTH shared and arena ownership.  We locate
        #     that `if` by scanning forward from each candidate and keeping
        #     the last one whose body still contains the free - a forward
        #     scan, because a backward brace walk gets confused by sibling
        #     blocks that closed earlier.
        cases = []
        for m in re.finditer(r"if\s*\(", src):
            d = 1
            k = m.end()
            while k < len(src) and d:
                if src[k] == "(":
                    d += 1
                elif src[k] == ")":
                    d -= 1
                k += 1
            cond = src[m.end() : k - 1]
            j = src.find("{", k)
            if j < 0:
                continue
            d = 1
            e = j + 1
            while e < len(src) and d:
                if src[e] == "{":
                    d += 1
                elif src[e] == "}":
                    d -= 1
                e += 1
            cases.append((m.start(), j, e, cond))

        for m in re.finditer(
                r"free\s*\(\s*(?:\w+->)?(?:state_snapshot|vao_snapshot)\s*\)", src):
            line = src[: m.start()].count("\n") + 1
            enclosing = [c for c in cases if c[1] < m.start() < c[2]]
            if not enclosing:
                findings.append(f"{rel(p)}:{line} frees a snapshot outside any if")
                continue
            _st, _jb, _e, cond = max(enclosing, key=lambda c: c[0])
            excludes_arena = ("!arena_managed" in cond
                              or "arena_managed ==" in cond
                              or "arena_managed)" in cond and "!" in cond)
            excludes_shared = "snapshot_shared" not in cond
            # the free may also be inside a nested `if` (e.g. `if (commands)`);
            # require one of the enclosing conditions to carry the exclusion.
            ok = any(
                ("!arena_managed" in c[3] or "arena_managed ==" in c[3]
                 or ("arena_managed" in c[3] and "!" in c[3]))
                for c in enclosing
            ) and not any("snapshot_shared" in c[3] for c in enclosing
                          if "!" not in c[3])
            if not ok:
                findings.append(
                    f"{rel(p)}:{line} frees a snapshot in a branch that does not "
                    "exclude shared/arena ownership")
    return findings


# --------------------------------------------------------------------------
# I8  batch cluster boundary: outside callers may only use the public header
# --------------------------------------------------------------------------
def check_batch_boundary(root, _unused=None):
    """The batch cluster declares ~157 functions across mgl_batch_*.h.  Only a
    handful are meant for callers outside the cluster, and those are declared in
    mgl/include/mgl_batch_public.h.  Without this check the boundary is
    invisible: a new cross-boundary call looks exactly like an internal one.

    Fails when a TU outside the cluster references a batch symbol that the
    public header does not declare.
    """
    cluster = lambda base: (base.startswith("mgl_batch_")
                            or base == "draw_command.c"
                            or base == "mgl_renderer_core_state.c")

    declared = {}
    for name in sorted(os.listdir(os.path.join(root, "MGL/include"))):
        if not name.startswith("mgl_batch_") or not name.endswith(".h"):
            continue
        src = strip_comments(read(os.path.join(root, "MGL/include", name)))
        for m in re.finditer(
                r"(?m)^[A-Za-z_][\w \*]*?\b(mgl[A-Za-z_0-9]+)\s*\(", src):
            declared.setdefault(m.group(1), set()).add(name)

    public_path = os.path.join(root, "MGL/include/mgl_batch_public.h")
    public = set()
    if os.path.exists(public_path):
        src = strip_comments(read(public_path))
        public = set(re.findall(
            r"(?m)^[A-Za-z_][\w \*]*?\b(mgl[A-Za-z_0-9]+)\s*\(", src))
    else:
        return None, "mgl_batch_public.h missing"

    # Enumerate the tree here rather than trusting a caller-supplied list:
    # main()'s `cluster` holds ONLY cluster files, and passing that in made this
    # check scan nothing at all while still reporting OK.
    src_dir = os.path.join(root, "MGL/src")
    all_files = [os.path.join(src_dir, n) for n in sorted(os.listdir(src_dir))
                 if n.endswith((".c", ".cpp"))]

    findings = []
    for p in all_files:
        if cluster(os.path.basename(p)):
            continue
        src = strip_comments(read(p))
        used = set(re.findall(r"\b(mgl[A-Za-z_0-9]+)\s*\(", src))
        for sym in sorted(used & set(declared)):
            if sym not in public:
                findings.append(
                    f"{rel(p)} calls {sym}, declared in "
                    f"{'/'.join(sorted(declared[sym]))} but not in "
                    "mgl_batch_public.h")
    return findings, f"{len(declared)} declared / {len(public)} public"


def main():
    state_h = os.path.join(ROOT, "MGL/include/mgl_types_state.h")
    dirty_h = os.path.join(ROOT, "MGL/include/mgl_dirty_bits.h")
    key_h = os.path.join(ROOT, "MGL/include/draw_command.h")
    draw_c = os.path.join(ROOT, "MGL/src/draw_command.c")

    state_src = read(state_h)
    key_src = read(key_h)
    draw_lines = read(draw_c).split("\n")

    cluster = [draw_c, state_h, dirty_h, key_h]
    for name in sorted(os.listdir(os.path.join(ROOT, "MGL/src"))):
        if name.startswith("mgl_batch_") or name == "mgl_renderer_core_state.c":
            cluster.append(os.path.join(ROOT, "MGL/src", name))

    key_fields = field_names(struct_body(key_src, "MGLStateKey") or "")

    print("=" * 78)
    print("C state machine invariant audit")
    print("=" * 78)
    print(f"cluster: {len(cluster)} files, "
          f"{sum(len(read(p).splitlines()) for p in cluster)} lines")
    print(f"MGLStateKey fields: {len(key_fields)}")
    print()

    findings = []

    # I1
    missing, note = check_key_populated(draw_lines, key_fields)
    print("=" * 78)
    print("I1  every MGLStateKey field written by mglComputeStateKey")
    print("=" * 78)
    if missing is None:
        print(f"  SKIP  {note}")
    elif missing:
        for f in missing:
            print(f"  FAIL  {f} is never written -> always zero, cannot discriminate")
        findings.extend(f"I1: unwritten key field {f}" for f in missing)
    else:
        print(f"  OK    all {len(key_fields)} fields written  ({note})")
    print()

    # I2
    res, note = check_key_layout(key_src)
    print("=" * 78)
    print("I2  MGLStateKey has no implicit padding inside memcmp")
    print("=" * 78)
    if res is None:
        print(f"  SKIP  {note}")
    else:
        holes, total, tail, sized = res
        if holes:
            for name, off, pad in holes:
                print(f"  FAIL  {pad} padding byte(s) before '{name}' at offset {off}")
            findings.extend(f"I2: padding before {n} at {o}" for n, o, _ in holes)
        else:
            print("  OK    no interior padding hole")
        if tail:
            print(f"  WARN  {tail} tail padding byte(s) inside sizeof()")
        print(f"        ({note}; fields sum to {sized})")
    print()

    # I3
    reg = check_hot_copy_regions(state_src)
    print("=" * 78)
    print("I3  hot-copy region boundaries pinned by _Static_assert")
    print("=" * 78)
    if reg:
        for f in reg:
            print(f"  FAIL  {f}")
        findings.extend(f"I3: {f}" for f in reg)
    else:
        print("  OK    all three region assertions present")
    print()

    # I4
    dirty = check_dirty_bits_single_source(cluster)
    print("=" * 78)
    print("I4  DIRTY_* masks have a single source (no hand-copied enum)")
    print("=" * 78)
    if dirty:
        for f in dirty:
            print(f"  FAIL  {f}")
        findings.extend(f"I4: {f}" for f in dirty)
    else:
        print("  OK    no hand-coded dirty-bit enum outside the authoritative header")
    print()

    # I5
    bc = check_batch_count_mutation(cluster)
    print("=" * 78)
    print("I5  batch_count mutated only by the recorder")
    print("=" * 78)
    if bc:
        for f in bc:
            print(f"  REVIEW {f}")
        findings.extend(f"I5: {f}" for f in bc)
    else:
        print("  OK    no external batch_count mutation")
    print()

    # I6
    dp = check_dual_proxy(cluster)
    print("=" * 78)
    print("I6  active_state switched only through the dual-proxy helpers")
    print("=" * 78)
    if dp:
        for f in dp:
            print(f"  REVIEW {f}")
        findings.extend(f"I6: {f}" for f in dp)
    else:
        print("  OK    no direct active_state assignment outside the helpers")
    print()

    # I7
    so = check_snapshot_ownership(cluster)
    print("=" * 78)
    print("I7  shared snapshots are not freed by the borrower")
    print("=" * 78)
    if so:
        for f in so:
            print(f"  REVIEW {f}")
        findings.extend(f"I7: {f}" for f in so)
    else:
        print("  OK    no unguarded state_snapshot free")
    print()

    # I8
    bfind, bnote = check_batch_boundary(ROOT, cluster)
    print("=" * 78)
    print("I8  batch-cluster boundary (outside callers use mgl_batch_public.h)")
    print("=" * 78)
    if bfind is None:
        print(f"  SKIP  {bnote}")
    elif bfind:
        for f in bfind:
            print(f"  FAIL  {f}")
        findings.extend(f"I8: {f}" for f in bfind)
    else:
        print(f"  OK    no undeclared cross-boundary batch call  ({bnote})")
    print()

    # ---- baseline gate ----------------------------------------------------
    # Findings recorded as of the audit. These are WORK ITEMS, not accepted
    # state; the gate exists so a NEW one fails the build.
    #
    #   I5  recorder-external batch_count writes : none
    #   I6  direct active_state writes           : mgl_renderer_core_state.c only
    #   I7  unguarded snapshot frees             : none
    # Recorded (reviewed, intentional) findings do not fail the gate; anything
    # else is a regression.
    BASELINE_MARKERS = ("re-syncs core->activeState",)
    recorded = [f for f in findings if any(k in f for k in BASELINE_MARKERS)]
    new = [f for f in findings if f not in recorded]

    print("=" * 78)
    if recorded:
        print(f"RECORDED (work items, do not fail the gate): {len(recorded)}")
        for f in recorded:
            print(f"  ~ {f}")
    if new:
        print(f"RESULT: NEW FINDINGS vs baseline: {len(new)}")
        for f in new:
            print(f"  + {f}")
        print("=" * 78)
        return 1
    print("RESULT: no new findings vs recorded baseline")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
