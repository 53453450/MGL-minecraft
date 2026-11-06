#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only

import contextlib
import importlib.util
import io
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"architecture_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchitectureGateTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write(self, relative, text=""):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def prepare_layering(self, lint):
        lint.ROOT = self.root
        lint.SRC = self.root / "MGL" / "src"
        lint.INC = self.root / "MGL" / "include"
        lint.BASELINE = self.root / "scripts" / "layering_baseline.txt"
        for name in lint.L0_FILES + lint.L3_FILES:
            self.write(f"MGL/src/L4_context/{name}")
        for name in lint.TYPE_HEADERS:
            self.write(f"MGL/include/{name}")

    def test_layering_scans_nested_sources(self):
        lint = load_script("check_layering.py")
        self.prepare_layering(lint)
        self.write("MGL/src/L4_context/state.c",
                   "void release_helper(void) { mglSafeReleaseMetalObj(0); }\n")
        hits = lint.collect()
        self.assertIn(("C1", "MGL/src/L4_context/state.c"), hits)

    def test_layering_missing_required_source_fails(self):
        lint = load_script("check_layering.py")
        self.prepare_layering(lint)
        (self.root / "MGL/src/L4_context/state.c").unlink()
        with self.assertRaises(SystemExit):
            lint.collect()

    def test_layering_move_preserves_violation_count(self):
        lint = load_script("check_layering.py")
        self.prepare_layering(lint)
        original = self.write("MGL/src/mgl_render_pass_manager_ops.c",
                              "MGLRendererStateAreas *areas;\n")
        before = sum(len(found) for found in lint.collect().values())
        nested = self.root / "MGL/src/L2_encode/mgl_render_pass_manager_ops.c"
        nested.parent.mkdir(parents=True, exist_ok=True)
        original.rename(nested)
        after = sum(len(found) for found in lint.collect().values())
        self.assertEqual(before, after)
        self.assertGreater(after, 0)

    def test_duplicate_scan_is_recursive(self):
        duplicates = load_script("check_c_duplicates.py")
        duplicates.ROOT = self.root
        self.write("MGL/src/a.c", "void mglDuplicate(void) { }\n")
        self.write("MGL/src/L1_backend/b.c", "void mglDuplicate(void) { }\n")
        locations = duplicates.definitions()["mglDuplicate"]
        self.assertEqual({str(path) for path, _ in locations}, {
            "MGL/src/a.c", "MGL/src/L1_backend/b.c",
        })

    def test_split_key_builder_is_checked_not_skipped(self):
        invariants = load_script("state_machine_invariants.py")
        lines = [
            "void mglComputeStateKey(MGLStateKey *out) {",
            "    out->program_name = 1;",
            "}",
        ]
        missing, note = invariants.check_key_populated(lines, ["program_name"])
        self.assertEqual(missing, [], note)

    def test_replay_cold_buffer_access_is_rejected(self):
        invariants = load_script("state_machine_invariants.py")
        invariants.ROOT = self.root
        self.write("MGL/include/mgl_types_buffer.h",
                   "#define MGL_SNAPSHOT_COLD_BUFFER_BASE_TYPES(_X_) \\\n"
                   "    _X_(_ARRAY_BUFFER) \\\n"
                   "    _X_(_PIXEL_PACK_BUFFER)\n")
        self.write("MGL/src/L2_encode/mgl_batch_test.c",
                   "void replay(GLMState *s) {\n"
                   "    s->buffer_base[_UNIFORM_BUFFER].buffers[0] = 0;\n"
                   "    s->buffer_base[_ARRAY_BUFFER].buffers[0] = 0;\n"
                   "}\n")
        findings = invariants.check_replay_cold_buffer_reads(self.root)
        self.assertEqual(len(findings), 1)
        self.assertIn("_ARRAY_BUFFER", findings[0])

    def test_batch_arena_header_is_a_public_boundary(self):
        invariants = load_script("state_machine_invariants.py")
        self.write("MGL/include/mgl_batch_public.h")
        self.write("MGL/include/mgl_batch_arena.h",
                   "void *mglBatchArenaAlloc(MGLBatchArena *arena, size_t size);\n")
        self.write("MGL/src/mgl_extension_hooks.c",
                   "void *allocate(void *arena) {\n"
                   "    return mglBatchArenaAlloc(arena, 16u);\n"
                   "}\n")
        findings, note = invariants.check_batch_boundary(self.root)
        self.assertEqual(findings, [], note)

    def test_root_source_files_are_rejected(self):
        gates = load_script("arch_layer_gates.py")
        gates.ROOT = str(self.root)
        gates.SRC = str(self.root / "MGL" / "src")
        gates.INC = str(self.root / "MGL" / "include")
        self.write("MGL/src/mgl_root_boundary.h")
        self.write("MGL/src/mgl_stray.c")
        hits = gates.collect()
        self.assertIn(("Lyr-Root-Sources", "MGL/src"), hits)
        self.assertEqual(1, len(hits[("Lyr-Root-Sources", "MGL/src")]))

    def test_missing_split_key_builder_fails(self):
        invariants = load_script("state_machine_invariants.py")
        invariants.ROOT = self.root
        self.write("MGL/src/L2_encode/draw_command.c")
        with self.assertRaisesRegex(SystemExit, "mgl_draw_state_key.c"):
            with contextlib.redirect_stdout(io.StringIO()):
                invariants.main()


if __name__ == "__main__":
    unittest.main()
