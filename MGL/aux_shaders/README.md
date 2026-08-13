# Auxiliary shader assets (P3)

The renderer never compiles MSL from source at runtime.  Every auxiliary
helper shader (scaled blit, MSAA integer resolve, scissored clear, safe
fallback) is precompiled here with the Apple SDK `metal`/`metallib` toolchain
into a metallib, then embedded into `MGL/src/mgl_aux_assets.c` as a read-only
byte table (see `MGL/include/mgl_aux_assets.h`).

Regenerate the embedded asset table with `make` (the `build/aux/*.metallib`
rule compiles every `*.metal` with the current SDK and the
`gen_aux_assets.py` step rewrites the table).  The committed
`mgl_aux_assets.c/.h` keep clean clones buildable without the Metal toolchain;
rebuild them only when a `.metal` or `MANIFEST` entry changes.

Requirements for P3: only the Apple SDK (Xcode command-line tools) and Python 3.
No `external/glslang` or `external/SPIRV-*` trees are involved.

- `MANIFEST` — asset table (`asset_name|source|comma-separated entry names`).
- `*.metal` — the precompiled shader sources (build-time inputs only).
