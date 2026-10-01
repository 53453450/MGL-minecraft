# spec_parser (archived)

`spec_parser.c` once generated `gl_core.c`, `gl_es.c`, `glm_dispatch.{h,c}` and
`mgl.h` from the Khronos `gl.xml` registry (output to `/tmp/`). It depends on
`ezxml`, which is not in the repository, and no Makefile target or script
invokes it.

The generated files under `MGL/` are now maintained by hand. Edit them
directly; re-running this tool would discard those edits.
