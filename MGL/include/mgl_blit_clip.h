/*
 * mgl_blit_clip.h
 * MGL
 *
 * Blit Axis Clipping Subsystem: pure helpers for clipping one axis of a
 * glBlitFramebuffer-style blit operation against source/destination
 * texture bounds.  Used by mglBlitFramebuffer to handle negative
 * coordinates, out-of-bounds destinations, and flipped axes.
 *
 * All functions here are pure (no self/ivar/global dependency beyond the
 * math library) and may be called from any translation unit.
 */

#ifndef MGL_BLIT_CLIP_H
#define MGL_BLIT_CLIP_H

#include <objc/objc.h>  /* BOOL */

#ifdef __cplusplus
extern "C" {
#endif

/* One axis (X or Y) of a blit operation.  src0/src1 and dst0/dst1 may be
 * in either order (flipped blits are valid in GL). */
typedef struct MGLBlitAxis_t {
    double src0;
    double src1;
    double dst0;
    double dst1;
} MGLBlitAxis;

/* Clips the blit axis against the destination bounds [0, dstLimit].
 * Adjusts src0/src1 proportionally to match the clipped dst0/dst1.
 * Returns NO if the axis is degenerate or the clipped region is empty. */
BOOL mglClipBlitAxisToDestination(MGLBlitAxis *axis, double dstLimit);

/* Clips the blit axis against the source bounds [0, srcLimit].
 * Adjusts dst0/dst1 proportionally to match the clipped src0/src1.
 * Returns NO if the axis is degenerate or the clipped region is empty. */
BOOL mglClipBlitAxisToSource(MGLBlitAxis *axis, double srcLimit);

/* Clips the blit axis against both source and destination bounds.
 * Equivalent to calling mglClipBlitAxisToDestination then
 * mglClipBlitAxisToSource.  Returns NO if either step rejects. */
BOOL mglClipBlitAxis(MGLBlitAxis *axis, double srcLimit, double dstLimit);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BLIT_CLIP_H */
