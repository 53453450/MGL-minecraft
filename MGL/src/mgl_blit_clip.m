/*
 * mgl_blit_clip.m
 * MGL
 *
 * Implementation of the Blit Axis Clipping Subsystem.
 * See mgl_blit_clip.h for the API contract.
 */

#import "mgl_blit_clip.h"

#include <math.h>

BOOL mglClipBlitAxisToDestination(MGLBlitAxis *axis, double dstLimit)
{
    if (!axis || axis->dst0 == axis->dst1 || axis->src0 == axis->src1 || dstLimit <= 0.0) {
        return NO;
    }

    double dstMin = fmin(axis->dst0, axis->dst1);
    double dstMax = fmax(axis->dst0, axis->dst1);
    double clippedMin = fmax(dstMin, 0.0);
    double clippedMax = fmin(dstMax, dstLimit);
    if (clippedMax <= clippedMin) {
        return NO;
    }

    double dstSpan = axis->dst1 - axis->dst0;
    double srcSpan = axis->src1 - axis->src0;
    double tMin = (clippedMin - axis->dst0) / dstSpan;
    double tMax = (clippedMax - axis->dst0) / dstSpan;
    double srcAtMin = axis->src0 + tMin * srcSpan;
    double srcAtMax = axis->src0 + tMax * srcSpan;

    if (axis->dst1 >= axis->dst0) {
        axis->dst0 = clippedMin;
        axis->dst1 = clippedMax;
        axis->src0 = srcAtMin;
        axis->src1 = srcAtMax;
    } else {
        axis->dst0 = clippedMax;
        axis->dst1 = clippedMin;
        axis->src0 = srcAtMax;
        axis->src1 = srcAtMin;
    }

    return YES;
}

BOOL mglClipBlitAxisToSource(MGLBlitAxis *axis, double srcLimit)
{
    if (!axis || axis->dst0 == axis->dst1 || axis->src0 == axis->src1 || srcLimit <= 0.0) {
        return NO;
    }

    double srcMin = fmin(axis->src0, axis->src1);
    double srcMax = fmax(axis->src0, axis->src1);
    double clippedMin = fmax(srcMin, 0.0);
    double clippedMax = fmin(srcMax, srcLimit);
    if (clippedMax <= clippedMin) {
        return NO;
    }

    double srcSpan = axis->src1 - axis->src0;
    double dstSpan = axis->dst1 - axis->dst0;
    double tMin = (clippedMin - axis->src0) / srcSpan;
    double tMax = (clippedMax - axis->src0) / srcSpan;
    double dstAtMin = axis->dst0 + tMin * dstSpan;
    double dstAtMax = axis->dst0 + tMax * dstSpan;

    if (axis->src1 >= axis->src0) {
        axis->src0 = clippedMin;
        axis->src1 = clippedMax;
        axis->dst0 = dstAtMin;
        axis->dst1 = dstAtMax;
    } else {
        axis->src0 = clippedMax;
        axis->src1 = clippedMin;
        axis->dst0 = dstAtMax;
        axis->dst1 = dstAtMin;
    }

    return YES;
}

BOOL mglClipBlitAxis(MGLBlitAxis *axis, double srcLimit, double dstLimit)
{
    return mglClipBlitAxisToDestination(axis, dstLimit) &&
           mglClipBlitAxisToSource(axis, srcLimit);
}
