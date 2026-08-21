/* gs_xfb_scatter.metal
 *
 * Pass 2 of the ordered GL4 GS transform-feedback path (mgl_air_gs_abi.h
 * section 5b).  One thread per pass-1 work item: reads the per-(work-item,
 * buffer) visible byte counts and their exclusive prefix offsets, applies
 * whole-primitive cross-buffer truncation (a primitive lands only if it
 * fits in every buffer its stream feeds - atomic because offsets are
 * ordered), repacks each captured varying from the pass-1 stage-out
 * record to its link-time component offset, and writes the slot-31 XFB
 * records in emission order.
 *
 * Record regions inside a work item's stage-out run (ABI section 2/5b):
 *   [0, expanded_offset)                     stream 0, ascending,
 *   (expanded_offset .. run end) descending  streams > 0, each record
 *                                            stamped with its stream id
 *                                            at byte 48 of the header.
 * Stream 0 is identified by region; streams > 0 by the stamp.
 *
 * This kernel is precompiled (P3 aux asset chain); it contains no runtime
 * source compilation.  All layout constants are mirrored by static asserts
 * in mgl_air_gs_abi.h.
 *
 * Buffer slots: Metal kernel buffer indices must stay in [0, 30], so the
 * scatter kernel does NOT reuse the pass-1 slot-31 XFB index directly; the
 * renderer binds the slot-31 stream to this kernel's buffer(4) and the
 * written counters to buffer(5).  The stage-out records stay at buffer(3)
 * to match the pass-1 layout; params/vis/offsets take the low slots. */

#include <metal_stdlib>
using namespace metal;

#define MGL_GS_XFB_MAX_STREAMS 4u
#define MGL_GS_XFB_MAX_FIELDS 30u
#define MGL_GS_XFB_STREAM_STAMP_OFFSET 48u
#define MGL_GS_XFB_NO_STREAM 0xFFFFFFFFu

struct MGLGSXFBBufferMeta {
    uint stride;          /* record bytes for this buffer; 0 = unused  */
    uint capacity_bytes;  /* store capacity from the bound offset      */
    uint capture_base;    /* byte offset of this buffer's slot-31 segment */
    uint written;         /* written-byte counter (kernel updated)     */
};

struct MGLGSXFBFieldDesc {
    uint buffer_index;    /* destination XFB buffer 0..3 */
    uint src_offset;      /* byte offset in the pass-1 stage-out record */
    uint dst_offset;      /* byte offset in the buffer's XFB record     */
    uint byte_count;      /* captured bytes (component_count * 4)       */
};

struct MGLGSXFBScatterParams {
    uint work_item_count;
    uint stage_out_stride;
    uint records_per_primitive;
    uint vertices_per_primitive;
    uint field_count;
    uint buffer_count;
    uint expanded_offset_records;
    uint _pad;
    uint buffer_stream[MGL_GS_XFB_MAX_STREAMS]; /* feeding stream, NO_STREAM */
    MGLGSXFBBufferMeta buffers[MGL_GS_XFB_MAX_STREAMS];
    MGLGSXFBFieldDesc  fields[MGL_GS_XFB_MAX_FIELDS];
};

kernel void mgl_gs_xfb_scatter(
    uint gid [[thread_position_in_grid]],
    constant MGLGSXFBScatterParams &p [[buffer(0)]],
    device const uint *vis [[buffer(1)]],
    device const uint *offsets [[buffer(2)]],
    device uchar *stage_out [[buffer(3)]],
    device uchar *xfb_out [[buffer(4)]],
    device uint *written [[buffer(5)]]) {
    if (gid >= p.work_item_count) return;

    const uint recStride = p.stage_out_stride;
    const device uchar *runBase =
        stage_out + (ulong)gid * (ulong)p.records_per_primitive * recStride +
        (ulong)p.expanded_offset_records * recStride;
    const uint expanded = p.records_per_primitive - p.expanded_offset_records;

    /* Per-buffer running write cursor within this work item, starting at the
     * ordered prefix offset for this work item, and the visible byte budget
     * truncated to the remaining buffer capacity (whole-primitive atomic
     * truncation: only fully-fitting primitives are written). */
    uint base[MGL_GS_XFB_MAX_STREAMS];
    uint consumed[MGL_GS_XFB_MAX_STREAMS];
    uint budget[MGL_GS_XFB_MAX_STREAMS];
    for (uint b = 0; b < MGL_GS_XFB_MAX_STREAMS; b++) {
        uint idx = gid * MGL_GS_XFB_MAX_STREAMS + b;
        uint visBytes = (b < p.buffer_count) ? vis[idx] : 0u;
        uint prefix = (b < p.buffer_count) ? offsets[idx] : 0u;
        uint cap = p.buffers[b].capacity_bytes;
        uint avail = (prefix < cap) ? (cap - prefix) : 0u;
        base[b] = prefix;
        consumed[b] = 0u;
        budget[b] = min(visBytes, avail);
    }

    /* Copy one record's fields for the buffers fed by stream `s`.
     * Caller has already verified the whole primitive fits. */
    auto copyFields = [&](const device uchar *srcRec, uint s) {
        for (uint f = 0; f < p.field_count; f++) {
            MGLGSXFBFieldDesc fd = p.fields[f];
            uint b = fd.buffer_index;
            if (b >= p.buffer_count || p.buffer_stream[b] != s) continue;
            device uchar *dst =
                xfb_out + (ulong)p.buffers[b].capture_base +
                (ulong)(base[b] + consumed[b]) + fd.dst_offset;
            const device uchar *src = srcRec + fd.src_offset;
            for (uint k = 0; k < fd.byte_count; k++) dst[k] = src[k];
        }
    };

    /* Stream 0: ascending records, vpp records per primitive.  Ordered
     * truncation: the first non-fitting primitive stops the stream because
     * later primitives need the same bytes. */
    const uint stride0 = p.buffers[0].stride;
    const uint vpp = p.vertices_per_primitive;
    if (stride0 != 0u && vpp != 0u) {
        uint prims = (vis[gid * MGL_GS_XFB_MAX_STREAMS] / stride0) / vpp;
        for (uint prim = 0; prim < prims; prim++) {
            bool fits = true;
            for (uint b = 0; b < p.buffer_count && fits; b++) {
                if (p.buffer_stream[b] != 0u) continue;
                if (consumed[b] + vpp * p.buffers[b].stride > budget[b])
                    fits = false;
            }
            if (!fits) break;
            for (uint v = 0; v < vpp; v++) {
                copyFields(runBase + (ulong)(prim * vpp + v) * recStride, 0u);
                for (uint b = 0; b < p.buffer_count; b++) {
                    if (p.buffer_stream[b] == 0u)
                        consumed[b] += p.buffers[b].stride;
                }
            }
        }
    }

    /* Streams > 0 (points-only): descending records stamped with the stream
     * id, one record per primitive.  A non-fitting primitive is dropped;
     * other streams keep capturing. */
    uint totalDown = 0u;
    for (uint b = 0; b < p.buffer_count; b++) {
        if (p.buffer_stream[b] > 0u && p.buffer_stream[b] != MGL_GS_XFB_NO_STREAM &&
            p.buffers[b].stride != 0u) {
            totalDown += vis[gid * MGL_GS_XFB_MAX_STREAMS + b] /
                         p.buffers[b].stride;
        }
    }
    for (uint j = 0; j < totalDown; j++) {
        const device uchar *srcRec =
            runBase + (ulong)(expanded - 1u - j) * recStride;
        uint stamp = *(const device uint *)(srcRec +
                                            MGL_GS_XFB_STREAM_STAMP_OFFSET);
        bool fits = false;
        for (uint b = 0; b < p.buffer_count; b++) {
            if (p.buffer_stream[b] != stamp) continue;
            if (consumed[b] + p.buffers[b].stride <= budget[b]) fits = true;
            else { fits = false; break; }
        }
        if (!fits) continue;
        copyFields(srcRec, stamp);
        for (uint b = 0; b < p.buffer_count; b++) {
            if (p.buffer_stream[b] == stamp)
                consumed[b] += p.buffers[b].stride;
        }
    }

    /* Publish written bytes per buffer.  Each work item's buffer regions are
     * disjoint, so a plain store to this work item's accumulator suffices;
     * the renderer reduces across work items after the dispatch. */
    for (uint b = 0; b < MGL_GS_XFB_MAX_STREAMS; b++) {
        if (b >= p.buffer_count) continue;
        written[gid * MGL_GS_XFB_MAX_STREAMS + b] = consumed[b];
    }
}
