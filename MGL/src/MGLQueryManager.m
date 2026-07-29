#import "MGLQueryManager.h"

/* One 8-byte visibility slot per render pass participating in a query.
 * Metal accumulates the visibility counter across draws only within one
 * encoder; at pass end it OVERWRITES the value at the slot's offset, so a
 * query spanning multiple passes must give each pass its own slot and sum
 * them at readback. */
enum { kMGLVisibilitySlots = 256 };

@implementation MGLQueryManager {
    id<MTLBuffer> _visibilityResultBuffer;
    BOOL _sampleQueryActive;
    BOOL _sampleQueryCounting;
    NSUInteger _sampleQuerySlot;
    uint64_t _timerQueryBeginGPU;
}

- (BOOL)beginSampleQueryWithDevice:(id<MTLDevice>)device counting:(BOOL)counting
{
    if (!_visibilityResultBuffer) {
        _visibilityResultBuffer =
            [device newBufferWithLength:kMGLVisibilitySlots * sizeof(uint64_t)
                                options:MTLResourceStorageModeShared];
        if (!_visibilityResultBuffer) {
            return NO;
        }
        _visibilityResultBuffer.label = @"MGL Visibility Result";
    }

    memset(_visibilityResultBuffer.contents, 0, _visibilityResultBuffer.length);
    _sampleQueryActive = YES;
    _sampleQueryCounting = counting;
    _sampleQuerySlot = 0;
    return YES;
}

- (void)endSampleQuery
{
    /* _sampleQuerySlot is intentionally kept: sampleQueryResult reads it
     * to know how many slots to sum.  It is reset at the next begin. */
    _sampleQueryActive = NO;
}

- (BOOL)isSampleQueryActive
{
    return _sampleQueryActive;
}

- (BOOL)hasSampleQueryResultBuffer
{
    return _visibilityResultBuffer != nil;
}

- (uint64_t)sampleQueryResult
{
    if (!_visibilityResultBuffer) {
        return 0;
    }
    const uint64_t *slots = (const uint64_t *)_visibilityResultBuffer.contents;
    uint64_t sum = 0;
    NSUInteger used = _sampleQuerySlot < kMGLVisibilitySlots ? _sampleQuerySlot
                                                             : kMGLVisibilitySlots;
    for (NSUInteger i = 0; i < used; i++) {
        sum += slots[i];
    }
    return sum;
}

- (void)configureRenderPassDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    if (!descriptor || !_visibilityResultBuffer) {
        return;
    }
    /* Always attach the visibility result buffer (when it exists) so that
     * mtlBeginSampleQuery: can enable visibility counting on an existing
     * encoder via setVisibilityResultMode: without ending the encoder.
     * The GPU only writes to it when setVisibilityResultMode is enabled
     * (default disabled).  The buffer is zeroed once at query begin; passes
     * write distinct slots, so no per-pass clear is needed (a clear here
     * would wipe results already written by earlier passes of the same
     * query). */
    descriptor.visibilityResultBuffer = _visibilityResultBuffer;
}

- (void)configureRenderEncoder:(id<MTLRenderCommandEncoder>)renderEncoder
{
    if (_sampleQueryActive && renderEncoder) {
        NSUInteger slot = _sampleQuerySlot;
        if (slot >= kMGLVisibilitySlots) {
            slot = kMGLVisibilitySlots - 1;  /* degrade: reuse last slot */
        } else {
            _sampleQuerySlot++;
        }
        [renderEncoder setVisibilityResultMode:(_sampleQueryCounting
                                                    ? MTLVisibilityResultModeCounting
                                                    : MTLVisibilityResultModeBoolean)
                                        offset:slot * sizeof(uint64_t)];
    }
}

- (void)beginTimerQueryWithDevice:(id<MTLDevice>)device
{
    _timerQueryBeginGPU = [self gpuTimestampWithDevice:device];
}

- (uint64_t)endTimerQueryWithDevice:(id<MTLDevice>)device
{
    uint64_t endGPU = [self gpuTimestampWithDevice:device];
    return endGPU >= _timerQueryBeginGPU ? endGPU - _timerQueryBeginGPU : 0;
}

- (uint64_t)gpuTimestampWithDevice:(id<MTLDevice>)device
{
    if (!device) {
        return 0;
    }
    uint64_t cpuTime = 0;
    uint64_t gpuTime = 0;
    [device sampleTimestamps:&cpuTime gpuTimestamp:&gpuTime];
    return gpuTime;
}

- (void)shutdown
{
    _visibilityResultBuffer = nil;
    _sampleQueryActive = NO;
    _timerQueryBeginGPU = 0;
}

@end
