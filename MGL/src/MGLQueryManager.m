#import "MGLQueryManager.h"

@implementation MGLQueryManager {
    id<MTLBuffer> _visibilityResultBuffer;
    BOOL _sampleQueryActive;
    uint64_t _timerQueryBeginGPU;
}

- (BOOL)beginSampleQueryWithDevice:(id<MTLDevice>)device
{
    if (!_visibilityResultBuffer) {
        _visibilityResultBuffer = [device newBufferWithLength:8
                                                      options:MTLResourceStorageModeShared];
        if (!_visibilityResultBuffer) {
            return NO;
        }
        _visibilityResultBuffer.label = @"MGL Visibility Result";
    }

    memset(_visibilityResultBuffer.contents, 0, _visibilityResultBuffer.length);
    _sampleQueryActive = YES;
    return YES;
}

- (void)endSampleQuery
{
    _sampleQueryActive = NO;
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
    return *(const uint64_t *)_visibilityResultBuffer.contents;
}

- (void)configureRenderPassDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    if (!_sampleQueryActive || !descriptor || !_visibilityResultBuffer) {
        return;
    }
    descriptor.visibilityResultBuffer = _visibilityResultBuffer;
    memset(_visibilityResultBuffer.contents, 0, _visibilityResultBuffer.length);
}

- (void)configureRenderEncoder:(id<MTLRenderCommandEncoder>)renderEncoder
{
    if (_sampleQueryActive && renderEncoder) {
        [renderEncoder setVisibilityResultMode:MTLVisibilityResultModeBoolean offset:0];
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
