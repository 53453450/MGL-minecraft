#ifndef MGLQueryManager_h
#define MGLQueryManager_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>

NS_ASSUME_NONNULL_BEGIN

@interface MGLQueryManager : NSObject

- (BOOL)beginSampleQueryWithDevice:(id<MTLDevice>)device counting:(BOOL)counting;
- (void)endSampleQuery;
- (BOOL)isSampleQueryActive;
- (BOOL)hasSampleQueryResultBuffer;
- (uint64_t)sampleQueryResult;
- (void)configureRenderPassDescriptor:(MTLRenderPassDescriptor *)descriptor;
- (void)configureRenderEncoder:(id<MTLRenderCommandEncoder>)renderEncoder;

- (void)beginTimerQueryWithDevice:(id<MTLDevice>)device;
- (uint64_t)endTimerQueryWithDevice:(id<MTLDevice>)device;
- (uint64_t)gpuTimestampWithDevice:(id<MTLDevice>)device;

- (void)shutdown;

@end

NS_ASSUME_NONNULL_END

#endif /* MGLQueryManager_h */
