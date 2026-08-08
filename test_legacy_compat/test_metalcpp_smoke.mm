// test_metalcpp_smoke.mm — Phase 0 验收：mglRenderCppInit 桥接现有 id<MTLDevice>
// 拿到非空 MTL::Device*（void* 形式）无崩溃；shutdown 幂等；可重建。
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>

#include "mgl_render_cpp.h"

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device (VM?)\n");
            return 2; // 无 GPU 环境，跳过（非失败）
        }

        int rc = mglRenderCppInit((__bridge void*)device);
        if (rc != 0) {
            fprintf(stderr, "FAIL: mglRenderCppInit rc=%d\n", rc);
            return 1;
        }
        void* dev = mglRenderCppGetDevice();
        if (!dev) {
            fprintf(stderr, "FAIL: device null after init\n");
            return 1;
        }
        printf("SMOKE_OK device=%p\n", dev);

        // 幂等：重复 init 不应崩溃/重建
        if (mglRenderCppInit((__bridge void*)device) != 0) {
            fprintf(stderr, "FAIL: idempotent init\n");
            return 1;
        }

        mglRenderCppShutdown();
        if (mglRenderCppGetDevice() != NULL) {
            fprintf(stderr, "FAIL: shutdown did not clear device\n");
            return 1;
        }
        mglRenderCppShutdown(); // 二次 shutdown 幂等

        // 重建
        if (mglRenderCppInit((__bridge void*)device) != 0) {
            fprintf(stderr, "FAIL: reinit\n");
            return 1;
        }
        mglRenderCppShutdown();

        printf("SMOKE_DONE\n");
    }
    return 0;
}
