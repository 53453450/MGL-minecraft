// MGLRenderer+Lifecycle.m
// Renderer construction, glm_ctx mtl_funcs binding, proactive texture
// priming, Metal frame capture, and dealloc — extracted from MGLRenderer.m.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Lifecycle_Private.h"
#import "mgl.h"
#import "mgl_metal_bridge.h"
#import "draw_command.h"

static MTLPixelFormat mglMetalLayerPixelFormatForContext(GLMContext drawCtx)
{
    MTLPixelFormat fallback = MTLPixelFormatBGRA8Unorm;
    if (!drawCtx) {
        return fallback;
    }

    MTLPixelFormat requested = (MTLPixelFormat)drawCtx->pixel_format.mtl_pixel_format;
    if (mglMetalLayerPixelFormatIsSupported(requested)) {
        return requested;
    }

    NSLog(@"MGL CAMetalLayer pixelFormat fallback glFormat=0x%x glType=0x%x requestedMtl=%lu fallback=%lu",
          drawCtx->pixel_format.format,
          drawCtx->pixel_format.type,
          (unsigned long)requested,
          (unsigned long)fallback);
    return fallback;
}

@implementation MGLRenderer (Lifecycle)

#pragma mark C interface to context functions

- (void) bindObjFuncsToGLMContext: (GLMContext) glm_ctx
{
    /* mtlObj is CFBridgingRetain +1 (see destroyGLMContext in glm_context.c).
     * Re-binding (e.g. GLFW window renderer replacing the auto-init headless
     * renderer) must release the previous owner or it leaks permanently. */
    void *previousObj = glm_ctx->mtl_funcs.mtlObj;
    if (previousObj != NULL) {
        CFRelease((CFTypeRef)previousObj);
    }

    glm_ctx->mtl_funcs.mtlObj = (void *)CFBridgingRetain(self);

    /* Assignment block generated from MGL_MTL_FUNC_LIST (see
     * mgl_types_metal_funcs.h — single source of truth). */
#define MGL_MTL_FUNC_ASSIGN(field, cname, ret, args) \
    glm_ctx->mtl_funcs.field = cname;
    MGL_MTL_FUNC_LIST(MGL_MTL_FUNC_ASSIGN)
#undef MGL_MTL_FUNC_ASSIGN
}

- (id) initMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
{
    assert (window);
    assert (glm_ctx);
    
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);

    [view setWantsLayer:YES];
    [window setContentView:view];
    
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    
    return self;
}

- (id) createMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
{
    assert (window);
    assert (glm_ctx);
    
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);

    [view setWantsLayer:YES];
    [window setContentView:view];
    
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    
    return renderer;
}


void* CppCreateMGLRendererFromContextAndBindToWindow (void *glm_ctx, void *window)
{
    assert (window);
    assert (glm_ctx);
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);
    NSWindow * w = (__bridge NSWindow *)(window); // just a plain bridge as the autorelease pool will try to release this and crash on exit
    assert (w);
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);
    [view setWantsLayer:YES];
    //assert(w.contentView);
    //[w.contentView addSubview:view];
    [w setContentView:view];
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    // Ownership: the returned pointer is NON-OWNING (borrowed).
    // The renderer's lifetime is tied to glm_ctx->mtl_funcs.mtlObj, which is
    // retained via CFBridgingRetain in bindObjFuncsToGLMContext.
    // The caller must NOT CFRelease/free the returned pointer, and must keep
    // glm_ctx alive while using the returned pointer.
    return  (__bridge void *)(renderer);
}

void* CppCreateMGLRendererHeadless (void *glm_ctx)
{
    assert (glm_ctx);
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    // Create a dummy NSView for headless rendering
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);
    [view setWantsLayer:YES];

    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    // Ownership: the returned pointer is NON-OWNING (borrowed).
    // The renderer's lifetime is tied to glm_ctx->mtl_funcs.mtlObj, which is
    // retained via CFBridgingRetain in bindObjFuncsToGLMContext.
    // The caller must NOT CFRelease/free the returned pointer, and must keep
    // glm_ctx alive while using the returned pointer.
    return  (__bridge void *)(renderer);
}

void* CppCreateMGLRendererAndBindToContext (void *glm_ctx)
{
    // Compatibility export used by reference libMGL.dylib.
    // Falls back to headless binding when no Cocoa window is supplied.
    return CppCreateMGLRendererHeadless(glm_ctx);
}

- (void) createMGLRendererAndBindToContext: (GLMContext) glm_ctx view: (NSView *) view
{
    ctx = glm_ctx;
    _queryManager = [MGLQueryManager new];
    _renderPassManager = [MGLRenderPassManager new];

    /* start the DontCare frame generation at 1 so it never matches a
     * texture's zero-initialized mtl_rt_frame_generation stamp until that
     * texture is actually written this frame. */
    [_renderPassManager setDontCareFrameGeneration:1u];

    // CRITICAL FIX: Initialize thread synchronization locks.
    // _metalStateLock: NSRecursiveLock (reentrant) — required because the
    //   MGLRenderer call graph has indirect re-entry paths through non-target
    //   helper methods.  A non-reentrant lock deadlocked on first frame.
    // _syncListLock: os_unfair_lock (non-reentrant, value type) - protects
    //   only MGLRenderPassManager sync-list access and is acquired after
    //   _metalStateLock when both locks are needed.
    _metalStateLock = [[NSRecursiveLock alloc] init];
    _syncListLock   = OS_UNFAIR_LOCK_INIT;
    NSLog(@"MGL INFO: Metal state lock (NSRecursiveLock) + sync list lock (os_unfair_lock) initialized");

    // Initialize AGX GPU error tracking
    _gpuRecovery.gpuErrorLock = OS_UNFAIR_LOCK_INIT;
    _gpuRecovery.consecutiveGPUErrors = 0;
    _gpuRecovery.lastGPUErrorTime = 0;
    _gpuRecovery.gpuErrorRecoveryMode = NO;
    // Kill-switchable opts: unset = ON, =0/false/no/off = OFF.
    _resourceFallback.mslCacheEnabled = mglEnvFlagEnabledDefaultOn("MGL_MSL_CACHE");
    // Bounded per-Program MSL texture type lookup cache (always on; no env var).
    // Keys include a process-unique Program lifetime ID and link generation.
    _resourceFallback.mslTextureTypeCache = [NSCache new];
    _resourceFallback.mslTextureTypeCache.countLimit = 4096u;
    BOOL psoDedupEnabled = mglEnvFlagEnabledDefaultOn("MGL_PSO_DEDUP");
    BOOL depthStencilCacheEnabled = mglEnvFlagEnabledDefaultOn("MGL_DS_CACHE");
    BOOL binaryArchiveEnabled = mglEnvFlagEnabledDefaultOn("MGL_BINARY_ARCHIVE");
    _pipelineCache = [[MGLPipelineCache alloc]
        initWithPSODedupEnabled:psoDedupEnabled
      depthStencilCacheEnabled:depthStencilCacheEnabled
           binaryArchiveEnabled:binaryArchiveEnabled];
    _bindingSync = [MGLBindingSync new];
    /* Snapshot arena: batch snapshot/commands from bump allocator. */
    _batching.arenaSnapshotEnabled = mglEnvFlagEnabledDefaultOn("MGL_ARENA_SNAPSHOT");
    if (_batching.arenaSnapshotEnabled) {
        if (mglInitBatchArena(&_batching.batchArena, 4u * 1024u * 1024u)) {
            ctx->batch_arena = &_batching.batchArena;
            NSLog(@"MGL INFO: Snapshot arena enabled (initial chunk capacity %zu bytes)",
                  _batching.batchArena.initial_capacity);
        } else {
            _batching.arenaSnapshotEnabled = NO;
            NSLog(@"MGL WARNING: Snapshot arena malloc failed; falling back to per-batch malloc");
        }
    }
    _batching.skipSameKeyRestoreEnabled = mglEnvFlagEnabledDefaultOn("MGL_SKIP_SAME_KEY_RESTORE");
    _batching.dirtyKeyDeltaEnabled = mglEnvFlagEnabledDefaultOn("MGL_DIRTY_KEY_DELTA");
    /* Initialize last-bound render encoder dedup state to a clean slate.
     * _bindingSync.state->lastBoundValid starts NO so the first bind on the first encoder is
     * never incorrectly skipped. */
    [self invalidateLastBoundState];
    NSLog(@"MGL INFO: AGX GPU error tracking initialized");
    NSLog(@"MGL INFO: perf gates pso_dedup=%d ds_cache=%d arena=%d msl_cache=%d "
          "same_key_restore=%d dirty_key_delta=%d (set VAR=0 to disable)",
          _pipelineCache.state->psoDedupEnabled ? 1 : 0,
          _pipelineCache.state->dsCacheEnabled ? 1 : 0,
          _batching.arenaSnapshotEnabled ? 1 : 0,
          _resourceFallback.mslCacheEnabled ? 1 : 0,
          _batching.skipSameKeyRestoreEnabled ? 1 : 0,
          _batching.dirtyKeyDeltaEnabled ? 1 : 0);

    [self bindObjFuncsToGLMContext: glm_ctx];

    // VIRTUALIZED AGX DETECTION: Create Metal device with virtualization safety
    NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating Metal device with virtualization detection");

    // Create the Metal device
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        NSLog(@"MGL ERROR: Metal device not found - this is required for Apple Silicon");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, _metalStateLock, _syncListLock, AGX GPU error tracking
        //        fields (_gpuRecovery.consecutiveGPUErrors/_gpuRecovery.lastGPUErrorTime/
        //        _gpuRecovery.gpuErrorRecoveryMode), _pipeline*Format/_pipelineCache.state->pipelineProgramName,
        //        _pipelineCache.state->pipelineStateCache, and glm_ctx->mtl_funcs (bound via
        //        bindObjFuncsToGLMContext, with mtlObj retained).
        //   NIL: _device, _commandQueue, _view.
        // Continuing is pointless without a Metal device — every subsequent
        // operation depends on it.
        return; // Exit early rather than continuing with nil device
    }

    NSLog(@"MGL INFO: Metal device created: %@", _device);
    _pipelineCache.device = _device;
    [_pipelineCache initializeCompilerIfAvailableUnlessDisabled:
        mglEnvFlagEnabled("MGL_DISABLE_MTL4_COMPILER")];

    /* Initialize AGX Capability Layer (centralized device detection +
     * capability queries + driver bug markers).  Replaces scattered
     * `containsString:@"AGX"` checks and hardcoded constants. */
    MGLCapabilityInit(&_capability, _device);

    // PROPER AGX VIRTUALIZATION DETECTION: Maintain Metal functionality with virtualization compatibility
    BOOL isVirtualized = _capability.isVirtualized;
    NSString *deviceName = [_device name];

    // DETECTION: Check if running in QEMU virtualization but keep Metal enabled
    if (isVirtualized) {
        isVirtualized = YES;
        NSLog(@"MGL INFO: AGX device detected - enabling virtualization compatibility mode: %@", deviceName);
        NSLog(@"MGL INFO: Metal functionality will be maintained with AGX virtualization safety measures");
    }

    // Create command queue with virtualization-safe settings
    MTLCommandQueueDescriptor *queueDescriptor = [[MTLCommandQueueDescriptor alloc] init];
    if (isVirtualized) {
        NSLog(@"MGL INFO: VIRTUALIZED AGX - Enabling virtualization-safe command queue settings");
        queueDescriptor.maxCommandBufferCount =
            MGLCapabilityMaxConcurrentCommandBuffers(&_capability);
    }

    _commandQueue = [_device newCommandQueueWithDescriptor:queueDescriptor];
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Failed to create Metal command queue");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, _metalStateLock, _syncListLock, AGX GPU error tracking
        //        fields, _pipeline*Format/_pipelineCache.state->pipelineProgramName,
        //        _pipelineCache.state->pipelineStateCache, glm_ctx->mtl_funcs (bound, mtlObj
        //        retained), _device, MTL4 compiler (if available), _capability.
        //   NIL: _commandQueue, _view.
        // Continuing is pointless without a command queue — no encoding or
        // submission is possible.
        return;
    }

    NSLog(@"MGL INFO: Metal command queue created successfully");

    /* Load or create Binary Archive for PSO compile acceleration.
     * Gated by MGL_BINARY_ARCHIVE (default ON; =0 disables).
     * The archive is stored in the user's Caches directory and persists
     * compiled PSO binaries across launches, reducing cold-start PSO
     * compile time from ~10s to ~2s on subsequent launches. */
    if (_pipelineCache.state->binaryArchiveEnabled) {
        if (@available(macOS 11.0, *)) {
            [_pipelineCache loadBinaryArchive];
        } else {
            [_pipelineCache disableBinaryArchive];
        }
    }

    _view = view;

    // PROPER FIX: Create Metal layer with AGX-safe settings
    NSLog(@"MGL INFO: PROPER FIX - Creating Metal layer with AGX-safe settings");

    _layer = [[CAMetalLayer alloc] init];
    if (!_layer) {
        NSLog(@"MGL ERROR: Failed to create Metal layer");
        return;
    }

    _layer.device = _device;
    MTLPixelFormat requestedPixelFormat = ctx ? (MTLPixelFormat)ctx->pixel_format.mtl_pixel_format
                                              : MTLPixelFormatInvalid;
    MTLPixelFormat pf = mglMetalLayerPixelFormatForContext(ctx);

    @try {
        _layer.pixelFormat = pf;
    } @catch (NSException *exception) {
        NSLog(@"MGL CAMetalLayer invalid pixelFormat=%lu requested=%lu exception=%@; falling back to BGRA8Unorm",
              (unsigned long)pf,
              (unsigned long)requestedPixelFormat,
              exception);
        pf = MTLPixelFormatBGRA8Unorm;
        _layer.pixelFormat = pf;
    }

    if (ctx && ctx->pixel_format.mtl_pixel_format != (GLuint)pf) {
        NSLog(@"MGL CAMetalLayer sync default framebuffer metal format glFormat=0x%x glType=0x%x oldMtl=%u newMtl=%lu",
              ctx->pixel_format.format,
              ctx->pixel_format.type,
              ctx->pixel_format.mtl_pixel_format,
              (unsigned long)pf);
        ctx->pixel_format.mtl_pixel_format = (GLuint)pf;
    }
    NSLog(@"MGL CAMetalLayer pixelFormat=%lu requested=%lu glFormat=0x%x glType=0x%x",
          (unsigned long)_layer.pixelFormat,
          (unsigned long)requestedPixelFormat,
          ctx ? ctx->pixel_format.format : 0u,
          ctx ? ctx->pixel_format.type : 0u);
    _layer.opaque = YES;
    _layer.framebufferOnly = NO; // enable blitting to main color buffer
    _layer.allowsNextDrawableTimeout = YES; // avoid indefinite nextDrawable stalls
    _layer.magnificationFilter = kCAFilterNearest;
    _layer.presentsWithTransaction = NO;

    // AGX-safe layer attachment
    if ([_view layer]) {
        [[_view layer] addSublayer: _layer];
    } else {
        [_view setLayer: _layer];
    }
    [self mglSyncLayerDrawableSizeFromView:"createRenderer"];

    mglDrawBuffer(glm_ctx, GL_FRONT);

    // Create initial command buffer for AGX safety
    @try {
        [_renderPassManager installNewCommandBufferFromQueue:_commandQueue];
        if (!_renderPassManager.state->currentCommandBuffer) {
            NSLog(@"MGL ERROR: Failed to create initial Metal command buffer");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating initial Metal command buffer: %@", exception);
    }
    
    glm_ctx->mtl_funcs.mtlView = (void *)CFBridgingRetain(view);

    // PROACTIVE TEXTURE CREATION: Create essential textures to break sync loop
    NSLog(@"MGL INFO: PROACTIVE - Creating essential textures to prevent magenta screen");
    [self createProactiveTextures];

    // capture Metal commands in MGL.gputrace
    // necessitates Info.plist in the cwd, see https://stackoverflow.com/a/64172784
    //MTLCaptureDescriptor *descriptor = [self setupCaptureToFile: _device];
    //[self startCapture:descriptor];
}

// PROACTIVE TEXTURE CREATION: Create essential textures during initialization to break sync loop
- (void)createProactiveTextures
{
    NSLog(@"MGL PROACTIVE: Starting essential texture creation");

    @try {
        // Create a simple 2D texture with gradient pattern to prevent magenta screens
        MTLTextureDescriptor *proactiveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                                          width:256
                                                                                                         height:256
                                                                                                      mipmapped:NO];
        proactiveDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        proactiveDesc.storageMode = MTLStorageModeShared;

        id<MTLTexture> proactiveTexture = [_device newTextureWithDescriptor:proactiveDesc];
        if (proactiveTexture) {
            // Create gradient pattern data
            uint32_t *gradientData = calloc(256 * 256, sizeof(uint32_t));
            if (gradientData) {
                // Create blue-green gradient pattern
                for (NSUInteger y = 0; y < 256; y++) {
                    for (NSUInteger x = 0; x < 256; x++) {
                        NSUInteger index = y * 256 + x;
                        uint8_t r = (uint8_t)((x * 128) / 256 + 64);      // Red: 64-192
                        uint8_t g = (uint8_t)((y * 128) / 256 + 64);      // Green: 64-192
                        uint8_t b = 255;                                  // Blue: 255
                        uint8_t a = 255;                                  // Alpha: 255
                        gradientData[index] = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
                    }
                }

                MTLRegion region = MTLRegionMake2D(0, 0, 256, 256);
                [proactiveTexture replaceRegion:region
                                     mipmapLevel:0
                                       withBytes:gradientData
                                     bytesPerRow:256 * sizeof(uint32_t)];

                free(gradientData);
                NSLog(@"MGL PROACTIVE SUCCESS: Created 256x256 gradient texture (prevents magenta screen)");
            } else {
                NSLog(@"MGL PROACTIVE WARNING: Could not allocate gradient data");
            }

            // Store the proactive texture for future use
            if (!_proactiveTextures) {
                _proactiveTextures = [[NSMutableArray alloc] init];
            }
            [_proactiveTextures addObject:proactiveTexture];

        } else {
            NSLog(@"MGL PROACTIVE ERROR: Could not create proactive texture");
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL PROACTIVE ERROR: Exception creating proactive textures: %@", exception.reason);
    }

    NSLog(@"MGL PROACTIVE: Essential texture creation completed");
}

- (MTLCaptureDescriptor *)setupCaptureToFile: (id<MTLDevice>)device//(nonnull MTLDevice* )device // (nonnull MTKView *)view
{
    MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:@"MGL.gputrace"];
    descriptor.captureObject = device; //((MTKView *)view).device;
    
    return descriptor;
}

- (void)startCapture:(MTLCaptureDescriptor *) descriptor
{
    NSError *error = nil;
    BOOL success = [MTLCaptureManager.sharedCaptureManager startCaptureWithDescriptor:descriptor
                                                                                error:&error];
    if (!success) {
        NSLog(@" error capturing mtl => %@ ", [error localizedDescription] );
    }
}

// Stop the capture.
- (void)stopCapture
{
    [MTLCaptureManager.sharedCaptureManager stopCapture];
}

// CRITICAL FIX: Proper resource cleanup to prevent memory leaks and crashes
- (void)dealloc
{
    NSLog(@"MGL INFO: MGLRenderer dealloc - cleaning up Metal resources");

    @try {
        // Stop any ongoing capture
        [MTLCaptureManager.sharedCaptureManager stopCapture];

        // End any active rendering
        [self endRenderEncoding];

        /* Drop strong references held by the last-bound dedup cache before
         * releasing the underlying Metal resources below. */
        [self invalidateLastBoundState];

        // Cleanup command buffer and encoder
        if (_renderPassManager.state->currentCommandBuffer) {
            NSLog(@"MGL INFO: Releasing current command buffer");
            [_renderPassManager discardCurrentCommandBuffer];
        }

        if (_renderPassManager.state->currentRenderEncoder) {
            NSLog(@"MGL INFO: Releasing current render encoder");
            [_renderPassManager clearCurrentRenderEncoder];
        }

        [_renderPassManager shutdown];
        _renderPassManager = nil;

        [_queryManager shutdown];
        _queryManager = nil;

        if (_pipelineCache) {
            if (_pipelineCache.state->pipelineState) {
                NSLog(@"MGL INFO: Releasing pipeline state");
            }
            [_pipelineCache saveBinaryArchive];
            [_pipelineCache shutdown];
            _pipelineCache = nil;
        }
        for (uint16_t i = 0; i < _resourceFallback.samplerSnapshotCacheCount; i++) {
            _resourceFallback.samplerSnapshotCacheStates[i] = nil;
        }
        _resourceFallback.samplerSnapshotCacheCount = 0;
        _resourceFallback.samplerSnapshotCacheNext = 0;
        memset(_resourceFallback.samplerSnapshotCacheIndex, 0,
               sizeof(_resourceFallback.samplerSnapshotCacheIndex));

        // Cleanup drawable and layer
        if (_drawable) {
            NSLog(@"MGL INFO: Releasing drawable");
            _drawable = nil;
        }

        if (_layer) {
            NSLog(@"MGL INFO: Removing and releasing layer");
            [_layer removeFromSuperlayer];
            _layer = nil;
        }

        // Cleanup command queue and device
        if (_commandQueue) {
            NSLog(@"MGL INFO: Releasing command queue");
            _commandQueue = nil;
        }

        if (_device) {
            NSLog(@"MGL INFO: Releasing Metal device");
            _device = nil;
        }

        // Cleanup thread lock — _metalStateLock is an NSRecursiveLock (ObjC object,
        // requires nil release under ARC). _syncListLock is an os_unfair_lock value
        // type and needs no cleanup.
        if (_metalStateLock) {
            _metalStateLock = nil;
        }

        /* Task 4: Release all address-stable snapshot arena chunks. */
        mglDestroyBatchArena(&_batching.batchArena);

    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during dealloc cleanup: %@", exception);
    }

    NSLog(@"MGL INFO: MGLRenderer dealloc completed");
}

@end
