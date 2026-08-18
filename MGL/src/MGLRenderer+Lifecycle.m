/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Lifecycle.m
// Renderer construction, backend callback binding, proactive texture
// priming, Metal frame capture, and dealloc — extracted from MGLRenderer.m.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Lifecycle_Private.h"
#import "mgl.h"
#import "draw_command.h"

/* KVO context shared by the observer registration in
 * createMGLRendererAndBindToContext:view: and observeValueForKeyPath:. */
static void *s_kvoViewGeometryContext = &s_kvoViewGeometryContext;

@interface MGLRenderer (LifecycleBackendBoundary)
- (void)mglBackendWillDestroy:(MGLRendererBackendHandle *)backend;
@end

@implementation MGLRenderer (Lifecycle)

#pragma mark C interface to context functions

void mglRendererPlatformBackendWillDestroy(
    void *platform_shell,
    MGLRendererBackendHandle *backend)
{
    MGLRenderer *renderer = (__bridge MGLRenderer *)platform_shell;
    [renderer mglBackendWillDestroy:backend];
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
    if (![renderer mglRendererIsReady]) {
        NSLog(@"MGL ERROR: renderer initialization failed closed");
        return NULL;
    }
    // Ownership: the returned pointer is NON-OWNING (borrowed).
    // The context retains the renderer through platform_renderer_shell.
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
    if (![renderer mglRendererIsReady]) {
        NSLog(@"MGL ERROR: headless renderer initialization failed closed");
        return NULL;
    }
    // Ownership: the returned pointer is NON-OWNING (borrowed).
    // The context retains the renderer through platform_renderer_shell.
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
    mglClaimGLThread();            /* idempotent; records the init thread as the GL thread */
    ctx = glm_ctx;
    _backend = NULL;
    self.view = view;
    self.layer = nil;
    if (!self.view) {
        NSLog(@"MGL ERROR: failed to bind platform renderer view");
        return;
    }
    _renderPassManager = [MGLRenderPassManager new];
    [_renderPassManager setRuntimeContext:glm_ctx];

    /* start the DontCare frame generation at 1 so it never matches a
     * texture's zero-initialized mtl_rt_frame_generation stamp until that
     * texture is actually written this frame. */
    [_renderPassManager setDontCareFrameGeneration:1u];

    BOOL psoDedupEnabled = mglEnvFlagEnabledDefaultOn("MGL_PSO_DEDUP");
    BOOL depthStencilCacheEnabled = mglEnvFlagEnabledDefaultOn("MGL_DS_CACHE");
    BOOL binaryArchiveEnabled = mglEnvFlagEnabledDefaultOn("MGL_BINARY_ARCHIVE");
    _pipelineCache = [[MGLPipelineCache alloc]
        initWithPSODedupEnabled:psoDedupEnabled
      depthStencilCacheEnabled:depthStencilCacheEnabled
           binaryArchiveEnabled:binaryArchiveEnabled];
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
     * The C++ binding state's valid bit starts false so the first bind on the first encoder is
     * never incorrectly skipped. */
    [self invalidateLastBoundState];
    NSLog(@"MGL INFO: AGX GPU error tracking initialized");
    NSLog(@"MGL INFO: perf gates pso_dedup=%d ds_cache=%d arena=%d "
          "same_key_restore=%d dirty_key_delta=%d (set VAR=0 to disable)",
          _pipelineCache.state->psoDedupEnabled ? 1 : 0,
          _pipelineCache.state->dsCacheEnabled ? 1 : 0,
          _batching.arenaSnapshotEnabled ? 1 : 0,
          _batching.skipSameKeyRestoreEnabled ? 1 : 0,
          _batching.dirtyKeyDeltaEnabled ? 1 : 0);

    if (glm_ctx->renderer_backend) {
        mglRendererBackendDestroy(
            (MGLRendererBackendHandle **)&glm_ctx->renderer_backend);
    }
    if (glm_ctx->platform_renderer_shell) {
        CFRelease(glm_ctx->platform_renderer_shell);
        glm_ctx->platform_renderer_shell = NULL;
    }

    // VIRTUALIZED AGX DETECTION: Create Metal device with virtualization safety
    NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating Metal device with virtualization detection");

    // Create the Metal device
    id device = [self mglCreateSystemDefaultDevice];
    if (!device) {
        NSLog(@"MGL ERROR: Metal device not found - this is required for Apple Silicon");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, AGX GPU error tracking
        //        command recovery owner, _pipeline*Format/
        //        _pipelineCache.state->pipelineProgramName and
        //        _pipelineCache.state->pipelineStateCache.
        //   NIL: _device, _commandQueue, _view.
        // Continuing is pointless without a Metal device — every subsequent
        // operation depends on it.
        return; // Exit early rather than continuing with nil device
    }

    NSLog(@"MGL INFO: Metal device created: %@", device);

    MGLRendererBackendCreateInfo backendInfo = {
        .objc_device = (__bridge void *)device,
        .context = glm_ctx,
        .binding_slot_count = TEXTURE_UNITS,
        .query_capacity = 256u,
    };
    if (mglRendererBackendCreate(&backendInfo, &_backend) != 0) {
        NSLog(@"MGL ERROR: failed to create Metal-cpp renderer backend");
        return;
    }
    glm_ctx->renderer_backend = _backend;
    glm_ctx->platform_renderer_shell = (void *)CFBridgingRetain(self);
    _bindingStateOwner = mglRendererBackendGetOwner(
        _backend, MGL_RENDERER_BACKEND_OWNER_BINDING);
    _queryStateOwner = mglRendererBackendGetOwner(
        _backend, MGL_RENDERER_BACKEND_OWNER_QUERY);
    _gpuRecovery.commandRecoveryOwner = mglRendererBackendGetOwner(
        _backend, MGL_RENDERER_BACKEND_OWNER_RECOVERY);
    NSLog(@"MGL INFO: Metal-cpp renderer backend ready (%p)", _backend);
    mglRenderCppAttachRuntimeOwners(
        glm_ctx,
        _renderPassManager.state->currentCommandBufferOwner,
        _renderPassManager.state->currentRenderEncoderOwner,
        _renderPassManager.state->renderPassStateOwner);
    _pipelineCache.device = _device;

    /* Initialize AGX Capability Layer (centralized device detection +
     * capability queries + driver bug markers).  Replaces scattered
     * `containsString:@"AGX"` checks and hardcoded constants. */
    MGLCapabilityInit(&_capability, (__bridge void *)_device);

    // PROPER AGX VIRTUALIZATION DETECTION: Maintain Metal functionality with virtualization compatibility
    BOOL isVirtualized = _capability.isVirtualized;
    char deviceName[128];
    (void)mglRenderCppGetDeviceIdentity(
        (__bridge const void *)_device, NULL,
        deviceName, sizeof(deviceName));

    // DETECTION: Check if running in QEMU virtualization but keep Metal enabled
    if (isVirtualized) {
        isVirtualized = YES;
        NSLog(@"MGL INFO: AGX device detected - enabling virtualization compatibility mode: %s", deviceName);
        NSLog(@"MGL INFO: Metal functionality will be maintained with AGX virtualization safety measures");
    }

    // Create command queue with virtualization-safe settings
    if (isVirtualized) {
        NSLog(@"MGL INFO: VIRTUALIZED AGX - Enabling virtualization-safe command queue settings");
    }

    uint32_t maxCommandBuffers = isVirtualized
        ? (uint32_t)MGLCapabilityMaxConcurrentCommandBuffers(&_capability)
        : 0u;
    void *commandQueue = NULL;
    (void)mglRendererBackendResetCommandQueue(
        _backend, maxCommandBuffers, &commandQueue);
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Failed to create Metal command queue");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, AGX GPU error tracking
        //        fields, _pipeline*Format/_pipelineCache.state->pipelineProgramName,
        //        _pipelineCache.state->pipelineStateCache, _device,
        //        MTL4 compiler (if available), _capability.
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
    if (_pipelineCache.binaryArchiveEnabled) {
        if (@available(macOS 11.0, *)) {
            [_pipelineCache loadBinaryArchive];
        } else {
            [_pipelineCache disableBinaryArchive];
        }
    }

    _view = view;

    // PROPER FIX: Create Metal layer with AGX-safe settings in the platform shell.
    NSLog(@"MGL INFO: PROPER FIX - Creating Metal layer with AGX-safe settings");

    uint32_t requestedPixelFormat = ctx ? ctx->pixel_format.mtl_pixel_format : 0u;
    uint32_t pf = 0u;
    if (![self mglConfigureMetalLayerWithDevice:_device
                          requestedPixelFormat:requestedPixelFormat
                           actualPixelFormat:&pf]) {
        NSLog(@"MGL ERROR: Failed to create Metal layer");
        return;
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
          (unsigned long)pf,
          (unsigned long)requestedPixelFormat,
          ctx ? ctx->pixel_format.format : 0u,
          ctx ? ctx->pixel_format.type : 0u);
    /* Initial geometry: the renderer is created on the main thread (AppKit
     * window setup), so read the view geometry synchronously here.  Later
     * changes arrive via KVO → mglMainThreadSyncViewGeometry. */
    if (NSThread.isMainThread) {
        [self mglMainThreadSyncViewGeometry];
    } else {
        (void)[self mglApplyPendingDrawableSize];
    }

    /* Observe view geometry changes so the GL thread never needs to touch
     * NSView/NSWindow/NSScreen.  KVO fires on the main thread (bounds is only
     * mutated there), publishing an atomic drawable-size snapshot.  The
     * "window" keyPath is observed as well so resize/backing notifications
     * can be attached lazily once the view joins a window. */
    [_view addObserver:self
            forKeyPath:@"bounds"
               options:0
               context:s_kvoViewGeometryContext];
    [_view addObserver:self
            forKeyPath:@"window"
               options:NSKeyValueObservingOptionInitial
               context:s_kvoViewGeometryContext];

    mglDrawBuffer(glm_ctx, GL_FRONT);

    // Create initial command buffer for AGX safety
    @try {
        [_renderPassManager installNewCommandBufferFromQueue:(__bridge void *)_commandQueue];
        MGLRenderCppCommandBufferState commandState = {0};
        if (!mglRenderCppCommandBufferOwnerHasState(
                _renderPassManager.state->currentCommandBufferOwner,
                &commandState)) {
            NSLog(@"MGL ERROR: Failed to create initial Metal command buffer");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating initial Metal command buffer: %@", exception);
    }
    
    // PROACTIVE TEXTURE CREATION: Create essential textures to break sync loop
    NSLog(@"MGL INFO: PROACTIVE - Creating essential textures to prevent magenta screen");
    [self createProactiveTextures];

    // GPU capture setup is exposed by MGLPlatformRendererShell when needed.
}

- (BOOL)mglRendererIsReady
{
    if (!ctx || !_device || !_backend ||
        mglRendererBackendIsReady(_backend) != 1 ||
        !_commandQueueOwner || !_commandQueue || !_layer || !_renderPassManager) {
        return NO;
    }

    MGLRenderCppCommandBufferState commandState = {0};
    return mglRenderCppCommandBufferOwnerHasState(
        _renderPassManager.state->currentCommandBufferOwner,
        &commandState);
}

- (void)mglBackendWillDestroy:(MGLRendererBackendHandle *)backend
{
    if (_backend != backend) return;
    _backend = NULL;
    _bindingStateOwner = NULL;
    _queryStateOwner = NULL;
    _gpuRecovery.commandRecoveryOwner = NULL;
}

/* Publish view geometry to the GL thread as an atomic snapshot.  Main thread
 * only — this is the sole place NSView/NSWindow/NSScreen are read, so the
 * render thread never touches AppKit.  The GL thread consumes the snapshot via
 * mglApplyPendingDrawableSize and sets CAMetalLayer.drawableSize. */
- (void)mglMainThreadSyncViewGeometry
{
    NSAssert(NSThread.isMainThread, @"AppKit geometry must be read on main thread");
    if (!_view || ![self mglHasMetalLayer]) {
        return;
    }

    NSRect bounds = [_view bounds];
    if (bounds.size.width <= 0.0 || bounds.size.height <= 0.0) {
        bounds = [_view frame];
        bounds.origin = NSZeroPoint;
    }

    NSRect backingBounds = [_view convertRectToBacking:bounds];
    CGFloat scale = 1.0;
    if (bounds.size.width > 0.0 && backingBounds.size.width > 0.0) {
        scale = backingBounds.size.width / bounds.size.width;
    } else {
        NSWindow *window = [_view window];
        if (window) {
            scale = [window backingScaleFactor];
        } else if ([NSScreen mainScreen]) {
            scale = [[NSScreen mainScreen] backingScaleFactor];
        }
        if (scale <= 0.0) {
            scale = 1.0;
        }
        backingBounds = NSMakeRect(0.0, 0.0, bounds.size.width * scale, bounds.size.height * scale);
    }

    [self mglSetMetalLayerFrame:bounds contentsScale:scale];

    uint32_t pw = (uint32_t)MAX(1.0, backingBounds.size.width + 0.5);
    uint32_t ph = (uint32_t)MAX(1.0, backingBounds.size.height + 0.5);
    atomic_store_explicit(&_pendingDrawableW, pw, memory_order_relaxed);
    atomic_store_explicit(&_pendingDrawableH, ph, memory_order_relaxed);
    atomic_store_explicit(&_drawableSizeDirty, true, memory_order_release);
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context
{
    if (context == s_kvoViewGeometryContext) {
        if ([keyPath isEqualToString:@"window"]) {
            [self mglUpdateWindowNotificationObserver];
        }
        [self mglMainThreadSyncViewGeometry];
        return;
    }
    [super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
}

/* Attach/detach window observation as the view's window changes.  The window
 * is not known when the renderer is created, so this is wired lazily. */
- (void)mglUpdateWindowNotificationObserver
{
    NSWindow *window = _view.window;
    if (window == _observedWindow) {
        return;
    }
    if (_observedWindow) {
        [[NSNotificationCenter defaultCenter] removeObserver:self
                                                        name:NSWindowDidResizeNotification
                                                      object:_observedWindow];
        [[NSNotificationCenter defaultCenter] removeObserver:self
                                                        name:NSWindowDidChangeBackingPropertiesNotification
                                                      object:_observedWindow];
    }
    _observedWindow = window;
    if (window) {
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(mglWindowGeometryChanged:)
                                                     name:NSWindowDidResizeNotification
                                                   object:window];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(mglWindowGeometryChanged:)
                                                     name:NSWindowDidChangeBackingPropertiesNotification
                                                   object:window];
    }
}

- (void)mglWindowGeometryChanged:(NSNotification *)notification
{
    (void)notification;
    [self mglMainThreadSyncViewGeometry];
}

// PROACTIVE TEXTURE CREATION: Create essential textures during initialization to break sync loop
- (void)createProactiveTextures
{
    NSLog(@"MGL PROACTIVE: Starting essential texture creation");

    @try {
        if (mglRendererBackendCreateProactiveTexture(_backend) == 0) {
            NSLog(@"MGL PROACTIVE SUCCESS: Created 256x256 gradient texture (prevents magenta screen)");
        } else {
            NSLog(@"MGL PROACTIVE ERROR: Could not create proactive texture");
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL PROACTIVE ERROR: Exception creating proactive textures: %@", exception.reason);
    }

    NSLog(@"MGL PROACTIVE: Essential texture creation completed");
}

// CRITICAL FIX: Proper resource cleanup to prevent memory leaks and crashes
- (void)dealloc
{
    NSLog(@"MGL INFO: MGLRenderer dealloc - cleaning up Metal resources");

    @try {
        /* Remove the geometry observers before any view/state teardown. */
        if (_view) {
            [_view removeObserver:self forKeyPath:@"bounds" context:s_kvoViewGeometryContext];
            [_view removeObserver:self forKeyPath:@"window" context:s_kvoViewGeometryContext];
        }
        /* Detach window notifications without the lazy re-wiring path. */
        if (_observedWindow) {
            [[NSNotificationCenter defaultCenter] removeObserver:self
                                                            name:NSWindowDidResizeNotification
                                                          object:_observedWindow];
            [[NSNotificationCenter defaultCenter] removeObserver:self
                                                            name:NSWindowDidChangeBackingPropertiesNotification
                                                          object:_observedWindow];
            _observedWindow = nil;
        }

        // Stop any ongoing capture
        [self mglStopCapture];

        // End any active rendering
        [self endRenderEncoding];

        /* Drop strong references held by the last-bound dedup cache before
         * releasing the underlying Metal resources below. */
        [self invalidateLastBoundState];
        // Cleanup command buffer and encoder
        MGLRenderCppCommandBufferState commandState = {0};
        if (mglRenderCppCommandBufferOwnerHasState(
                _renderPassManager.state->currentCommandBufferOwner,
                &commandState)) {
            NSLog(@"MGL INFO: Releasing current command buffer");
            [_renderPassManager discardCurrentCommandBuffer];
        }

        if (mglRenderCppRenderEncoderOwnerHasCurrent(
                _renderPassManager.state->currentRenderEncoderOwner) == 1) {
            NSLog(@"MGL INFO: Releasing current render encoder");
            [_renderPassManager clearCurrentRenderEncoder];
        }

        MGLRendererBackendShutdownResult shutdownResult = {0};
        if (_backend &&
            mglRendererBackendShutdown(_backend, &shutdownResult) != 0) {
            NSLog(@"MGL ERROR: renderer backend shutdown wait failed code=%lld",
                  shutdownResult.last_submission_error_code);
        }

        [_renderPassManager setRuntimeContext:NULL];
        [_renderPassManager shutdown];
        _renderPassManager = nil;

        mglRenderCppDetachRuntimeOwners(ctx);

        if (_pipelineCache) {
            if (_pipelineCache.state->pipelineState) {
                NSLog(@"MGL INFO: Releasing pipeline state");
            }
            [_pipelineCache saveBinaryArchive];
            [_pipelineCache shutdown];
            _pipelineCache = nil;
        }
        if (_backend && mglRendererBackendIsDestroying(_backend) != 1) {
            if (ctx && ctx->renderer_backend == _backend) {
                mglRendererBackendDestroy(
                    (MGLRendererBackendHandle **)&ctx->renderer_backend);
            } else {
                mglRendererBackendDestroy(&_backend);
            }
        }
        _backend = NULL;
        _bindingStateOwner = NULL;
        _queryStateOwner = NULL;
        _gpuRecovery.commandRecoveryOwner = NULL;

        // Cleanup drawable and layer
        if (_drawable) {
            NSLog(@"MGL INFO: Releasing drawable");
            _drawable = nil;
        }

        if ([self mglHasMetalLayer]) {
            NSLog(@"MGL INFO: Removing and releasing layer");
            [self mglDetachMetalLayer];
        }

        /* Task 4: Release all address-stable snapshot arena chunks. */
        mglDestroyBatchArena(&_batching.batchArena);
        _view = nil;

    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during dealloc cleanup: %@", exception);
    }

    NSLog(@"MGL INFO: MGLRenderer dealloc completed");
}

@end
