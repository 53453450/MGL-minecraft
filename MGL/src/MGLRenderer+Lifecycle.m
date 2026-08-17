// MGLRenderer+Lifecycle.m
// Renderer construction, backend callback binding, proactive texture
// priming, Metal frame capture, and dealloc — extracted from MGLRenderer.m.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Lifecycle_Private.h"
#import "mgl.h"
#import "draw_command.h"
#include "mgl_render_cpp_objc.h" /* C ABI + ObjC descriptor state adapter */

/* KVO context shared by the observer registration in
 * createMGLRendererAndBindToContext:view: and observeValueForKeyPath:. */
static void *s_kvoViewGeometryContext = &s_kvoViewGeometryContext;

@interface MGLRenderer (LifecycleBackendBoundary)
- (void)mglBackendWillDestroy:(MGLRendererBackendHandle *)backend;
@end

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

static id<MTLTexture> mglLifecycleCreateTexture(
    id<MTLDevice> device,
    MTLTextureDescriptor *descriptor)
{
    (void)device;
    void *texture = NULL;
    MGLRenderCppTextureDescriptorState state =
        mglRenderCppTextureDescriptorStateFromObjC(descriptor);
    if (mglRenderCppCreateTextureFromState(&state, NULL, &texture) == 0 &&
        texture) {
        return (__bridge_transfer id<MTLTexture>)texture;
    }
    return nil;
}

static void mglLifecycleReplaceTextureRegion(id<MTLTexture> texture,
                                             MTLRegion region,
                                             NSUInteger level,
                                             const void *bytes,
                                             NSUInteger bytesPerRow)
{
    if (mglRenderCppTextureReplaceRegion(
            (__bridge void *)texture,
            region.origin.x, region.origin.y, region.origin.z,
            region.size.width, region.size.height, region.size.depth,
            level, 0, bytes, bytesPerRow, 0, 0) == 0) {
        return;
    }
    NSLog(@"MGL ERROR: proactive texture upload failed through Metal-cpp");
}

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
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
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

    NSLog(@"MGL INFO: Metal device created: %@", _device);

    MGLRendererBackendCreateInfo backendInfo = {
        .objc_device = (__bridge void *)_device,
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
    NSString *deviceName = [_device name];

    // DETECTION: Check if running in QEMU virtualization but keep Metal enabled
    if (isVirtualized) {
        isVirtualized = YES;
        NSLog(@"MGL INFO: AGX device detected - enabling virtualization compatibility mode: %@", deviceName);
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
    if (mglRendererBackendResetCommandQueue(
            _backend, maxCommandBuffers, &commandQueue) == 0) {
        _commandQueue = (__bridge id<MTLCommandQueue>)commandQueue;
        _commandQueueOwner = mglRendererBackendGetOwner(
            _backend, MGL_RENDERER_BACKEND_OWNER_COMMAND_QUEUE);
    }
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
        [_renderPassManager installNewCommandBufferFromQueue:_commandQueue];
        MGLRenderCppCommandBufferState commandState = {0};
        if (!mglRenderCommandBufferOwnerState(
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

    // capture Metal commands in MGL.gputrace
    // necessitates Info.plist in the cwd, see https://stackoverflow.com/a/64172784
    //MTLCaptureDescriptor *descriptor = [self setupCaptureToFile: _device];
    //[self startCapture:descriptor];
}

- (BOOL)mglRendererIsReady
{
    if (!ctx || !_device || !_backend ||
        mglRendererBackendIsReady(_backend) != 1 ||
        !_commandQueueOwner || !_commandQueue || !_layer || !_renderPassManager) {
        return NO;
    }

    MGLRenderCppCommandBufferState commandState = {0};
    return mglRenderCommandBufferOwnerState(
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
    _commandQueueOwner = NULL;
}

/* Publish view geometry to the GL thread as an atomic snapshot.  Main thread
 * only — this is the sole place NSView/NSWindow/NSScreen are read, so the
 * render thread never touches AppKit.  The GL thread consumes the snapshot via
 * mglApplyPendingDrawableSize and sets CAMetalLayer.drawableSize. */
- (void)mglMainThreadSyncViewGeometry
{
    NSAssert(NSThread.isMainThread, @"AppKit geometry must be read on main thread");
    if (!_view || !_layer) {
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

    _layer.frame = bounds;
    _layer.contentsScale = scale;

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
        // Create a simple 2D texture with gradient pattern to prevent magenta screens
        MTLTextureDescriptor *proactiveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                                          width:256
                                                                                                         height:256
                                                                                                      mipmapped:NO];
        proactiveDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        proactiveDesc.storageMode = MTLStorageModeShared;

        id<MTLTexture> proactiveTexture =
            mglLifecycleCreateTexture(_device, proactiveDesc);
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
                mglLifecycleReplaceTextureRegion(
                    proactiveTexture, region, 0, gradientData,
                    256 * sizeof(uint32_t));

                free(gradientData);
                NSLog(@"MGL PROACTIVE SUCCESS: Created 256x256 gradient texture (prevents magenta screen)");
            } else {
                NSLog(@"MGL PROACTIVE WARNING: Could not allocate gradient data");
            }

            // Store the proactive texture for future use
            if (mglRendererBackendRetainProactiveTexture(
                    _backend, (__bridge void *)proactiveTexture) != 0) {
                NSLog(@"MGL WARNING: Failed to retain proactive texture in renderer backend");
            }

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
        [MTLCaptureManager.sharedCaptureManager stopCapture];

        // End any active rendering
        [self endRenderEncoding];

        /* Drop strong references held by the last-bound dedup cache before
         * releasing the underlying Metal resources below. */
        [self invalidateLastBoundState];
        // Cleanup command buffer and encoder
        MGLRenderCppCommandBufferState commandState = {0};
        if (mglRenderCommandBufferOwnerState(
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
        _commandQueueOwner = NULL;

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

        /* Task 4: Release all address-stable snapshot arena chunks. */
        mglDestroyBatchArena(&_batching.batchArena);
        _view = nil;

    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during dealloc cleanup: %@", exception);
    }

    NSLog(@"MGL INFO: MGLRenderer dealloc completed");
}

@end
