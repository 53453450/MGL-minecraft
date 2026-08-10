// MGLRenderer+Lifecycle.m
// Renderer construction, glm_ctx mtl_funcs binding, proactive texture
// priming, Metal frame capture, and dealloc — extracted from MGLRenderer.m.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Lifecycle_Private.h"
#import "mgl.h"
#import "mgl_metal_bridge.h"
#import "draw_command.h"
#include "mgl_render_cpp_objc.h" /* C ABI + ObjC descriptor state adapter */

/* KVO context shared by the observer registration in
 * createMGLRendererAndBindToContext:view: and observeValueForKeyPath:. */
static void *s_kvoViewGeometryContext = &s_kvoViewGeometryContext;

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
    if (mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() != NULL) {
        void *texture = NULL;
        MGLRenderCppTextureDescriptorState state =
            mglRenderCppTextureDescriptorStateFromObjC(descriptor);
        if (mglRenderCppCreateTextureFromState(&state, NULL, &texture) == 0 &&
            texture) {
            return (__bridge_transfer id<MTLTexture>)texture;
        }
    }
    return [device newTextureWithDescriptor:descriptor];
}

static void mglLifecycleReplaceTextureRegion(id<MTLTexture> texture,
                                             MTLRegion region,
                                             NSUInteger level,
                                             const void *bytes,
                                             NSUInteger bytesPerRow)
{
    if (mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() != NULL &&
        mglRenderCppTextureReplaceRegion(
            (__bridge void *)texture,
            region.origin.x, region.origin.y, region.origin.z,
            region.size.width, region.size.height, region.size.depth,
            level, 0, bytes, bytesPerRow, 0, 0) == 0) {
        return;
    }
    [texture replaceRegion:region mipmapLevel:level withBytes:bytes
                bytesPerRow:bytesPerRow];
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

    /* These callbacks release bridge-owned Metal objects directly through
     * Metal-cpp and do not need the ObjC renderer target.  Owner-dependent
     * callbacks remain on mgl_metal_bridge until their state moves to C++. */
    if (mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP")) {
        if (mglRenderCppGetDevice() != NULL) {
            glm_ctx->mtl_funcs.mtlBindBuffer = mglRenderCppBindBuffer;
            glm_ctx->mtl_funcs.mtlBufferSubData =
                mglRenderCppBufferSubData;
            glm_ctx->mtl_funcs.mtlMapUnmapBuffer =
                mglRenderCppMapUnmapBuffer;
            glm_ctx->mtl_funcs.mtlReadBackBuffer =
                mglRenderCppReadBackBuffer;
            glm_ctx->mtl_funcs.mtlFlushBufferRange =
                mglRenderCppFlushBufferRange;
            glm_ctx->mtl_funcs.mtlBindProgram = mglRenderCppBindProgram;
        }
        glm_ctx->mtl_funcs.mtlDeleteMTLObj = mglRenderCppDeleteMTLObj;
        glm_ctx->mtl_funcs.release_buffer_metal_data =
            mglRenderCppReleaseBufferMetalData;
        glm_ctx->mtl_funcs.mtlWaitForSync = mglRenderCppWaitForSync;
        glm_ctx->mtl_funcs.mtlGetSyncStatus = mglRenderCppGetSyncStatus;
        glm_ctx->mtl_funcs.mtlReleaseSync = mglRenderCppReleaseSync;
    }
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
    mglClaimGLThread();            /* idempotent; records the init thread as the GL thread */
    ctx = glm_ctx;
    _renderPassManager = [MGLRenderPassManager new];

    /* start the DontCare frame generation at 1 so it never matches a
     * texture's zero-initialized mtl_rt_frame_generation stamp until that
     * texture is actually written this frame. */
    [_renderPassManager setDontCareFrameGeneration:1u];

    // Initialize AGX GPU error tracking
    _gpuRecovery.gpuErrorLock = OS_UNFAIR_LOCK_INIT;
    _gpuRecovery.consecutiveGPUErrors = 0;
    _gpuRecovery.lastGPUErrorTime = 0;
    _gpuRecovery.gpuErrorRecoveryMode = NO;
    BOOL psoDedupEnabled = mglEnvFlagEnabledDefaultOn("MGL_PSO_DEDUP");
    BOOL depthStencilCacheEnabled = mglEnvFlagEnabledDefaultOn("MGL_DS_CACHE");
    BOOL binaryArchiveEnabled = mglEnvFlagEnabledDefaultOn("MGL_BINARY_ARCHIVE");
    _pipelineCache = [[MGLPipelineCache alloc]
        initWithPSODedupEnabled:psoDedupEnabled
      depthStencilCacheEnabled:depthStencilCacheEnabled
           binaryArchiveEnabled:binaryArchiveEnabled];
    _bindingStateOwner = mglRenderCppBindingCreate(TEXTURE_UNITS);
    if (!_bindingStateOwner) {
        NSLog(@"MGL ERROR: failed to create Metal-cpp binding state owner");
    }
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

    [self bindObjFuncsToGLMContext: glm_ctx];

    // VIRTUALIZED AGX DETECTION: Create Metal device with virtualization safety
    NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating Metal device with virtualization detection");

    // Create the Metal device
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        NSLog(@"MGL ERROR: Metal device not found - this is required for Apple Silicon");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, AGX GPU error tracking
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

    /* METALCPP 路径（Phase 1）：把现有 id<MTLDevice> 桥接给 C++ 渲染门面
     * （+1 retain，shutdown 时 release）。AIR 加载器/PSO 走 MGL_USE_METALCPP=1
     * 时经 mglRenderCppGetDevice() 取用。 */
    if (mglRenderCppInit((__bridge void *)_device) != 0) {
        NSLog(@"MGL ERROR: mglRenderCppInit failed (Metal-cpp bridge)");
    } else {
        NSLog(@"MGL INFO: Metal-cpp renderer bridge ready (%p)",
              mglRenderCppGetDevice());
        /* Rebind now that the C++ device exists.  The first bind above keeps
         * early-failure cleanup valid; this bind selects migrated callbacks. */
        [self bindObjFuncsToGLMContext:glm_ctx];
        if (mglRenderCppCreateQueryStateOwner(256u, &_queryStateOwner) != 0) {
            _queryStateOwner = NULL;
            NSLog(@"MGL ERROR: failed to create Metal-cpp query state owner");
        }
    }
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

    if (mglEnvFlagEnabledDefaultOn("MGL_USE_METALCPP") && mglRenderCppGetDevice()) {
        uint32_t maxCommandBuffers = isVirtualized
            ? (uint32_t)MGLCapabilityMaxConcurrentCommandBuffers(&_capability)
            : 0u;
        _commandQueue = mglRenderCppCreateOrResetCommandQueueOwner(
            &_commandQueueOwner, maxCommandBuffers);
    }
    if (!_commandQueue) {
        mglRenderCppDestroyCommandQueueOwner(&_commandQueueOwner);
        _commandQueue = [_device newCommandQueueWithDescriptor:queueDescriptor];
    }
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Failed to create Metal command queue");
        // Intentional early return on critical Metal initialization failure.
        // The renderer is left in a PARTIALLY INITIALIZED state:
        //   SET: ctx, AGX GPU error tracking
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
        /* Destroy the per-context C++ binding state before final renderer
         * shutdown releases any remaining renderer-owned Metal objects. */
        mglRenderCppBindingDestroy(_bindingStateOwner);
        _bindingStateOwner = NULL;

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

        mglRenderCppDestroyQueryStateOwner(&_queryStateOwner);

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
        mglRenderCppDestroyCommandQueueOwner(&_commandQueueOwner);

        if (_device) {
            NSLog(@"MGL INFO: Releasing Metal device");
            _device = nil;
        }

        /* METALCPP 路径：释放 C++ 渲染门面持有的 device 引用。 */
        mglRenderCppShutdown();

        /* Task 4: Release all address-stable snapshot arena chunks. */
        mglDestroyBatchArena(&_batching.batchArena);

    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during dealloc cleanup: %@", exception);
    }

    NSLog(@"MGL INFO: MGLRenderer dealloc completed");
}

@end
