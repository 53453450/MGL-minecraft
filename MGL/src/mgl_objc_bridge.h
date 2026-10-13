/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_objc_bridge.h - plain-C++ helpers for talking to the Objective-C runtime
 * (P0-1, T5 option (a); log 206).
 *
 * The platform shell is being rewritten as C++ so that MGL/ needs no .m at all.
 * Everything the Objective-C language gave that code for free is spelled out
 * here instead, with no Objective-C syntax and no ARC in this header:
 *
 *   [object doThing:x]        -> mglSend<R>(object, sel, x)
 *   @selector(doThing:)       -> mglSel*("doThing:")  (cached per call site)
 *   @"literal"                -> CFSTR("literal")
 *   self->_ivar               -> mglRIvars(self)->_ivar (see mgl_renderer_ivars.h)
 *   __bridge / __bridge_transfer -> mglBridgingRetain / mglBridgingRelease
 *   @autoreleasepool { }      -> MGLScopedAutoreleasePool pool;
 *   @try/@catch (NSException *) -> try/catch (...) + mglTakeCaughtException()
 *
 * Rule 67: ARC's bridged casts are not stylistic.  `(__bridge_transfer id)x`
 * releases a +1, `(__bridge id)x` borrows, and the method-return convention
 * autoreleases - so a conversion that drops them without a replacement leaks.
 */

#ifndef MGL_OBJC_BRIDGE_H
#define MGL_OBJC_BRIDGE_H

#if !defined(__cplusplus)
#error "mgl_objc_bridge.h is C++ only; C callers use the mgl_* C ports"
#endif

#include <objc/message.h>
#include <objc/runtime.h>

#include <CoreFoundation/CoreFoundation.h>

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ------------------------------------------------------------- ABI gate --- */

/* mglSend's typed function-pointer cast models the arm64 objc_msgSend ABI:
 * every argument and the return value travel in registers as the method
 * signature declares them.  On x86_64 a struct return or a floating-point
 * return would instead go through hidden _stret / _fpret registers, which the
 * cast does not model -- so on that ABI the template would need a different
 * implementation, not a comment.  MGL ships arm64 only (the platform floor is
 * macOS 26 / Metal 4), so make that a compile-time fact. */
#if !defined(__arm64__) && !defined(__aarch64__)
#error "mgl_objc_bridge.h implements the arm64 objc_msgSend ABI only (x86_64 needs _stret/_fpret handling)"
#endif

static_assert(sizeof(void *) == 8,
              "the message-send cast assumes 64-bit pointers");

/* `id`, spelled without Objective-C.  Struct name must stay `objc_object` so
 * that runtime functions returning `id` need no cast at all. */
typedef struct objc_object *MGLObjectId;

/* libobjc declares the pool API only for ObjC/ARC TUs. */
extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void *pool);

/* ARC's retain/release/autorelease entry points.  libobjc exports them -- they
 * are the ABI every ARC-compiled binary calls -- but the public SDK headers
 * declare them for ObjC/ARC TUs only, exactly like the pool API above, so they
 * are spelled out here.  Preferring them over `-retain`/`-release` messages
 * keeps selector lookup out of the hot ownership paths and routes tagged and
 * non-pointer-isa objects through the runtime's own handling.  Signatures are
 * the documented ones with `id` spelled as MGLObjectId (same struct pointer). */
extern "C" MGLObjectId objc_retain(MGLObjectId object);
extern "C" void objc_release(MGLObjectId object);
extern "C" MGLObjectId objc_autorelease(MGLObjectId object);

/* ------------------------------------------------------------ messages --- */

/* Typed message send.  The cast is what makes objc_msgSend's ABI work out:
 * every argument and the return value (struct returns included) must match the
 * real method signature. */
template <typename R, typename... Args>
static inline R mglSend(MGLObjectId object, SEL selector, Args... args)
{
    return ((R (*)(MGLObjectId, SEL, Args...))objc_msgSend)(object, selector, args...);
}

/* `[Class doThing]`: same thing with the class object in the receiver slot. */
static inline Class mglClass(const char *name)
{
    return objc_getClass(name);
}

static inline SEL mglCachedSel(SEL *slot, const char *name)
{
    if (!*slot) {
        *slot = sel_registerName(name);
    }
    return *slot;
}

/* Declare `static SEL s_selFoo;` at file scope, then MGL_SEL(s_selFoo, "foo"). */
#define MGL_SEL(slot, name) mglCachedSel(&(slot), (name))

/* ---------------------------------------------------------- ownership --- */

/* __bridge_retained id -> hand a +1 to C (CFBridgingRetain). */
static inline void *mglBridgingRetain(MGLObjectId object)
{
    return object ? (void *)objc_retain(object) : NULL;
}

/* __bridge_transfer id discarded / CFBridgingRelease -> give the +1 back. */
static inline void mglBridgingRelease(void *pointer)
{
    if (pointer) {
        objc_release((MGLObjectId)pointer);
    }
}

static inline void mglReleaseObject(MGLObjectId object)
{
    if (object) {
        objc_release(object);
    }
}

/* The return convention for a method/function returning `id` under ARC is
 * `objc_autoreleaseReturnValue`; a C caller that just borrows the result needs
 * exactly the same autorelease. */
static inline MGLObjectId mglAutoreleaseObject(MGLObjectId object)
{
    return object ? objc_autorelease(object) : NULL;
}

/* ------------------------------------------------------------ strings --- */

/* NSString from a NUL-terminated UTF-8 C string.  The result is *autoreleased*
 * (borrowed), not +1: `stringWithUTF8String:` is the autoreleasing convenience
 * constructor, so the name must not say `New`/`Create`.  Do not release the
 * result, and do not keep it past the enclosing autorelease pool. */
static inline MGLObjectId mglAutoreleasedUTF8String(const char *utf8)
{
    return utf8 ? mglSend<MGLObjectId>((MGLObjectId)objc_getClass("NSString"),
                                       sel_registerName("stringWithUTF8String:"), utf8)
                : NULL;
}

/* Borrowed NUL-terminated UTF-8 bytes owned by `string`: valid as long as the
 * string is, and (for an autoreleased string) until its pool drains.  Never
 * free the result. */
static inline const char *mglUTF8String(MGLObjectId string)
{
    if (!string) {
        return NULL;
    }
    const char *utf8 = mglSend<const char *>(string,
                                            sel_registerName("UTF8String"));
    return utf8;
}

/* --------------------------------------------------------- pools / ivars --- */

/* @autoreleasepool { ... } */
struct MGLScopedAutoreleasePool {
    void *token;
    MGLScopedAutoreleasePool() : token(objc_autoreleasePoolPush()) {}
    ~MGLScopedAutoreleasePool() { objc_autoreleasePoolPop(token); }
    MGLScopedAutoreleasePool(const MGLScopedAutoreleasePool &) = delete;
    MGLScopedAutoreleasePool &operator=(const MGLScopedAutoreleasePool &) = delete;
};

/* @finally, including the `@try/@finally` form with no `@catch`: the body runs
 * when the scope exits, whether by return, by fallthrough or while an exception
 * unwinds - and in the last case the exception keeps propagating, which is what
 * `@try/@finally` without `@catch` does.
 *
 *   auto finally = mglScopeExit([&] { teardown(); });
 */
template <typename Body>
struct MGLScopeExitGuard {
    Body body;
    explicit MGLScopeExitGuard(Body b) : body(b) {}
    ~MGLScopeExitGuard() { body(); }
    MGLScopeExitGuard(const MGLScopeExitGuard &) = delete;
    MGLScopeExitGuard &operator=(const MGLScopeExitGuard &) = delete;
};

template <typename Body>
static inline MGLScopeExitGuard<Body> mglScopeExit(Body body)
{
    return MGLScopeExitGuard<Body>(body);
}

/* Ivar offsets are resolved on first use (never at load time: the class must be
 * finished loading, categories included, before its layout is asked for) and
 * cached in a per-call-site `static ptrdiff_t`, which starts at PTRDIFF_MIN. */
static inline ptrdiff_t mglIvarOffset(ptrdiff_t &cache, Class cls, const char *name)
{
    if (cache == PTRDIFF_MIN) {
        Ivar ivar = cls ? class_getInstanceVariable(cls, name) : NULL;
        cache = ivar ? ivar_getOffset(ivar) : (ptrdiff_t)-1;
    }
    return cache;
}

#define MGL_IVAR(cache, cls, name) mglIvarOffset((cache), (cls), (name))

/* --------------------------------------------------------- exceptions --- */

/* `@catch (NSException *e)` cannot be spelled in C++: an ObjC exception is not
 * a typed C++ exception (verified: `catch (objc_object *)` does not match it,
 * only `catch (...)` does).  mgl_objc_exception_bridge.cpp installs an
 * exception preprocessor that records the thrown object per thread, so a
 * catch(...) block can still recover it - which the platform boundary has to
 * do, because MGLPlatformRendererShellResult carries the name and reason. */
MGLObjectId mglTakeCaughtException(void);
void mglFillCaughtException(MGLObjectId exception,
                            char *name_out, size_t name_cap,
                            char *reason_out, size_t reason_cap);
/* Both return borrowed UTF-8 bytes taken from the object's -description (the
 * second is that same call on the recorded exception; nil becomes "(null)").
 * The bytes belong to an autoreleased NSString, so they stay valid only until
 * the enclosing autorelease pool drains -- copy them instead of holding them
 * (mglFillCaughtException is the copying variant). */
const char *mglObjectDescriptionUTF8(MGLObjectId object);
const char *mglCaughtExceptionDescription(MGLObjectId exception);

#endif /* MGL_OBJC_BRIDGE_H */
