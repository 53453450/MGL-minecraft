/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_objc_exception_bridge.cpp - recovering the thrown object inside a C++
 * catch(...) block (P0-1, T5 option (a); log 206).
 *
 * Converting the shell's @try/@catch blocks to C++ means losing the binding of
 * `@catch (NSException *e)`.  Measured on this platform before writing any of
 * this (log 206, /private/tmp/ehtest2):
 *
 *   - a C++ `catch (...)` DOES catch an ObjC exception, but
 *   - `catch (objc_object *)` does NOT match it (the runtime raises it as a
 *     foreign C++ exception), and `__cxa_current_exception_type()` is null -
 *     so the object cannot be reached from the catch clause itself, and
 *   - libobjc's own escape hatch, objc_begin_catch(), needs the landing pad's
 *     exception buffer, which only the compiler-generated @catch has.
 *
 * objc_setExceptionPreprocessor() is the supported hook that runs inside
 * objc_exception_throw before unwinding starts, so it sees every thrown object,
 * whatever raised it (@throw or -[NSException raise], verified with a real
 * Foundation exception).  We chain to the previous preprocessor instead of
 * replacing it, record the object in thread-local storage, and hand it to the
 * catch block.  If a later component installs its own preprocessor ours is
 * simply no longer called: the callers then fall back to the same placeholder
 * strings the Objective-C code used for a nil exception ("NSException" /
 * "unknown"), so diagnostics degrade instead of breaking.
 *
 * The recorded pointer is borrowed, never retained: it is only read inside the
 * catch block that the same throw reaches, where the unwinder keeps it alive.
 */

#include "mgl_objc_bridge.h"

#include <objc/objc-exception.h>

#include <stdio.h>

/* One TU owns the hook: a preprocessor installed from several TUs would record
 * into the installing TU's thread-local and every other TU would read NULL. */
static _Thread_local MGLObjectId mglLastThrown = NULL;
static objc_exception_preprocessor mglPreviousPreprocessor = NULL;

static MGLObjectId mglRecordThrownException(MGLObjectId exception)
{
    mglLastThrown = exception;
    return mglPreviousPreprocessor ? mglPreviousPreprocessor(exception)
                                   : exception;
}

/* Rule 64: nothing calls this - the constructor attribute is the only entry
 * point.  Installing at load time means an exception raised during any later
 * startup code is already recorded. */
__attribute__((constructor))
static void mglInstallExceptionRecorder(void)
{
    mglPreviousPreprocessor = objc_setExceptionPreprocessor(mglRecordThrownException);
}

MGLObjectId mglTakeCaughtException(void)
{
    MGLObjectId exception = mglLastThrown;
    mglLastThrown = NULL;
    return exception;
}

/* `exception.name.UTF8String ?: "NSException"` and the reason twin, which is
 * what the shell's exception boundary published in both directions. */
void mglFillCaughtException(MGLObjectId exception,
                            char *name_out, size_t name_cap,
                            char *reason_out, size_t reason_cap)
{
    if (name_out && name_cap) {
        const char *name = exception
            ? mglUTF8String(mglSend<MGLObjectId>(exception, sel_registerName("name")))
            : NULL;
        snprintf(name_out, name_cap, "%s", name ? name : "NSException");
    }
    if (reason_out && reason_cap) {
        const char *reason = exception
            ? mglUTF8String(mglSend<MGLObjectId>(exception, sel_registerName("reason")))
            : NULL;
        snprintf(reason_out, reason_cap, "%s", reason ? reason : "unknown");
    }
}

/* NSLog's %@ for an object (and "(null)" for nil, which is what %@ prints). */
const char *mglObjectDescriptionUTF8(MGLObjectId object)
{
    if (!object) {
        return "(null)";
    }
    const char *utf8 =
        mglUTF8String(mglSend<MGLObjectId>(object, sel_registerName("description")));
    return utf8 ? utf8 : "(null)";
}

const char *mglCaughtExceptionDescription(MGLObjectId exception)
{
    return mglObjectDescriptionUTF8(exception);
}
