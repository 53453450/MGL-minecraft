#import "MGLPlatformRendererShell.h"

#include <stdio.h>
#include <string.h>

@implementation MGLPlatformRendererShell

- (instancetype)initWithView:(NSView *)view
{
    self = [super init];
    if (self) {
        _view = view;
    }
    return self;
}

- (int)performOperation:(MGLPlatformRendererShellOperation)operation
                context:(void *)context
                 result:(MGLPlatformRendererShellResult *)result
{
    if (result) memset(result, 0, sizeof(*result));
    if (!operation) return -1;
    @try {
        int status = operation(context);
        if (result) result->status = status;
        return status;
    } @catch (NSException *exception) {
        if (result) {
            result->status = -1;
            snprintf(result->exception_name, sizeof(result->exception_name),
                     "%s", exception.name.UTF8String ?: "NSException");
            snprintf(result->exception_reason, sizeof(result->exception_reason),
                     "%s", exception.reason.UTF8String ?: "unknown");
        }
        return -1;
    }
}

@end
