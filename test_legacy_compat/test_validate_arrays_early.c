/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Pure unit test for mglDrawValidateArraysEarly (O1.4 residual).
 */
#include "mgl_draw_validate.h"

#include <stdio.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static void test_early_gates(void)
{
    MGLValidateArraysEarlyStatus st = MGL_VALIDATE_ARRAYS_OK;
    int ok = 0;
    uint64_t first = 0, last = 0;

    expect(mglDrawValidateArraysEarly(0, 1, 0, 10, &first, &last, &st, &ok) == 0,
           "disabled → stop");
    expect(st == MGL_VALIDATE_ARRAYS_DISABLED && ok == 1, "disabled ok");

    expect(mglDrawValidateArraysEarly(1, 0, 0, 10, &first, &last, &st, &ok) == 0,
           "null ctx");
    expect(st == MGL_VALIDATE_ARRAYS_NULL_CTX && ok == 0, "null status");

    expect(mglDrawValidateArraysEarly(1, 1, 0, 0, &first, &last, &st, &ok) == 0,
           "zero count");
    expect(st == MGL_VALIDATE_ARRAYS_ZERO_COUNT, "zero status");

    expect(mglDrawValidateArraysEarly(1, 1, -1, 4, &first, &last, &st, &ok) == 0,
           "neg first");
    expect(st == MGL_VALIDATE_ARRAYS_INVALID_RANGE, "invalid range");

    expect(mglDrawValidateArraysEarly(1, 1, 2, 5, &first, &last, &st, &ok) == 1,
           "ok continue");
    expect(st == MGL_VALIDATE_ARRAYS_OK && ok == 1, "ok status");
    expect(first == 2u && last == 6u, "first/last");
}

int main(void)
{
    test_early_gates();
    if (g_fails) {
        fprintf(stderr, "%d failure(s)\n", g_fails);
        return 1;
    }
    puts("test_validate_arrays_early: ok");
    return 0;
}
