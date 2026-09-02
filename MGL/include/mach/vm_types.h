/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Host shim for mach/vm_types.h — real header on Apple, stubs elsewhere
 * so pure compile-time gtests can build on Linux CI.
 */

#ifndef MGL_MACH_VM_TYPES_H
#define MGL_MACH_VM_TYPES_H

#ifdef __APPLE__
#include_next <mach/vm_types.h>
#else

#include <stddef.h>
#include <stdint.h>

typedef uintptr_t vm_address_t;
typedef uintptr_t vm_offset_t;
typedef size_t vm_size_t;
typedef uint32_t natural_t;
typedef uint32_t mach_port_t;

#endif /* __APPLE__ */

#endif /* MGL_MACH_VM_TYPES_H */
