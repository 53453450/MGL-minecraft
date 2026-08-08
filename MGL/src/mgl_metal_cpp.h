//------------------------------------------------------------------------------------------------
// mgl_metal_cpp.h — Metal-cpp 头引入 + ObjC device 桥接（唯一公共头）
//
// 私有实现宏（NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION）只能在
// 一个 TU 定义（mgl_render_cpp.cpp：include 本头之前定义宏，实现符号由此 TU
// 产出）。其他 TU（如 mgl_air_loader.cpp）仅 include 本头 —— Metal.hpp 的方法
// 实现均为 inline，重复 include 无重复符号问题。
//
// 本头不暴露给 C 侧：C 边界一律走 mgl_render_cpp.h / mgl_air_loader.h 的纯 C 接口。
//------------------------------------------------------------------------------------------------
#pragma once

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

namespace mgl {

// 桥接现有 ObjC id<MTLDevice>：C++ 侧 +1 retain，生命周期由渲染器持有，
// mglRenderCppShutdown 时 Release；ObjC 侧保留自己那份，两侧各自 balance。
// （MTL::Device* 与 id<MTLDevice> 指针同地址，reinterpret 桥接，DXMT 同款。）
MTL::Device* wrapDevice(void* objcDevice);

} // namespace mgl
