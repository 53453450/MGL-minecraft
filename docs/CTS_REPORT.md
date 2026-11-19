# GL 4.6 CTS 一致性结果

> 面向 README 展示的 CTS 运行摘要。数据取自 2026-11-19 全量运行，测试集为 Khronos 官方 `gl46-main.txt`（19,714 例）。

## 结果

| 指标 | 数值 |
|---|---:|
| 总用例 | 19,714 |
| **Pass** | **17,697** |
| **Fail** | **0** |
| NotSupported | 2,017 |
| Crash / Timeout | 0 / 0 |

## NotSupported 构成

剩余 2,017 例 NS 按测试组分布：

| 测试组 | 数量 | 说明 |
|---|---:|---|
| `fragment_shading_rate` | 1,067 | 已在实现路线中，暂未覆盖 |
| `gpu_shader_fp64` | 658 | FP64 不在支持范围 |
| `texture_swizzle` | 192 | 多数为特定 target 不适用 |
| `es_31_compatibility`、`sample_variables` | 38 + 38 | 采样数超过格式支持上限 |
| `cull_distance` | 16 | 不适用 |
| `vertex_attrib_64bit` | 4 | FP64 相关 |
| 其他 | 4 | ES-only / 不适用于 OpenGL 上下文 |

NS 中主要是**能力范围与待实现项**，不是执行失败。MGL 在 Apple Silicon 上通过 Metal 转译提供 GL 4.6 / GLSL 4.60 接口；FP64 与部分 MSAA 8x 组合属于已知不覆盖的能力。

## 近期闭环的一致性工作

| 工作项 | 覆盖用例 |
|---|---|
| MS swizzle 格式可渲染性 | 24 例 NS → Pass |
| `ARB_shader_viewport_layer_array` | 3 例 NS → Pass |
| `KHR_blend_equation_advanced` | 88 例 NS → Pass |
| `GL_COLOR_LOGIC_OP` | 16 例 |
| TES 控制点 / 图元重启 / GL 错误码 / 双源混合 | 反例构造 25 例 |

上述修复均通过了对应子集的 before/after 回归；MS swizzle 一项另跑了 7,220 例格式/FBO 回归，全部通过。

## 说明

- 测试在 macOS Apple Silicon（M4）上完成，使用官方 Khronos GL 4.6 must-pass 用例集。
- 当前全量为 0 Fail；此前导致 8 例 NS 与 1 例 Fail 的 CTS 侧问题已在本机修正，并已向上游提交 PR：[NS](https://github.com/KhronosGroup/VK-GL-CTS/pull/656) · [Fail](https://github.com/KhronosGroup/VK-GL-CTS/pull/657)。