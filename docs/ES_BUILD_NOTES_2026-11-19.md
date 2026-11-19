# ES 构建临时记录（2026-11-19）

## 背景

`make es` 在 `MGL/src/L4_context/framebuffers.c` 失败：ES 构建使用的头文件没有提供
`GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS`，但 `MGL_GL_ES` 的 FBO 完整性检查路径会引用该枚举。

## 当前临时修复

在 `framebuffers.c` 增加了受 `#ifndef` 保护的兼容定义：

```c
#ifndef GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS
#define GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS 0x8CD9
#endif
```

该定义只补齐编译期常量；`MGL_GL_CORE` 路径不进入 ES 专用的尺寸检查分支，因此不会改变当前桌面 GL 主线行为。

## 验证

- `make es`：通过
- `make test-es-smoke`：通过

## 规范与后续工作

这只是暂时恢复 ES 动态库的构建，不代表 ES FBO 完整性实现已经完成规范收口。当前非核心路径仍会在附件尺寸不一致时返回
`GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS`。对于 ES 3.0 及以上上下文，这个状态不应作为正常的完整性结果生成；后续需要按实际支持的 ES 版本重新审查尺寸检查逻辑，并补充不匹配附件的 ES 规范回归测试。

ES 目前不是主线，本次先保留这个最小构建修复并记录技术债，后续单独处理行为层面的规范一致性。
