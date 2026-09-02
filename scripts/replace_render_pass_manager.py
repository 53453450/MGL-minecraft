#!/usr/bin/env python3
"""Replace MGLRenderPassManager ObjC shell with embedded MGLCommandState + mglCmd* API."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MGL_SRC = ROOT / "MGL" / "src"
MGL_INC = ROOT / "MGL" / "include"

METHOD_MAP = {
    "setRuntimeContext:": "mglCmdSetRuntimeContext(&_commandState, ",
    "updateRenderPassIdentityForContext:": "mglCmdUpdateRenderPassIdentityForContext(&_commandState, ",
    "clearRenderPassIdentity": "mglCmdClearRenderPassIdentity(&_commandState)",
    "installNewCommandBufferFromQueue:": "mglCmdInstallNewCommandBufferFromQueue(&_commandState, ",
    "detachCurrentCommandBufferForSubmission": "mglCmdDetachCurrentCommandBufferForSubmission(&_commandState)",
    "discardCurrentCommandBuffer": "mglCmdDiscardCurrentCommandBuffer(&_commandState)",
    "commitDetachedCommandBufferIfOwned:": "mglCmdCommitDetachedCommandBufferIfOwned(&_commandState, ",
    "commitCommandBufferTransaction:recoveryOwner:waitForCompletion:result:":
        "mglCmdCommitCommandBufferTransaction(&_commandState, ",
    "hasLastSubmittedCommandBuffer": "mglCmdHasLastSubmittedCommandBuffer(&_commandState)",
    "waitForLastSubmittedCommandBuffer:": "mglCmdWaitForLastSubmittedCommandBuffer(&_commandState, ",
    "consumeTransactionCreatedCurrentCommandBuffer":
        "mglCmdConsumeTransactionCreatedCurrentCommandBuffer(&_commandState)",
    "releaseDetachedCommandBufferIfOwned:":
        "mglCmdReleaseDetachedCommandBufferIfOwned(&_commandState, ",
    "appendSyncToCurrentCommandBuffer:":
        "mglCmdAppendSyncToCurrentCommandBuffer(&_commandState, ",
    "clearCurrentCommandBufferSyncListEntries":
        "mglCmdClearCurrentCommandBufferSyncListEntries(&_commandState)",
    "preparePendingEventWithDevice:syncName:":
        "mglCmdPreparePendingEventWithDevice(&_commandState, ",
    "detachPendingEventWithSyncName:":
        "mglCmdDetachPendingEventWithSyncName(&_commandState, ",
    "clearPendingEvent": "mglCmdClearPendingEvent(&_commandState)",
    "installRenderEncoder:": "mglCmdInstallRenderEncoder(&_commandState, ",
    "createRenderEncoder": "mglCmdCreateRenderEncoder(&_commandState)",
    "endCurrentRenderEncoder": "mglCmdEndCurrentRenderEncoder(&_commandState)",
    "clearCurrentRenderEncoder": "mglCmdClearCurrentRenderEncoder(&_commandState)",
    "beginCommandBufferCommit": "mglCmdBeginCommandBufferCommit(&_commandState)",
    "endCommandBufferCommit": "mglCmdEndCommandBufferCommit(&_commandState)",
    "mdiArgumentScratchBufferWithDevice:length:offset:":
        "mglCmdMdiArgumentScratchBufferWithDevice(&_commandState, ",
    "resetMDIScratch": "mglCmdResetMdiScratch(&_commandState)",
    "installNewRenderPassDescriptor": "mglCmdInstallNewRenderPassDescriptor(&_commandState)",
    "setFboMatchCacheResult:fboName:generation:":
        "mglCmdSetFboMatchCacheResult(&_commandState, ",
    "clearFboMatchCache": "mglCmdClearFboMatchCache(&_commandState)",
    "setTraceReplayFlushId:batchIndex:":
        "mglCmdSetTraceReplayFlushId(&_commandState, ",
    "setCurrentDrawUsesRTSampledCopy:":
        "mglCmdSetCurrentDrawUsesRTSampledCopy(&_commandState, ",
    "setDontCareFrameGeneration:":
        "mglCmdSetDontCareFrameGeneration(&_commandState, ",
    "incrementDontCareFrameGenerationWithWrap":
        "mglCmdIncrementDontCareFrameGenerationWithWrap(&_commandState)",
    "shutdown": "mglCmdShutdown(&_commandState)",
}


def replace_state_access(text: str) -> str:
    text = text.replace("_renderPassManager.state->", "_commandState.")
    text = re.sub(r"_renderPassManager\.state\b", "&_commandState", text)
    return text


def replace_method_calls(text: str) -> str:
    # Multi-line: [_renderPassManager\n    method...]
    for selector, repl in sorted(METHOD_MAP.items(), key=lambda x: -len(x[0])):
        pattern = (
            r"\[\s*_renderPassManager\s+"
            + re.escape(selector.rstrip(":"))
            + (r":" if selector.endswith(":") else "")
            + r"[^\]]*\]"
        )
        # Handled below with a simpler state-machine parser.
        pass

    out = []
    i = 0
    n = len(text)
    while i < n:
        start = text.find("[_renderPassManager", i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        j = start + 1  # skip '['
        depth = 1
        while j < n and depth:
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
            j += 1
        call = text[start:j]
        replaced = False
        for selector, repl in METHOD_MAP.items():
            if selector.endswith(":"):
                base = selector[:-1]
                if re.search(r"\[\s*_renderPassManager\s+" + re.escape(base) + r":", call):
                    args = re.sub(
                        r"^\[\s*_renderPassManager\s+" + re.escape(base) + r":\s*",
                        "",
                        call[:-1],
                        count=1,
                    ).strip()
                    out.append(repl + args + ")")
                    replaced = True
                    break
            else:
                if re.search(r"\[\s*_renderPassManager\s+" + re.escape(selector) + r"\s*\]", call):
                    out.append(repl)
                    replaced = True
                    break
        if not replaced:
            out.append(call)
        i = j
    return "".join(out)


def replace_init_shutdown(text: str) -> str:
    text = text.replace(
        "_renderPassManager = [MGLRenderPassManager new];",
        "mglCmdInit(&_commandState);",
    )
    text = text.replace("_renderPassManager = nil;", "")
    text = re.sub(r"!\s*_renderPassManager\b", "NO /* command state embedded */", text)
    text = re.sub(r"_renderPassManager\b(?!\.)", "_commandState", text)
    return text


def transform_file(path: Path) -> bool:
    original = path.read_text()
    text = original
    text = replace_state_access(text)
    text = replace_method_calls(text)
    if path.name == "MGLRenderer+Lifecycle.m":
        text = replace_init_shutdown(text)
    elif "_renderPassManager" in text:
        text = replace_init_shutdown(text)
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> int:
    changed = []
    for path in sorted(MGL_SRC.glob("*.m")):
        if transform_file(path):
            changed.append(path)
    private_h = MGL_INC / "MGLRenderer_Private.h"
    if private_h.exists():
        original = private_h.read_text()
        text = original
        text = text.replace('#import "MGLRenderPassManager.h"', '#include "mgl_render_pass_coordinator.h"')
        text = text.replace("MGLRenderPassManager *_renderPassManager;", "MGLCommandState _commandState;")
        if text != original:
            private_h.write_text(text)
            changed.append(private_h)
    print("Updated:", *[p.relative_to(ROOT) for p in changed], sep="\n  ")
    remaining = []
    for path in list(MGL_SRC.glob("*.m")) + [private_h]:
        if path.exists() and "_renderPassManager" in path.read_text():
            remaining.append(path)
    if remaining:
        print("WARNING: _renderPassManager still present in:", file=sys.stderr)
        for p in remaining:
            print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
