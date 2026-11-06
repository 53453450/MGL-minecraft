/* SPDX-License-Identifier: LGPL-3.0-only */

#ifndef MGL_AIR_CONTROL_FLOW_H
#define MGL_AIR_CONTROL_FLOW_H

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class BasicBlock;
class PHINode;
class Value;
}

namespace mgl {
namespace air {

struct LoopCtx {
    llvm::BasicBlock *condBB = nullptr;  /* do-while continue target */
    llvm::BasicBlock *endBB = nullptr;   /* break target */
    llvm::BasicBlock *incrBB = nullptr;  /* merge block; while/for continue target */
    llvm::BasicBlock *condExitBB = nullptr; /* false-condition exit (after side effects) */
    std::map<std::string, llvm::Value *> condExitSnap;
    std::map<std::string, llvm::PHINode *> phis;
    std::vector<std::pair<llvm::BasicBlock *,
                          std::map<std::string, llvm::Value *>>> contSnaps;
};

/* Shared by loops and switch: break jumps to endBB carrying a snapshot
 * of the live values; the owner merges them into phis at endBB. */
struct BreakCtx {
    llvm::BasicBlock *endBB;
    std::vector<std::pair<llvm::BasicBlock *,
                          std::map<std::string, llvm::Value *>>> snaps;
};

struct ControlFlowState {
public:
    void pushLoop(LoopCtx *loop, BreakCtx *breaks) {
        loopStack.push_back(loop);
        breakStack.push_back(breaks);
    }
    void popLoop() {
        loopStack.pop_back();
        breakStack.pop_back();
    }
    void pushBreak(BreakCtx *breaks) { breakStack.push_back(breaks); }
    void popBreak() { breakStack.pop_back(); }

    LoopCtx *currentLoop() {
        return loopStack.empty() ? nullptr : loopStack.back();
    }
    BreakCtx *currentBreak() {
        return breakStack.empty() ? nullptr : breakStack.back();
    }

private:
    std::vector<LoopCtx *> loopStack;
    std::vector<BreakCtx *> breakStack;
};

} // namespace air
} // namespace mgl

#endif
