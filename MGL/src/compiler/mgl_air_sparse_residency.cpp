#include "mgl_air_sparse_residency.h"

#include "glm_limits.h"
#include "mgl_sparse_residency.h"

namespace mgl {
namespace air {
namespace {

llvm::Value *loadWord(Codegen &cg, llvm::Value *base, uint64_t byte_offset,
                      const char *name) {
    llvm::Type *i8 = cg.b->getInt8Ty();
    llvm::Type *i32 = cg.b->getInt32Ty();
    llvm::Value *address = cg.b->CreateGEP(
        i8, base, cg.b->getInt64(byte_offset));
    address = cg.b->CreateBitCast(address, i32->getPointerTo(1));
    return cg.b->CreateAlignedLoad(i32, address, llvm::Align(4), name);
}

llvm::Value *loadWordAt(Codegen &cg, llvm::Value *base, llvm::Value *index,
                        uint64_t stride, uint64_t member_offset,
                        const char *name) {
    llvm::Type *i8 = cg.b->getInt8Ty();
    llvm::Value *byte_index = cg.b->CreateMul(index, cg.b->getInt64(stride));
    if (member_offset) {
        byte_index = cg.b->CreateAdd(byte_index, cg.b->getInt64(member_offset));
    }
    llvm::Value *address = cg.b->CreateGEP(i8, base, byte_index);
    llvm::Type *i32 = cg.b->getInt32Ty();
    address = cg.b->CreateBitCast(address, i32->getPointerTo(1));
    return cg.b->CreateAlignedLoad(i32, address,
                                   llvm::Align(4), name);
}

} // namespace

llvm::Value *sparseCommitResidency(Codegen &cg, llvm::Value *metal_slot,
                                   llvm::Value *relative_level,
                                   llvm::Value *x, llvm::Value *y,
                                   llvm::Value *z)
{
    llvm::Value *pack = cg.resources.sparseResidencyPtr;
    if (!pack || !metal_slot || !relative_level || !x || !y || !z) return nullptr;
    llvm::Type *i32 = cg.b->getInt32Ty();
    llvm::Value *slot_base = cg.b->CreateGEP(
        cg.b->getInt8Ty(), pack,
        cg.b->CreateMul(cg.b->CreateZExt(metal_slot, cg.b->getInt64Ty()),
                        cg.b->getInt64(sizeof(MGLSparseResidencySlot))));
    llvm::Value *first = loadWord(cg, slot_base,
        offsetof(MGLSparseResidencySlot, first_rect), "sparse.first");
    llvm::Value *count = loadWord(cg, slot_base,
        offsetof(MGLSparseResidencySlot, rect_count), "sparse.count");
    llvm::Value *level_origin = loadWord(cg, slot_base,
        offsetof(MGLSparseResidencySlot, level_origin), "sparse.level_origin");
    llvm::Value *layer_origin = loadWord(cg, slot_base,
        offsetof(MGLSparseResidencySlot, layer_origin), "sparse.layer_origin");
    llvm::Value *flags = loadWord(cg, slot_base,
        offsetof(MGLSparseResidencySlot, flags), "sparse.flags");
    llvm::Value *absolute_level = cg.b->CreateAdd(relative_level, level_origin);
    llvm::Value *absolute_z = cg.b->CreateAdd(z, layer_origin);

    llvm::BasicBlock *entry = cg.b->GetInsertBlock();
    llvm::BasicBlock *loop = llvm::BasicBlock::Create(*cg.ctx, "sparse.rect.loop", cg.fn);
    llvm::BasicBlock *body = llvm::BasicBlock::Create(*cg.ctx, "sparse.rect.body", cg.fn);
    llvm::BasicBlock *next = llvm::BasicBlock::Create(*cg.ctx, "sparse.rect.next", cg.fn);
    llvm::BasicBlock *done = llvm::BasicBlock::Create(*cg.ctx, "sparse.rect.done", cg.fn);

    llvm::Value *active = cg.b->CreateICmpNE(
        cg.b->CreateAnd(flags, cg.b->getInt32(MGL_SPARSE_RESIDENCY_ACTIVE)),
        cg.b->getInt32(0));
    cg.b->CreateCondBr(active, loop, done);

    cg.b->SetInsertPoint(loop);
    llvm::PHINode *index = cg.b->CreatePHI(i32, 3, "sparse.rect.index");
    index->addIncoming(cg.b->getInt32(0), entry);
    llvm::Value *more = cg.b->CreateICmpULT(index, count);
    cg.b->CreateCondBr(more, body, done);

    cg.b->SetInsertPoint(body);
    llvm::Value *rect_index = cg.b->CreateAdd(first, index);
    llvm::Value *rect_base = cg.b->CreateGEP(
        cg.b->getInt8Ty(), pack,
        cg.b->CreateAdd(
            cg.b->getInt64((uint64_t)TEXTURE_UNITS * sizeof(MGLSparseResidencySlot)),
            cg.b->CreateMul(cg.b->CreateZExt(rect_index, cg.b->getInt64Ty()),
                            cg.b->getInt64(sizeof(MGLSparseResidencyRect)))));
    llvm::Value *rect_level = loadWordAt(cg, rect_base, cg.b->getInt64(0), 0,
        offsetof(MGLSparseResidencyRect, level), "sparse.rect.level");
    llvm::Value *rx = loadWordAt(cg, rect_base, cg.b->getInt64(0), 0,
        offsetof(MGLSparseResidencyRect, x), "sparse.rect.x");
    llvm::Value *ry = loadWordAt(cg, rect_base, cg.b->getInt64(0), 0,
        offsetof(MGLSparseResidencyRect, y), "sparse.rect.y");
    llvm::Value *rz = loadWordAt(cg, rect_base, cg.b->getInt64(0), 0,
        offsetof(MGLSparseResidencyRect, z), "sparse.rect.z");
    llvm::Value *rw = loadWordAt(cg, rect_base, cg.b->getInt64(0), 0,
        offsetof(MGLSparseResidencyRect, w), "sparse.rect.w");
    llvm::Value *rh = loadWordAt(cg, rect_base, cg.b->getInt64(0), 0,
        offsetof(MGLSparseResidencyRect, h), "sparse.rect.h");
    llvm::Value *rd = loadWordAt(cg, rect_base, cg.b->getInt64(0), 0,
        offsetof(MGLSparseResidencyRect, d), "sparse.rect.d");
    llvm::Value *match = cg.b->CreateICmpEQ(absolute_level, rect_level);
    llvm::Value *tests[3] = {
        cg.b->CreateAnd(cg.b->CreateICmpSGE(x, rx),
                        cg.b->CreateICmpSLT(x, cg.b->CreateAdd(rx, rw))),
        cg.b->CreateAnd(cg.b->CreateICmpSGE(y, ry),
                        cg.b->CreateICmpSLT(y, cg.b->CreateAdd(ry, rh))),
        cg.b->CreateAnd(cg.b->CreateICmpSGE(absolute_z, rz),
                        cg.b->CreateICmpSLT(absolute_z, cg.b->CreateAdd(rz, rd))),
    };
    match = cg.b->CreateAnd(match, tests[0]);
    match = cg.b->CreateAnd(match, tests[1]);
    match = cg.b->CreateAnd(match, tests[2]);
    cg.b->CreateCondBr(match, done, next);

    cg.b->SetInsertPoint(next);
    llvm::Value *increment = cg.b->CreateAdd(index, cg.b->getInt32(1));
    index->addIncoming(increment, next);
    cg.b->CreateBr(loop);

    cg.b->SetInsertPoint(done);
    llvm::PHINode *resident = cg.b->CreatePHI(i32, 4, "sparse.resident");
    resident->addIncoming(cg.b->getInt32(1), entry); // non-sparse slot
    resident->addIncoming(cg.b->getInt32(0), loop);  // exhausted active list
    resident->addIncoming(cg.b->getInt32(1), body);  // containing rect
    return resident;
}

llvm::Value *sparseCommitResidency(Codegen &cg, uint32_t metal_slot,
                                   llvm::Value *relative_level,
                                   llvm::Value *x, llvm::Value *y,
                                   llvm::Value *z)
{
    return sparseCommitResidency(cg, cg.b->getInt32(metal_slot),
                                 relative_level, x, y, z);
}

llvm::Value *sparseSampleCount(Codegen &cg, uint32_t metal_slot)
{
    llvm::Value *pack = cg.resources.sparseResidencyPtr;
    if (!pack) return nullptr;
    llvm::Value *slot_base = cg.b->CreateGEP(
        cg.b->getInt8Ty(), pack,
        cg.b->getInt64((uint64_t)metal_slot * sizeof(MGLSparseResidencySlot)));
    return loadWord(cg, slot_base,
                    offsetof(MGLSparseResidencySlot, sample_count),
                    "sparse.sample_count");
}

} // namespace air
} // namespace mgl
