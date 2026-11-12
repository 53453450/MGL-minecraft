/* Software sampler reduction for targets whose native reduction mode is inert. */
/* Fast programs omit this body (cg.softwareSamplerReduction == false).
 * Reduction-capable metallibs always run the select footprint; bind those
 * only when a MIN/MAX sampler is active so ordinary draws stay cheap. */
static llvm::Value *emitSamplerReduction(
    Codegen &cg, const char *samplerName, llvm::Value *arrayIndex,
    llvm::Value *tex, llvm::Value *smp, llvm::Value *coord,
    llvm::Value *arrayLayer, MGLIRTexKind kind, MGLIRScalar texel,
    bool explicitLod, llvm::Value *lodValue, llvm::Value *nativeResult)
{
    if (!cg.softwareSamplerReduction)
        return nativeResult;
    if (!cg.resources.samplerWrapPtr || !tex || !smp || !coord || !nativeResult)
        return nativeResult;
    const bool supported =
        kind == MGLIR_TEX_1D || kind == MGLIR_TEX_1D_ARRAY ||
        kind == MGLIR_TEX_2D || kind == MGLIR_TEX_2D_ARRAY ||
        kind == MGLIR_TEX_3D || kind == MGLIR_TEX_CUBE;
    if (!supported) return nativeResult;

    llvm::Type *i32 = cg.b->getInt32Ty();
    llvm::Type *f32 = cg.b->getFloatTy();
    auto *v4 = llvm::FixedVectorType::get(
        texel == MGLIR_SCALAR_FLOAT ? f32 : i32, 4);
    uint32_t slot = 0;
    auto found = cg.resources.texMetalIndex.find(samplerName ? samplerName : "");
    if (found != cg.resources.texMetalIndex.end()) slot = found->second;
    llvm::Value *slotIndex = arrayIndex
        ? cg.b->CreateAdd(cg.b->getInt32(slot), arrayIndex)
        : cg.b->getInt32(slot);
    llvm::Value *table = cg.b->CreateBitCast(
        cg.resources.samplerWrapPtr, i32->getPointerTo(1));
    llvm::Value *slotBase = cg.b->CreateMul(slotIndex, cg.b->getInt32(13));
    auto tableWord = [&](uint32_t word) {
        return cg.b->CreateAlignedLoad(
            i32, cg.b->CreateGEP(i32, table,
                cg.b->CreateAdd(slotBase, cg.b->getInt32(word))), llvm::Align(4));
    };
    llvm::Value *mode = tableWord(3);
    llvm::Value *filters = tableWord(8);
    llvm::Value *minFilter = cg.b->CreateAnd(filters, cg.b->getInt32(0xffff));
    llvm::Value *magFilter = cg.b->CreateLShr(filters, 16);
    llvm::Value *wrapModes[3] = {tableWord(0), tableWord(1), tableWord(2)};
    llvm::Value *border = llvm::UndefValue::get(v4);
    for (uint32_t c = 0; c < 4; ++c) {
        llvm::Value *component = tableWord(4 + c);
        if (texel == MGLIR_SCALAR_FLOAT) {
            component = cg.b->CreateBitCast(tableWord(9 + c), f32);
        }
        border = cg.b->CreateInsertElement(border, component, c);
    }

    auto component = [&](llvm::Value *v, uint32_t index) -> llvm::Value * {
        if (v->getType()->isVectorTy())
            return cg.b->CreateExtractElement(v, index);
        return index == 0 ? v : llvm::ConstantFP::get(f32, 0.0);
    };
    auto vecFrom = [&](llvm::ArrayRef<llvm::Value *> values) {
        llvm::Type *type = llvm::FixedVectorType::get(f32, values.size());
        llvm::Value *result = llvm::UndefValue::get(type);
        for (size_t i = 0; i < values.size(); ++i)
            result = cg.b->CreateInsertElement(result, values[i], (uint64_t)i);
        return result;
    };

    llvm::Value *sourceCoord = coord;
    llvm::Value *cubeFace = nullptr;
    if (kind == MGLIR_TEX_CUBE) {
        llvm::Value *x = component(coord, 0);
        llvm::Value *y = component(coord, 1);
        llvm::Value *z = component(coord, 2);
        auto abs1 = [&](llvm::Value *v) {
            return cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::fabs, v);
        };
        llvm::Value *ax = abs1(x), *ay = abs1(y), *az = abs1(z);
        llvm::Value *xMajor = cg.b->CreateAnd(
            cg.b->CreateFCmpOGE(ax, ay), cg.b->CreateFCmpOGE(ax, az));
        llvm::Value *yMajor = cg.b->CreateFCmpOGE(ay, az);
        llvm::Value *xPositive = cg.b->CreateFCmpOGE(
            x, llvm::ConstantFP::get(f32, 0.0));
        llvm::Value *yPositive = cg.b->CreateFCmpOGE(
            y, llvm::ConstantFP::get(f32, 0.0));
        llvm::Value *zPositive = cg.b->CreateFCmpOGE(
            z, llvm::ConstantFP::get(f32, 0.0));
        llvm::Value *major = cg.b->CreateSelect(
            xMajor, ax, cg.b->CreateSelect(yMajor, ay, az));
        major = cg.b->CreateSelect(
            cg.b->CreateFCmpONE(major, llvm::ConstantFP::get(f32, 0.0)), major,
            llvm::ConstantFP::get(f32, 1.0));
        llvm::Value *u = cg.b->CreateSelect(
            xMajor, cg.b->CreateSelect(xPositive, cg.b->CreateFNeg(z), z),
            cg.b->CreateSelect(yMajor, x,
                cg.b->CreateSelect(zPositive, x, cg.b->CreateFNeg(x))));
        llvm::Value *v = cg.b->CreateSelect(
            xMajor, cg.b->CreateFNeg(y),
            cg.b->CreateSelect(yMajor,
                cg.b->CreateSelect(yPositive, z, cg.b->CreateFNeg(z)),
                cg.b->CreateFNeg(y)));
        llvm::Value *half = llvm::ConstantFP::get(f32, 0.5);
        llvm::Value *one = llvm::ConstantFP::get(f32, 1.0);
        u = cg.b->CreateFMul(
            cg.b->CreateFAdd(cg.b->CreateFDiv(u, major), one), half);
        v = cg.b->CreateFMul(
            cg.b->CreateFAdd(cg.b->CreateFDiv(v, major), one), half);
        coord = vecFrom({u, v});
        llvm::Value *xFace = cg.b->CreateSelect(
            xPositive, cg.b->getInt32(0), cg.b->getInt32(1));
        llvm::Value *yFace = cg.b->CreateSelect(
            yPositive, cg.b->getInt32(2), cg.b->getInt32(3));
        llvm::Value *zFace = cg.b->CreateSelect(
            zPositive, cg.b->getInt32(4), cg.b->getInt32(5));
        cubeFace = cg.b->CreateSelect(
            xMajor, xFace, cg.b->CreateSelect(yMajor, yFace, zFace));
    }

    const bool fragment = cg.stage == MGL_STAGE_FRAGMENT;
    llvm::Value *lod = llvm::ConstantFP::get(f32, 0.0);
    if (explicitLod && lodValue) {
        lod = coerceScalar(cg, lodValue, MGLIR_SCALAR_FLOAT);
    } else if (fragment) {
        if (kind == MGLIR_TEX_1D || kind == MGLIR_TEX_1D_ARRAY) {
            llvm::Value *x = component(coord, 0);
            llvm::Value *width = cg.b->CreateSIToFP(
                callAirFn(cg, "air.get_width_texture_2d", i32,
                          {tex, cg.b->getInt32(0)}), f32);
            llvm::Value *dx = callAirFn(cg, "air.dfdx.f32", f32, {x});
            llvm::Value *dy = callAirFn(cg, "air.dfdy.f32", f32, {x});
            auto abs1 = [&](llvm::Value *v) {
                return cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::fabs, v);
            };
            llvm::Value *rho = cg.b->CreateIntrinsic(
                llvm::Intrinsic::maxnum, {f32},
                {abs1(cg.b->CreateFMul(dx, width)),
                 abs1(cg.b->CreateFMul(dy, width))});
            lod = cg.b->CreateIntrinsic(llvm::Intrinsic::log2, {f32}, {rho});
        } else {
            const char *suffix = kind == MGLIR_TEX_2D ? "texture_2d"
                : kind == MGLIR_TEX_2D_ARRAY ? "texture_2d_array"
                : kind == MGLIR_TEX_3D ? "texture_3d" : "texture_cube";
            std::string fn = std::string("air.calculate_clamped_lod_") + suffix;
            lod = callAirFn(cg, fn.c_str(), f32,
                            {tex, smp, kind == MGLIR_TEX_CUBE ? sourceCoord : coord,
                             cg.b->getInt32(0)});
        }
    }

    const char *mipSuffix =
        (kind == MGLIR_TEX_1D || kind == MGLIR_TEX_2D) ? "texture_2d"
        : (kind == MGLIR_TEX_1D_ARRAY || kind == MGLIR_TEX_2D_ARRAY)
            ? "texture_2d_array"
        : kind == MGLIR_TEX_3D ? "texture_3d" : "texture_cube";
    llvm::Value *levelCount = callAirFn(
        cg, (std::string("air.get_num_mip_levels_") + mipSuffix).c_str(),
        i32, {tex});
    auto clampLevel = [&](llvm::Value *level) {
        level = cg.b->CreateSelect(cg.b->CreateICmpSGT(level, cg.b->getInt32(0)),
                                   level, cg.b->getInt32(0));
        llvm::Value *last = cg.b->CreateSub(levelCount, cg.b->getInt32(1));
        return cg.b->CreateSelect(cg.b->CreateICmpSGT(level, last), last, level);
    };
    auto floatToLevel = [&](llvm::Value *v) {
        return cg.b->CreateFPToSI(v, i32);
    };
    llvm::Value *floorLod = cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::floor, lod);
    llvm::Value *nearestLod = cg.b->CreateUnaryIntrinsic(
        llvm::Intrinsic::floor,
        cg.b->CreateFAdd(lod, llvm::ConstantFP::get(f32, 0.5)));
    llvm::Value *useMin = explicitLod || !fragment
        ? cg.b->getInt1(true)
        : cg.b->CreateFCmpOGT(lod, llvm::ConstantFP::get(f32, 0.0));
    llvm::Value *filter = cg.b->CreateSelect(useMin, minFilter, magFilter);
    auto filterIs = [&](GLint value) {
        return cg.b->CreateICmpEQ(filter, cg.b->getInt32(value));
    };
    llvm::Value *hasMips = cg.b->CreateOr(
        cg.b->CreateOr(filterIs(GL_NEAREST_MIPMAP_NEAREST),
                       filterIs(GL_LINEAR_MIPMAP_NEAREST)),
        cg.b->CreateOr(filterIs(GL_NEAREST_MIPMAP_LINEAR),
                       filterIs(GL_LINEAR_MIPMAP_LINEAR)));
    llvm::Value *linear = cg.b->CreateOr(
        cg.b->CreateOr(filterIs(GL_LINEAR), filterIs(GL_LINEAR_MIPMAP_NEAREST)),
        filterIs(GL_LINEAR_MIPMAP_LINEAR));
    llvm::Value *mipLinear = cg.b->CreateOr(
        filterIs(GL_NEAREST_MIPMAP_LINEAR), filterIs(GL_LINEAR_MIPMAP_LINEAR));
    llvm::Value *levelA = cg.b->CreateSelect(
        cg.b->CreateAnd(useMin, hasMips),
        cg.b->CreateSelect(mipLinear, floatToLevel(floorLod),
                           floatToLevel(nearestLod)), cg.b->getInt32(0));
    levelA = clampLevel(levelA);
    llvm::Value *levelB = cg.b->CreateSelect(
        cg.b->CreateAnd(useMin, mipLinear),
        cg.b->CreateAdd(floatToLevel(floorLod), cg.b->getInt32(1)), levelA);
    levelB = clampLevel(levelB);

    llvm::StructType *samplerType = llvm::StructType::getTypeByName(
        *cg.ctx, "struct._sampler_t");
    if (!samplerType)
        samplerType = llvm::StructType::create(*cg.ctx, "struct._sampler_t");
    llvm::Value *readSampler = callAirFn(
        cg, "air.get_read_sampler", samplerType->getPointerTo(2), {});
    auto *readRet = llvm::StructType::get(*cg.ctx, {v4, cg.b->getInt8Ty()});
    const char *readSuffix = kind == MGLIR_TEX_3D ? "texture_3d" :
        (kind == MGLIR_TEX_1D || kind == MGLIR_TEX_2D) ? "texture_2d" :
        kind == MGLIR_TEX_CUBE ? "texture_cube" : "texture_2d_array";
    std::string readName = std::string("air.read_") + readSuffix +
        (texel == MGLIR_SCALAR_FLOAT ? ".v4f32"
         : texel == MGLIR_SCALAR_INT ? ".s.v4i32" : ".u.v4i32");

    auto reduce = [&](llvm::Value *a, llvm::Value *b, bool isMin) -> llvm::Value * {
        if (texel == MGLIR_SCALAR_FLOAT) {
            return cg.b->CreateIntrinsic(isMin ? llvm::Intrinsic::minnum
                                               : llvm::Intrinsic::maxnum,
                                         {a->getType()}, {a, b});
        }
        llvm::CmpInst::Predicate pred;
        if (isMin)
            pred = texel == MGLIR_SCALAR_UINT ? llvm::CmpInst::ICMP_ULT
                                               : llvm::CmpInst::ICMP_SLT;
        else
            pred = texel == MGLIR_SCALAR_UINT ? llvm::CmpInst::ICMP_UGT
                                               : llvm::CmpInst::ICMP_SGT;
        return cg.b->CreateSelect(cg.b->CreateICmp(pred, a, b), a, b);
    };

    auto sampleLevel = [&](llvm::Value *level) -> llvm::Value * {
        uint32_t dims = (kind == MGLIR_TEX_1D || kind == MGLIR_TEX_1D_ARRAY) ? 1u
            : (kind == MGLIR_TEX_2D || kind == MGLIR_TEX_2D_ARRAY ||
               kind == MGLIR_TEX_CUBE) ? 2u : 3u;
        llvm::Value *sizes[3] = {nullptr, nullptr, nullptr};
        llvm::Value *coords[3] = {nullptr, nullptr, nullptr};
        for (uint32_t axis = 0; axis < dims; ++axis) {
            const char *query = axis == 0 ? "width" : axis == 1 ? "height" : "depth";
            sizes[axis] = callAirFn(
                cg, (std::string("air.get_") + query + "_" + mipSuffix).c_str(),
                i32, {tex, level});
            coords[axis] = component(coord, axis);
        }
        auto nearestIndex = [&](uint32_t axis) {
            llvm::Value *p = cg.b->CreateFMul(
                coords[axis], cg.b->CreateSIToFP(sizes[axis], f32));
            return cg.b->CreateFPToSI(
                cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::floor, p), i32);
        };
        auto linearIndex = [&](uint32_t axis) {
            llvm::Value *p = cg.b->CreateFSub(
                cg.b->CreateFMul(coords[axis],
                                 cg.b->CreateSIToFP(sizes[axis], f32)),
                llvm::ConstantFP::get(f32, 0.5));
            return cg.b->CreateFPToSI(
                cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::floor, p), i32);
        };
        auto readAt = [&](llvm::Value *const *indices) -> llvm::Value * {
            llvm::Value *wrapped[3] = {nullptr, nullptr, nullptr};
            llvm::Value *oob = cg.b->getInt1(false);
            for (uint32_t axis = 0; axis < dims; ++axis) {
                llvm::Value *t = indices[axis];
                llvm::Value *n = sizes[axis];
                llvm::Value *wrap = wrapModes[axis];
                llvm::Value *zero = cg.b->getInt32(0);
                llvm::Value *n1 = cg.b->CreateSub(n, cg.b->getInt32(1));
                llvm::Value *nSafe = cg.b->CreateSelect(
                    cg.b->CreateICmpSGT(n, zero), n, cg.b->getInt32(1));
                llvm::Value *rem = cg.b->CreateSRem(t, nSafe);
                rem = cg.b->CreateSelect(cg.b->CreateICmpSLT(rem, zero),
                                         cg.b->CreateAdd(rem, nSafe), rem);
                llvm::Value *period = cg.b->CreateMul(nSafe, cg.b->getInt32(2));
                llvm::Value *mirror = cg.b->CreateSRem(t, period);
                mirror = cg.b->CreateSelect(cg.b->CreateICmpSLT(mirror, zero),
                    cg.b->CreateAdd(mirror, period), mirror);
                mirror = cg.b->CreateSelect(cg.b->CreateICmpSGE(mirror, nSafe),
                    cg.b->CreateSub(cg.b->CreateSub(period, mirror),
                                    cg.b->getInt32(1)), mirror);
                llvm::Value *edge = cg.b->CreateSelect(
                    cg.b->CreateICmpSLT(t, zero), zero, t);
                edge = cg.b->CreateSelect(cg.b->CreateICmpSGT(edge, n1), n1, edge);
                llvm::Value *mirrorEdge = cg.b->CreateSelect(
                    cg.b->CreateICmpSLT(t, zero),
                    cg.b->CreateSub(cg.b->CreateNeg(t), cg.b->getInt32(1)), t);
                mirrorEdge = cg.b->CreateSelect(
                    cg.b->CreateICmpSGT(mirrorEdge, n1), n1, mirrorEdge);
                llvm::Value *axisOob = cg.b->CreateOr(
                    cg.b->CreateICmpSLT(t, zero), cg.b->CreateICmpSGE(t, n));
                llvm::Value *isBorder = cg.b->CreateICmpEQ(
                    wrap, cg.b->getInt32(GL_CLAMP_TO_BORDER));
                oob = cg.b->CreateOr(oob, cg.b->CreateAnd(isBorder, axisOob));
                llvm::Value *result = edge;
                result = cg.b->CreateSelect(cg.b->CreateICmpEQ(
                    wrap, cg.b->getInt32(GL_REPEAT)), rem, result);
                result = cg.b->CreateSelect(cg.b->CreateICmpEQ(
                    wrap, cg.b->getInt32(GL_MIRRORED_REPEAT)), mirror, result);
                result = cg.b->CreateSelect(cg.b->CreateICmpEQ(
                    wrap, cg.b->getInt32(GL_MIRROR_CLAMP_TO_EDGE)), mirrorEdge, result);
                wrapped[axis] = result;
            }
            std::vector<llvm::Value *> args = {tex, readSampler};
            const uint32_t readDims =
                (kind == MGLIR_TEX_1D || kind == MGLIR_TEX_1D_ARRAY) ? 2u : dims;
            llvm::Value *xyz = llvm::UndefValue::get(
                llvm::FixedVectorType::get(i32, readDims));
            for (uint32_t axis = 0; axis < readDims; ++axis) {
                llvm::Value *index = axis < dims ? wrapped[axis] : cg.b->getInt32(0);
                xyz = cg.b->CreateInsertElement(xyz, index, axis);
            }
            args.push_back(xyz);
            if (kind == MGLIR_TEX_1D_ARRAY || kind == MGLIR_TEX_2D_ARRAY)
                args.push_back(arrayLayer ? arrayLayer : cg.b->getInt32(0));
            if (kind == MGLIR_TEX_CUBE) args.push_back(cubeFace);
            if (readDims > 1 && kind != MGLIR_TEX_CUBE)
                args.push_back(llvm::Constant::getNullValue(
                    llvm::FixedVectorType::get(i32, readDims)));
            args.push_back(level);
            args.push_back(cg.b->getInt32(0));
            llvm::Value *read = callAirFn(cg, readName.c_str(), readRet, args);
            llvm::Value *value = cg.b->CreateExtractValue(read, 0);
            return cg.b->CreateSelect(oob, border, value);
        };
        llvm::Value *nearestIndices[3] = {nullptr, nullptr, nullptr};
        llvm::Value *linearIndices[3] = {nullptr, nullptr, nullptr};
        llvm::Value *linearFract[3] = {nullptr, nullptr, nullptr};
        for (uint32_t axis = 0; axis < dims; ++axis) {
            nearestIndices[axis] = nearestIndex(axis);
            linearIndices[axis] = linearIndex(axis);
            llvm::Value *p = cg.b->CreateFSub(
                cg.b->CreateFMul(coords[axis],
                                 cg.b->CreateSIToFP(sizes[axis], f32)),
                llvm::ConstantFP::get(f32, 0.5));
            linearFract[axis] = cg.b->CreateFSub(
                p, cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::floor, p));
        }
        llvm::Value *nearest = readAt(nearestIndices);
        llvm::Value *lo = nullptr, *hi = nullptr, *average = nullptr;
        llvm::Value *taps[8] = {nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr, nullptr};
        for (uint32_t item = 0; item < (1u << dims); ++item) {
            llvm::Value *indices[3] = {nullptr, nullptr, nullptr};
            for (uint32_t axis = 0; axis < dims; ++axis)
                indices[axis] = cg.b->CreateAdd(
                    linearIndices[axis], cg.b->getInt32((item >> axis) & 1u));
            llvm::Value *value = readAt(indices);
            taps[item] = value;
            lo = item == 0 ? value : reduce(lo, value, true);
            hi = item == 0 ? value : reduce(hi, value, false);
        }
        if (kind == MGLIR_TEX_CUBE && texel == MGLIR_SCALAR_FLOAT) {
            auto lerp = [&](llvm::Value *v0, llvm::Value *v1, llvm::Value *f) {
                llvm::Value *fv = cg.b->CreateVectorSplat(4, f);
                return cg.b->CreateFAdd(
                    v0, cg.b->CreateFMul(cg.b->CreateFSub(v1, v0), fv));
            };
            llvm::Value *row0 = lerp(taps[0], taps[1], linearFract[0]);
            llvm::Value *row1 = lerp(taps[2], taps[3], linearFract[0]);
            average = lerp(row0, row1, linearFract[1]);
        }
        if (!average) average = nearest;
        llvm::Value *linearValue = cg.b->CreateSelect(
            cg.b->CreateICmpEQ(mode, cg.b->getInt32(1)), lo,
            cg.b->CreateSelect(cg.b->CreateICmpEQ(mode, cg.b->getInt32(2)),
                               hi, average));
        return cg.b->CreateSelect(linear, linearValue, nearest);
    };

    llvm::Value *a = sampleLevel(levelA);
    llvm::Value *b = sampleLevel(levelB);
    llvm::Value *lo = reduce(a, b, true);
    llvm::Value *hi = reduce(a, b, false);
    llvm::Value *software = cg.b->CreateSelect(
        cg.b->CreateICmpEQ(mode, cg.b->getInt32(1)), lo, hi);
    if (kind == MGLIR_TEX_CUBE && texel == MGLIR_SCALAR_FLOAT) {
        llvm::Value *fraction = cg.b->CreateFSub(lod, floorLod);
        nativeResult = cg.b->CreateSelect(
            mipLinear, cg.b->CreateFAdd(
                a, cg.b->CreateFMul(cg.b->CreateFSub(b, a),
                                    cg.b->CreateVectorSplat(4, fraction))), a);
    }
    return cg.b->CreateSelect(cg.b->CreateICmpEQ(mode, cg.b->getInt32(0)),
                              nativeResult, software);
}

