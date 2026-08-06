/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * mgl_metallib_writer.cpp
 * MGL - .metallib container serializer (see mgl_metallib_writer.h).
 *
 * Layout, reverse-engineered from `xcrun metallib` (macOS 26 SDK) and
 * validated with newLibraryWithData + PSO creation:
 *
 *   header (88 bytes)
 *   u32 function count
 *   u32 byte length of record 0
 *   record 0 .. record N-1, each followed by u32(byte length of the next
 *     record); the final record is followed by the "ENDT" fourcc instead
 *   per-function public metadata  (u32 len + payload, N entries)
 *   per-function private metadata (u32 len + payload, N entries)
 *   per-function bitcode blobs, concatenated
 *
 * Each bitcode blob must be a standalone LLVM module for one entry
 * function (the loader resolves each function's blob via MDSZ/OFFT).
 */

#include "mgl_metallib_writer.h"

#include <cstring>

namespace mgl {

namespace {

/* SHA-256 (FIPS 180-4), self-contained; the writer has no external
 * crypto dependency. */
struct Sha256 {
    uint32_t h[8];
    uint8_t buf[64];
    uint64_t total;

    static uint32_t rotr(uint32_t x, unsigned n) {
        return (x >> n) | (x << (32 - n));
    }

    void init() {
        static const uint32_t IV[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                       0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                       0x1f83d9abu, 0x5be0cd19u};
        std::memcpy(h, IV, sizeof(h));
        total = 0;
    }

    void block(const uint8_t *p) {
        static const uint32_t K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
            0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
            0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
            0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
            0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
            0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
            0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
            0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
            0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (unsigned i = 0; i < 16; i++) {
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
                   (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
        }
        for (unsigned i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
                          (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
                          (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (unsigned i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const uint8_t *p, size_t n) {
        while (n > 0) {
            size_t fill = 64 - (total % 64);
            if (fill > n) {
                fill = n;
            }
            std::memcpy(buf + (total % 64), p, fill);
            total += fill;
            p += fill;
            n -= fill;
            if (total % 64 == 0) {
                block(buf);
            }
        }
    }

    void final(uint8_t out[32]) {
        uint64_t bits = total * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (total % 64 != 56) {
            update(&zero, 1);
        }
        for (int i = 7; i >= 0; i--) {
            uint8_t b = uint8_t(bits >> (8 * (7 - i)));
            update(&b, 1);
        }
        for (unsigned i = 0; i < 8; i++) {
            out[4 * i] = uint8_t(h[i] >> 24);
            out[4 * i + 1] = uint8_t(h[i] >> 16);
            out[4 * i + 2] = uint8_t(h[i] >> 8);
            out[4 * i + 3] = uint8_t(h[i]);
        }
    }
};

void sha256(const std::vector<uint8_t> &data, uint8_t out[32]) {
    Sha256 s;
    s.init();
    s.update(data.data(), data.size());
    s.final(out);
}

struct TagWriter {
    llvm::raw_ostream &os;
    void fourcc(const char *tag) { os.write(tag, 4); }
    void u16(uint16_t v) { os.write(reinterpret_cast<const char *>(&v), 2); }
    void u32(uint32_t v) { os.write(reinterpret_cast<const char *>(&v), 4); }
    void u64(uint64_t v) { os.write(reinterpret_cast<const char *>(&v), 8); }
};

/* Tags only (NAME..ENDT); the trailing next-record length is appended by
 * the caller.  OFFT carries the blob's bitcode-section offset and the
 * per-function public/private metadata offsets. */
void writeRecordTags(TagWriter &w, const MTLBFunction &fn,
                     uint64_t bitcodeOffset, uint64_t pubOffset,
                     uint64_t privOffset) {
    uint16_t nameLen = uint16_t(fn.name.size() + 1);
    w.fourcc("NAME");
    w.u16(nameLen);
    w.os.write(fn.name.data(), fn.name.size());
    w.os.write('\0');

    w.fourcc("TYPE");
    w.u16(1);
    w.os.write(static_cast<char>(fn.type));

    uint8_t hash[32];
    sha256(fn.bitcode, hash);
    w.fourcc("HASH");
    w.u16(0x20);
    w.os.write(reinterpret_cast<const char *>(hash), 32);

    w.fourcc("MDSZ");
    w.u16(8);
    w.u64(fn.bitcode.size());

    w.fourcc("OFFT");
    w.u16(24);
    w.u64(pubOffset);
    w.u64(privOffset);
    w.u64(bitcodeOffset);

    /* Container versions must agree with the bitcode module's
     * !air.version (2.8) and !air.language_version (Metal 4.0). */
    w.fourcc("VERS");
    w.u16(8);
    w.u16(2); /* air major */
    w.u16(8); /* air minor */
    w.u16(4); /* language major */
    w.u16(0); /* language minor */

    w.fourcc("ENDT");
}

} /* namespace */

void mglMTLBWrite(const std::vector<MTLBFunction> &fns, llvm::raw_ostream &os) {
    const size_t n = fns.size();

    /* Per-function public/private metadata: empty blocks in this backend
     * (vertex attribute VATT/VATY blocks are emitted later if needed). */
    std::vector<uint8_t> pub, priv;
    for (size_t i = 0; i < n; i++) {
        const uint32_t len = 4;
        const char endt[4] = {'E', 'N', 'D', 'T'};
        pub.insert(pub.end(), reinterpret_cast<const char *>(&len),
                   reinterpret_cast<const char *>(&len) + 4);
        pub.insert(pub.end(), endt, endt + 4);
        priv.insert(priv.end(), reinterpret_cast<const char *>(&len),
                    reinterpret_cast<const char *>(&len) + 4);
        priv.insert(priv.end(), endt, endt + 4);
    }

    /* Function records.  Each record is tags + 4-byte tail; the tail of
     * record i holds the byte length of record i+1 (which is
     * tags_i+1 + 8: its own tags + its tail), and the final record's
     * tail is the "ENDT" fourcc.  Tag sizes are measured in a throwaway
     * pass first so the tails can be interleaved correctly. */
    std::vector<uint32_t> tagSizes(n);
    {
        llvm::SmallVector<char, 0> scratch;
        llvm::raw_svector_ostream stream(scratch);
        TagWriter w{stream};
        for (size_t i = 0; i < n; i++) {
            uint32_t before = uint32_t(scratch.size());
            writeRecordTags(w, fns[i], 0, 0, 0);
            tagSizes[i] = uint32_t(scratch.size()) - before;
        }
    }
    llvm::SmallVector<char, 0> defs;
    {
        llvm::raw_svector_ostream stream(defs);
        TagWriter w{stream};
        uint64_t bcOff = 0;
        for (size_t i = 0; i < n; i++) {
            writeRecordTags(w, fns[i], bcOff, 8 * i, 8 * i);
            if (i + 1 < n) {
                uint32_t nextLen = tagSizes[i + 1] + 8;
                w.u32(nextLen);
            } else {
                const char endt[4] = {'E', 'N', 'D', 'T'};
                w.os.write(endt, 4);
            }
            bcOff += fns[i].bitcode.size();
        }
    }

    uint64_t bitcodeTotal = 0;
    for (const auto &fn : fns) {
        bitcodeTotal += fn.bitcode.size();
    }

    /* Header. */
    const uint64_t fnListOffset = 88;
    const uint64_t fnListSize = defs.size();
    const uint64_t pubOffset = fnListOffset + 4 + 4 + fnListSize;
    const uint64_t privOffset = pubOffset + pub.size();
    const uint64_t bcOffset = privOffset + priv.size();

    os.write("MTLB", 4);
    uint16_t platform = 0x8001, vMajor = 2, vMinor = 9;
    uint8_t type = 0, target = 0x81;
    uint16_t osMajor = 26, osMinor = 0;
    os.write(reinterpret_cast<const char *>(&platform), 2);
    os.write(reinterpret_cast<const char *>(&vMajor), 2);
    os.write(reinterpret_cast<const char *>(&vMinor), 2);
    os.write(reinterpret_cast<const char *>(&type), 1);
    os.write(reinterpret_cast<const char *>(&target), 1);
    os.write(reinterpret_cast<const char *>(&osMajor), 2);
    os.write(reinterpret_cast<const char *>(&osMinor), 2);

    uint64_t fileSize = bcOffset + bitcodeTotal;
    os.write(reinterpret_cast<const char *>(&fileSize), 8);
    os.write(reinterpret_cast<const char *>(&fnListOffset), 8);
    os.write(reinterpret_cast<const char *>(&fnListSize), 8);
    os.write(reinterpret_cast<const char *>(&pubOffset), 8);
    uint64_t pubSize = pub.size();
    os.write(reinterpret_cast<const char *>(&pubSize), 8);
    os.write(reinterpret_cast<const char *>(&privOffset), 8);
    uint64_t privSize = priv.size();
    os.write(reinterpret_cast<const char *>(&privSize), 8);
    os.write(reinterpret_cast<const char *>(&bcOffset), 8);
    os.write(reinterpret_cast<const char *>(&bitcodeTotal), 8);

    /* Function list: count + first record length + records. */
    uint32_t count = uint32_t(n);
    os.write(reinterpret_cast<const char *>(&count), 4);
    uint32_t rec0Len = tagSizes.empty() ? 0 : tagSizes[0] + 8;
    os.write(reinterpret_cast<const char *>(&rec0Len), 4);
    os.write(defs.data(), defs.size());

    os.write(reinterpret_cast<const char *>(pub.data()), pub.size());
    os.write(reinterpret_cast<const char *>(priv.data()), priv.size());
    for (const auto &fn : fns) {
        os.write(reinterpret_cast<const char *>(fn.bitcode.data()),
                 fn.bitcode.size());
    }
}

} /* namespace mgl */
