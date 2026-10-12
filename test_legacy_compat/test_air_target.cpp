/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * test_air_target.cpp
 *
 * Golden for the AIR / Metal / MTLB container version contract.
 *
 * The target reaches Metal through two serializers that never see each other:
 * the LLVM module metadata (target triple, !air.version,
 * !air.language_version) and the MTLB container (file header, per-function
 * VERS tag).  Both now derive from AirTarget, and this test checks the two
 * *artifacts* - the metadata an LLVM module actually carries, and the bytes
 * mglMTLBWrite actually emits - against the expected numbers spelled out
 * here, plus the cross-artifact equalities Metal enforces.
 *
 * The negatives matter as much as the positives: a mutated copy of the target
 * has to be rejected by the consistency check and has to change both
 * artifacts.  Without that, a hard-coded literal could satisfy the golden
 * while the derivation was gone.
 *
 * Links mgl_air_target.cpp + mgl_metallib_writer.cpp + LLVM only - no Metal
 * device, no backend, no frontend.  Runs anywhere llvm@15 does.
 */

#include "mgl_air_target.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "mgl_metallib_writer.h"

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, label)                                    \
    do {                                                      \
        tests_run++;                                          \
        if (cond) {                                           \
            tests_passed++;                                   \
            printf("  [PASS] %s\n", (label));                 \
        } else {                                              \
            printf("  [FAIL] %s\n", (label));                 \
        }                                                     \
    } while (0)

namespace {

/* The expected target, written out once.  This is the only place outside
 * mgl_air_target.cpp where the numbers appear, and it is a test - editing the
 * production values has to fail here. */
const char *const kExpectedTriple = "air64_v28-apple-macosx26.0.0";
const uint32_t kExpectedAirVersion[3] = {2u, 8u, 0u};
const char *const kExpectedLanguage = "Metal";
const uint32_t kExpectedLanguageVersion[3] = {4u, 0u, 0u};
const uint32_t kExpectedContainerFormat[2] = {2u, 9u};
const uint32_t kExpectedOs[2] = {26u, 0u};
const uint32_t kExpectedPlatform = 0x8001u;
const uint32_t kExpectedTargetField = 0x81u;

uint16_t readU16(const std::vector<uint8_t> &b, size_t off) {
    return uint16_t(b[off]) | uint16_t(uint16_t(b[off + 1]) << 8);
}

/* The container is little-endian throughout (TagWriter::u16 writes the host
 * bytes), so the readers above are spelled out rather than memcpy'd - the test
 * must not inherit the writer's endianness assumption silently. */

bool findFourcc(const std::vector<uint8_t> &b, const char *tag, size_t *out) {
    for (size_t i = 0; i + 4u <= b.size(); i++) {
        if (std::memcmp(b.data() + i, tag, 4) == 0) {
            *out = i;
            return true;
        }
    }
    return false;
}

std::vector<uint8_t> writeContainer(void) {
    mgl::MTLBFunction fn;
    fn.name = "main";
    fn.type = mgl::MTLB_FN_VERTEX;
    fn.tessellation = 0;
    fn.bitcode = {0u, 1u, 2u, 3u}; /* opaque; the container only sizes it */

    std::vector<mgl::MTLBFunction> fns{fn};
    std::string out;
    llvm::raw_string_ostream os(out);
    mgl::mglMTLBWrite(fns, os);
    os.flush();
    return std::vector<uint8_t>(out.begin(), out.end());
}

uint32_t readMDInt(const llvm::MDNode *node, unsigned index) {
    auto *c = llvm::dyn_cast_or_null<llvm::ConstantAsMetadata>(
        node->getOperand(index).get());
    if (!c) return 0xFFFFFFFFu;
    return uint32_t(llvm::cast<llvm::ConstantInt>(c->getValue())
                        ->getZExtValue());
}

std::string readMDString(const llvm::MDNode *node, unsigned index) {
    auto *s = llvm::dyn_cast_or_null<llvm::MDString>(
        node->getOperand(index).get());
    return s ? s->getString().str() : std::string("<none>");
}

/* Applies AirTarget to a fresh module and reports what the module carries, so
 * the assertions below read the artifact rather than the source struct. */
struct ModuleFacts {
    std::string triple;
    uint32_t air[3] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    std::string language;
    uint32_t language_version[3] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
};

ModuleFacts compileModuleFacts(const mgl::AirTarget &t) {
    llvm::LLVMContext ctx;
    llvm::Module module("mgl_air_target_probe", ctx);
    module.setTargetTriple(mgl::mglAirTargetTriple(t));
    mgl::mglAirTargetApplyModuleMetadata(module, t);

    ModuleFacts facts;
    facts.triple = module.getTargetTriple();
    if (const llvm::NamedMDNode *ver = module.getNamedMetadata("air.version")) {
        const llvm::MDNode *n = ver->getOperand(0);
        for (unsigned i = 0; i < 3u; i++) {
            facts.air[i] = readMDInt(n, i);
        }
    }
    if (const llvm::NamedMDNode *lver =
            module.getNamedMetadata("air.language_version")) {
        const llvm::MDNode *n = lver->getOperand(0);
        facts.language = readMDString(n, 0);
        for (unsigned i = 0; i < 3u; i++) {
            facts.language_version[i] = readMDInt(n, i + 1u);
        }
    }
    return facts;
}

/* "air64_v28-apple-macosx26.0.0" -> {28, 26, 0, 0}; false when the string does
 * not have the expected shape. */
bool parseTriple(const std::string &triple, uint32_t out[4]) {
    unsigned v = 0, os1 = 0, os2 = 0, os3 = 0;
    if (std::sscanf(triple.c_str(), "air64_v%u-apple-macosx%u.%u.%u", &v, &os1,
                    &os2, &os3) != 4) {
        return false;
    }
    out[0] = v / 10u;
    out[1] = v % 10u;
    out[2] = os1;
    out[3] = os2;
    return os3 == 0u;
}

void testGolden(const mgl::AirTarget &t, const std::vector<uint8_t> &container) {
    printf("golden values\n");

    const std::string triple = mgl::mglAirTargetTriple(t);
    CHECK(triple == kExpectedTriple, "mglAirTargetTriple is the pinned triple");

    uint32_t parts[4] = {0, 0, 0, 0};
    CHECK(parseTriple(triple, parts), "triple parses as air64_v<M><m>-macosx<a>.<b>.<c>");
    CHECK(parts[0] == t.air_major && parts[1] == t.air_minor,
          "triple's v-component encodes air_major/air_minor");
    CHECK(parts[2] == t.os_major && parts[3] == t.os_minor,
          "triple's OS encodes os_major/os_minor");

    ModuleFacts facts = compileModuleFacts(t);
    CHECK(facts.triple == kExpectedTriple, "module target triple is the pinned triple");
    CHECK(facts.air[0] == kExpectedAirVersion[0] &&
              facts.air[1] == kExpectedAirVersion[1] &&
              facts.air[2] == kExpectedAirVersion[2],
          "!air.version is the pinned 2.8.0");
    CHECK(facts.air[0] == t.air_major && facts.air[1] == t.air_minor &&
              facts.air[2] == t.air_patch,
          "!air.version follows AirTarget");
    CHECK(facts.language == kExpectedLanguage,
          "!air.language_version names the pinned language");
    CHECK(facts.language_version[0] == kExpectedLanguageVersion[0] &&
              facts.language_version[1] == kExpectedLanguageVersion[1] &&
              facts.language_version[2] == kExpectedLanguageVersion[2],
          "!air.language_version is the pinned 4.0.0");
    CHECK(facts.language_version[0] == t.language_major &&
              facts.language_version[1] == t.language_minor &&
              facts.language_version[2] == t.language_patch,
          "!air.language_version follows AirTarget");
    CHECK(facts.language == t.language, "language name follows AirTarget");

    CHECK(container.size() >= 88u, "container has a full 88-byte header");
    CHECK(std::memcmp(container.data(), "MTLB", 4) == 0, "container magic is MTLB");
    CHECK(readU16(container, 4) == kExpectedPlatform, "header platform is 0x8001");
    CHECK(readU16(container, 6) == kExpectedContainerFormat[0] &&
              readU16(container, 8) == kExpectedContainerFormat[1],
          "header format version is the pinned 2.9");
    CHECK(container[10] == 0u, "header type byte is 0");
    CHECK(container[11] == kExpectedTargetField, "header target is 0x81");
    CHECK(readU16(container, 12) == kExpectedOs[0] &&
              readU16(container, 14) == kExpectedOs[1],
          "header OS is the pinned 26.0");

    CHECK(readU16(container, 4) == t.platform, "header platform follows AirTarget");
    CHECK(readU16(container, 6) == t.container_major &&
              readU16(container, 8) == t.container_minor,
          "header format version follows AirTarget");
    CHECK(container[11] == t.target, "header target follows AirTarget");
    CHECK(readU16(container, 12) == t.os_major && readU16(container, 14) == t.os_minor,
          "header OS follows AirTarget");

    size_t vers = 0;
    const bool haveVers = findFourcc(container, "VERS", &vers);
    CHECK(haveVers, "container carries a VERS tag");
    if (!haveVers) {
        return;
    }
    CHECK(readU16(container, vers + 4u) == 8u, "VERS payload is 8 bytes");
    CHECK(readU16(container, vers + 6u) == kExpectedAirVersion[0] &&
              readU16(container, vers + 8u) == kExpectedAirVersion[1],
          "VERS air pair is the pinned 2.8");
    CHECK(readU16(container, vers + 10u) == kExpectedLanguageVersion[0] &&
              readU16(container, vers + 12u) == kExpectedLanguageVersion[1],
          "VERS language pair is the pinned 4.0");
    CHECK(readU16(container, vers + 6u) == t.air_major &&
              readU16(container, vers + 8u) == t.air_minor &&
              readU16(container, vers + 10u) == t.language_major &&
              readU16(container, vers + 12u) == t.language_minor,
          "VERS follows AirTarget");

    printf("cross-artifact equalities\n");
    CHECK(readU16(container, 12) == parts[2] && readU16(container, 14) == parts[3],
          "container OS equals the triple's OS");
    CHECK(readU16(container, vers + 6u) == facts.air[0] &&
              readU16(container, vers + 8u) == facts.air[1],
          "container VERS air pair equals the module's !air.version");
    CHECK(readU16(container, vers + 10u) == facts.language_version[0] &&
              readU16(container, vers + 12u) == facts.language_version[1],
          "container VERS language pair equals !air.language_version");
    /* The two version numbers are separate fields on purpose.  Collapsing them
     * ("the container says 2.9, air.version says 2.9 too") is the edit this
     * assertion exists to stop. */
    CHECK(t.container_major != t.air_major || t.container_minor != t.air_minor,
          "container format version stays distinct from the AIR version");
}

void testConsistencyChecker(const mgl::AirTarget &good) {
    printf("consistency checker\n");

    char err[192];
    err[0] = '\0';
    CHECK(mgl::mglAirTargetIsConsistent(good, err, sizeof(err)),
          "the shipped target is consistent");

    auto rejects = [&](const char *label, const mgl::AirTarget &broken) {
        err[0] = '\0';
        CHECK(!mgl::mglAirTargetIsConsistent(broken, err, sizeof(err)) &&
                  err[0] != '\0',
              label);
    };

    mgl::AirTarget t = good;
    t.air_patch = 1u;
    rejects("a non-zero AIR patch is rejected (container cannot carry it)", t);

    t = good;
    t.language_patch = 1u;
    rejects("a non-zero language patch is rejected", t);

    t = good;
    t.os_patch = 1u;
    rejects("a non-zero OS patch is rejected (header cannot carry it)", t);

    t = good;
    t.air_major = 12u;
    rejects("a two-digit AIR major is rejected (triple is v<M><m>)", t);

    t = good;
    t.air_minor = 10u;
    rejects("a two-digit AIR minor is rejected", t);

    t = good;
    t.language = "";
    rejects("an empty language name is rejected", t);

    t = good;
    t.language = nullptr;
    rejects("a null language name is rejected", t);

    t = good;
    t.container_major = 0x10000u;
    rejects("a container format version wider than u16 is rejected", t);

    t = good;
    t.platform = 0x10000u;
    rejects("a platform wider than u16 is rejected", t);

    t = good;
    t.target = 0x100u;
    rejects("a target wider than u8 is rejected", t);
}

/* A hard-coded literal inside the triple builder or the metadata writer would
 * satisfy every golden above.  Mutating the struct has to move both artifacts.
 *
 * The container is a singleton read (mglMTLBWrite takes no target), so it
 * cannot be fed a mutation; its derivation is covered by the golden checks
 * above, and what is checked here is that two writes agree byte for byte. */
void testDerivationIsLive(const mgl::AirTarget &good,
                          const std::vector<uint8_t> &baseline) {
    printf("derivation is live, not hard-coded\n");

    mgl::AirTarget t = good;
    t.air_minor = 7u;
    t.os_minor = 1u;
    const std::string triple = mgl::mglAirTargetTriple(t);
    CHECK(triple == "air64_v27-apple-macosx26.1.0",
          "a mutated target changes the triple");
    CHECK(triple != mgl::mglAirTargetTriple(good),
          "triple differs from the shipped one");

    ModuleFacts facts = compileModuleFacts(t);
    CHECK(facts.air[1] == 7u, "a mutated target changes !air.version");
    CHECK(facts.language_version[0] == 4u,
          "the untouched language version is unchanged");

    CHECK(writeContainer() == baseline,
          "the container is deterministic for the shipped target");
}

} /* namespace */

int main(void) {
    printf("air target golden (non-Metal)\n");

    const mgl::AirTarget &t = mgl::mglAirTarget();
    const std::vector<uint8_t> container = writeContainer();

    testGolden(t, container);
    testConsistencyChecker(t);
    testDerivationIsLive(t, container);

    printf("%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
