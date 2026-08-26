// Pair a real MGL PTVS AIR (vertex) with a variant FS AIR and build a PSO.
//   ./pair_pso /tmp/poison_ptvs.air /tmp/pairA.air
#include <Metal/Metal.hpp>
#include <dispatch/dispatch.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static MTL::Library* loadAir(MTL::Device* dev, const char* path, NS::Error** err)
{
    FILE* f = fopen(path, "rb");
    if (!f) { printf("open fail %s\n", path); return nullptr; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); return nullptr; }
    fclose(f);
    dispatch_data_t data = dispatch_data_create(
        buf, (size_t)n, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    return dev->newLibrary(data, err);
}

int main(int argc, char** argv)
{
    if (argc < 3) { printf("usage: %s ptvs.air fs.air\n", argv[0]); return 1; }
    NS::Error* err = nullptr;
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::Library* vlib = loadAir(device, argv[1], &err);
    if (!vlib) { printf("ptvs lib fail\n"); return 1; }
    MTL::Function* vfn = vlib->newFunction(NS::MakeConstantString("main"));
    MTL::Library* flib = loadAir(device, argv[2], &err);
    if (!flib) { printf("fs lib fail\n"); return 1; }
    MTL::Function* ffn = flib->newFunction(NS::MakeConstantString("main"));

    MTL::RenderPipelineDescriptor* pd =
        MTL::RenderPipelineDescriptor::alloc()->init();
    pd->setVertexFunction(vfn);
    pd->setFragmentFunction(ffn);
    pd->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    MTL::RenderPipelineState* pso = device->newRenderPipelineState(pd, &err);
    if (!pso) {
        printf("PSO FAIL: %s\n",
               err ? err->localizedDescription()->utf8String() : "?");
        return 2;
    }
    printf("PSO OK\n");
    return 0;
}
