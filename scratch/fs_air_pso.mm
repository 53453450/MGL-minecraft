// Minimal reproducer: load a dumped AIR fragment function and build a
// render PSO so Apple's AGX compiler runs over it offline.
//   ./fs_air_pso /tmp/poison_fs.air
#include <Metal/Metal.hpp>
#include <dispatch/dispatch.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char* kVS =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "vertex float4 main0(uint vid [[vertex_id]]) { return float4(0, 0, 0, 1); }\n";

int main(int argc, char** argv)
{
    if (argc < 2) { printf("usage: %s file.air\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("open fail\n"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { return 1; }
    fclose(f);

    NS::Error* err = nullptr;
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    dispatch_data_t airData = dispatch_data_create(
        buf, (size_t)n, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    MTL::Library* flib = device->newLibrary(airData, &err);
    if (!flib) {
        printf("air lib fail: %s\n",
               err ? err->localizedDescription()->utf8String() : "?");
        return 1;
    }
    printf("library ok, functions:\n");
    NS::Array* fns = flib->functionNames();
    for (NS::UInteger i = 0; i < fns->count(); i++)
        printf("  %s\n", fns->object<NS::String>(i)->utf8String());
    MTL::Function* ffn = flib->newFunction(NS::MakeConstantString("main"));
    if (!ffn) { printf("no fn 'main'\n"); return 1; }

    MTL::Library* vlib = device->newLibrary(
        NS::String::string(kVS, NS::UTF8StringEncoding), nullptr, &err);
    if (!vlib) { printf("vs lib fail\n"); return 1; }
    MTL::Function* vfn = vlib->newFunction(
        NS::MakeConstantString("main0"));

    MTL::RenderPipelineDescriptor* pd =
        MTL::RenderPipelineDescriptor::alloc()->init();
    pd->setVertexFunction(vfn);
    pd->setFragmentFunction(ffn);
    pd->colorAttachments()->object(0)->setPixelFormat(
        MTL::PixelFormatRGBA8Unorm);
    MTL::RenderPipelineState* pso = device->newRenderPipelineState(pd, &err);
    if (!pso) {
        printf("PSO FAIL: %s\n",
               err ? err->localizedDescription()->utf8String() : "?");
        return 2;
    }
    printf("PSO OK\n");
    return 0;
}
