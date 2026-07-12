set SDKROOT=`xcrun --show-sdk-path`

# GLFW keeps its own thin facades in glfw/src/{MGLContext,MGLRenderer}.h.
# Do NOT copy MGL/include/MGLRenderer.h into glfw — that header pulls
# mgl_metal_bridge.h → glcorearb.h and redefines GLFW internal.h's GL_* macros
# (-Wmacro-redefined). Context enums/API live in glfw's MGLContext.h already.
# cp ../MGL/include/MGLContext.h glfw/src
# cp ../MGL/include/MGLRenderer.h glfw/src
cd SPIRV-Tools
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 4
cd ../..
cd SPIRV-Cross
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 4
cd ../..
cd SPIRV-Headers
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 4
cd ../..
cd glslang
./update_glslang_sources.py
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 4
cd ../..
cd glfw
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 4 glfw
cd ../..
