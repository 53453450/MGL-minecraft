//
//  MGL_Tests.m
//  MGL Tests
//
//  Created by Michael Larson on 1/3/25.
//

#import <XCTest/XCTest.h>

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>

#include <stdbool.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <limits.h>
#include <stdarg.h>
#include <sys/stat.h>

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

extern "C" {
#include "MGLContext.h"
}
#include "MGLRenderer.h"

#import "MGL_test_utils.h"


@interface MGL_Tests : XCTestCase
{
    NSApplication *m_application;
    NSWindow *m_window;
    GLMContext m_glm_ctx;
    MGLRenderer *m_renderer;
}
- (GLuint) winWidth;
- (GLuint) winHeight;
@end

@implementation MGL_Tests

- (NSRect) windowFrame
{
    return [m_window contentLayoutRect];
}

- (GLuint) winWidth
{
    return [self windowFrame].size.width;
}

- (GLuint) winHeight
{
    return [self windowFrame].size.height;
}

- (const char *) getTestDirectory
{
    return "/tmp/MGL_Testing";
}

- (bool) doesTestDirectoryExist
{
    const char *path;
    
    path = [self getTestDirectory];
    
    struct stat st;
    if (stat(path, &st) == 0)
    {
        if (S_ISDIR(st.st_mode))
        {
            return true;
        }
    }
    
    return false;
}

- (char *) getTestResultPath: (const char *)testname
{
    const char *test_dir;
    size_t len;
    
    test_dir = [self getTestDirectory];
    
    len = strlen(test_dir) + strlen(testname);
    len += 20;
    
    char *path;
    path = (char *)malloc(len);
    
    snprintf(path, len, "%s/%s.tga", test_dir, testname);
    
    return path;
}

- (char *) getGoldenImagePath: (const char *)testname
{
    const char *test_dir;
    size_t len;
    
    test_dir = [self getTestDirectory];
    
    len = strlen(test_dir) + strlen(testname);
    len += 20;
    
    char *path;
    path = (char *)malloc(len);
    
    snprintf(path, len, "%s/Golden_%s.tga", test_dir, testname);
    
    return path;
}

- (bool) isGoldenImageAvailable: (const char *)path
{
    FILE *fp;
    
    fp = fopen(path, "r");
    
    if (fp == NULL)
        return false;
    
    fclose(fp);
    
    return true;
}

- (size_t) fileSizeInBytes: (const char *)filePath
{
    struct stat fileStat;
    int err;
    
    err = stat(filePath, &fileStat);
    
    if (err == 0)
    {
        return fileStat.st_size; // File size in bytes
    }

    perror("stat failed");
    
    return 0;
}

- (size_t) readFile: (const char *)path toBuf: (char *)buf ofLen: (size_t) len;
{
    FILE *fp;
    
    if (len == 0)
        return [self fileSizeInBytes: path];;

    if (buf == NULL)
        return 0;
    
    fp = fopen(path, "rb");
    if (fp == NULL)
        return 0;
    
    len = fread(buf, len, 1, fp);
    
    fclose(fp);
    
    // return num items read
    return len;
}

- (bool) compareFiles: (const char *)path golden: (const char *)golden_path
{
    size_t buf_len, golden_buf_len;
    char *buf, *golden_buf;
    bool result;
    
    result = false;

    do
    {
        buf_len = [self fileSizeInBytes: path];
        if (buf_len == 0)
            continue;
        
        golden_buf_len = [self fileSizeInBytes: golden_path];
        if (golden_buf_len == 0)
            continue;

        // compare buffer sizes
        if (buf_len != golden_buf_len)
            continue;

        // read images
        buf = (char *)malloc(buf_len);
        [self readFile: path toBuf: buf ofLen: buf_len];
        
        golden_buf = (char *)malloc(golden_buf_len);
        [self readFile: golden_path toBuf: golden_buf ofLen: golden_buf_len];

        // compare images
        if (memcmp(golden_buf, buf, buf_len) == 0)
        {
            result = true;
        }
        
        if (buf)
            free(buf);
        
        if (golden_buf)
            free(golden_buf);
        
    } while(0);
    
    return result;
}

- (void) createResultDirectoryIfNeeded
{
    bool result;
    
    // create test directory if it doesn't exist
    result = [self doesTestDirectoryExist];
    if (result == false)
    {
        mkdir([self getTestDirectory], 0700);
    }
}

- (bool) writeResult: (const char *)testname size:(NSSize)size pixels:(void *)pixels
{
    bool result;
    char *path;

    [self createResultDirectoryIfNeeded];

    do {
        path = [self getTestResultPath: testname];
        
        result = tga_write(path, size.width, size.height, (uint8_t *)pixels, 4, 4);
        if (result == false)
            continue;

        result = true;
    } while(0);
        
    free(path);

    return result;
}

- (bool) writeGoldenResult: (const char *)testname size:(NSSize)size pixels:(void *)pixels
{
    bool result;
    char *golden_path;

    [self createResultDirectoryIfNeeded];

    do {
        golden_path = [self getGoldenImagePath: testname];
        
        result = tga_write(golden_path, size.width, size.height, (uint8_t *)pixels, 4, 4);
        if (result == false)
            continue;

        result = true;
    } while(0);
        
    free(golden_path);

    return result;
}

- (bool) writeAndCompareResults: (const char *)testname size:(NSSize)size pixels:(void *)pixels
{
    bool result;
    char *path, *golden_path;

    // create test directory if it doesn't exist
    [self createResultDirectoryIfNeeded];

    path = NULL; // get rid of warning
    do {
        golden_path = [self getGoldenImagePath: testname];
        
        if ([self isGoldenImageAvailable: golden_path] == false)
        {
            result = tga_write(golden_path, size.width, size.height, (uint8_t *)pixels, 4, 4);
            if (result == false)
                continue;
        }
        
        path = [self getTestResultPath: testname];
        
        result = tga_write(path, size.width, size.height, (uint8_t *)pixels, 4, 4);
        if (result == false)
            continue;

        result = [self compareFiles: path golden: golden_path];
        if (result == false)
            continue;
        
        result = true;
    } while(0);
        
    free(golden_path);
    free(path);

    return result;
}

- (bool) compareResults: (const char *)test_name
{
    size_t len;
    len = [self winWidth] * [self winHeight] * 4;
    
    uint8_t *pixels;
    pixels = (uint8_t *)malloc(len);

    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, [self winWidth], [self winHeight], GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glFlush();

    NSSize size;
    bool test_passed;

    test_passed = false;

    size = NSMakeSize([self winWidth], [self winHeight]);
    test_passed = [self writeAndCompareResults: test_name size: size pixels: pixels];
    
    free(pixels);
                      
    if(test_passed == false)
        XCTFail(@"Image comparison failed");
    
    return test_passed;
}

- (void)setUp
{
    dispatch_async(dispatch_get_main_queue(), ^{
        // Initialize NSApplication
        if (NSApp == NULL)
        {
            self->m_application = [NSApplication sharedApplication];
        }

        // Create and configure the NSWindow
        self->m_window = [[NSWindow alloc] initWithContentRect:NSMakeRect(32, 32, 512, 512)
                                                     styleMask:(NSWindowStyleMaskTitled |
                                                                NSWindowStyleMaskClosable)
                                                       backing:NSBackingStoreBuffered
                                                         defer:NO];
        
        // Run the main loop temporarily to simulate user interaction
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow: 1]];
        
        // Show the window
        [self->m_window makeKeyAndOrderFront:nil];
        [self->m_window makeMainWindow];
                
        [self->m_window setBackgroundColor: [NSColor blackColor]];
        
        [self->m_window display];
        [self->m_window makeKeyAndOrderFront:nil];
        
        [NSApp activateIgnoringOtherApps:YES];
        
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow: 1]];

        self->m_glm_ctx = createGLMContext(GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
        self->m_renderer = [[MGLRenderer alloc] initMGLRendererFromContext: self->m_glm_ctx andBindToWindow: self->m_window];
        
        if (!self->m_renderer)
        {
            exit(EXIT_FAILURE);
        }
        
        MGLsetCurrentContext(self->m_glm_ctx);
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow: 1]];
    });
}

- (void)tearDown
{
    dispatch_async(dispatch_get_main_queue(), ^{
        self->m_window = nil;
    });

    self->m_renderer = NULL;
    self->m_glm_ctx = NULL;
    
    [super tearDown];
}

- (void)testGLClear
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"Task on main thread completed"];
    
    // Ensure everything runs on the main thread
    dispatch_async(dispatch_get_main_queue(), ^{
        int a = 0;
        int e = 1;
        
        int count;
        count = 200;
        while(count--)
        {
            glClearColor(0.2, 0.2, (float)a/100.0, 0.0);
            glClear(GL_COLOR_BUFFER_BIT);
            
            a += e;
            if(a>=100){e=-1;}
            if(a==0){e=1;}

            MGLswapBuffers(NULL);
        }

        bool result;

        result = [self compareResults: "testGLClear"];

        [expectation fulfill];
    });
    
    // Wait for the expectation
    [self waitForExpectationsWithTimeout:60.0 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"The task did not complete in time");
    }];
}

- (void)testGLDrawArrays
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"Task on main thread completed"];
    
    // Ensure everything runs on the main thread
    dispatch_async(dispatch_get_main_queue(), ^{
        GLuint vbo = 0, vao = 0;

        const char* vertex_shader =
        GLSL(460,
             layout(location = 0) in vec3 position;
             void main() {
                gl_Position = vec4(position, 1.0);
            }
        );

        const char* fragment_shader =
        GLSL(460,
             layout(location = 0) out vec4 frag_colour;
             void main() {
                frag_colour = vec4(0.5, 0.0, 0.5, 1.0);
            }
        );

        float points[] = {
           0.0f,  0.5f,  0.0f,
           0.5f, -0.5f,  0.0f,
          -0.5f, -0.5f,  0.0f
        };

        vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
        vao = bindVAO();

        bindAttribute(0, GL_ARRAY_BUFFER, vbo, 3, GL_FLOAT, false, 0, NULL);

        GLuint shader_program = compileGLSLProgram(2, GL_VERTEX_SHADER, vertex_shader, GL_FRAGMENT_SHADER, fragment_shader);
        glUseProgram(shader_program);

        glViewport(0, 0, [self winWidth], [self winHeight]);

        glClearColor(0.2, 0.2, 0.2, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
            
        MGLswapBuffers(NULL);

        bool result;

        result = [self compareResults: "testGLDrawArrays"];

        [expectation fulfill];
    });
    
    // Wait for the expectation
    [self waitForExpectationsWithTimeout:120 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"The task did not complete in time");
    }];
}

- (void)testGLDrawArraysUniform1i
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"Task on main thread completed"];
    
    // Ensure everything runs on the main thread
    dispatch_async(dispatch_get_main_queue(), ^{
        GLuint vbo = 0, vao = 0;

        const char* vertex_shader =
        GLSL(460,
             layout(location = 0) in vec3 position;
             void main() {
                gl_Position = vec4(position.x, position.y, position.z, 1.0);
            }
        );
        const char* fragment_shader =
        GLSL(460,
             layout(location = 0) out vec4 frag_colour;
             layout(location = 0) uniform int mp;
             void main() {
                frag_colour = vec4(0.0, 0.0, float(mp)/100.0, 1.0);
            }
        );

        float points[] = {
           0.0f,  0.5f,  0.0f,
           0.5f, -0.5f,  0.0f,
          -0.5f, -0.5f,  0.0f
        };

        vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
        vao = bindVAO();

        bindAttribute(0, GL_ARRAY_BUFFER, vbo, 3, GL_FLOAT, false, 0, NULL);

        GLuint shader_program = compileGLSLProgram(2, GL_VERTEX_SHADER, vertex_shader, GL_FRAGMENT_SHADER, fragment_shader);
        glUseProgram(shader_program);

        glViewport(0, 0, [self winWidth], [self winHeight]);

        GLint mp_loc = glGetUniformLocation(shader_program, "mp");
        std::cout << mp_loc << std::endl;
        
        int a = 0;
        int e = 1;
        
        glClearColor(0.2, 0.2, 0.2, 0.0);
        
        glClear(GL_COLOR_BUFFER_BIT);
        MGLswapBuffers(NULL);

        glUseProgram(shader_program);
        
        int count;
        count = 100;
        while(count--)
        {
            glBindVertexArray(vao);
            glUniform1i(mp_loc, a);
            
            glClearColor(0.2, 0.2, (float)a/100.0, 0.0);
            glClear(GL_COLOR_BUFFER_BIT);
            
            glDrawArrays(GL_TRIANGLES, 0, 3);
            
            MGLswapBuffers(NULL);
            
            a += e;
            if(a>=100){e=-1;}
            if(a==0){e=1;}
        }
        
        bool result;

        result = [self compareResults: "testGLDrawArraysUniform1i"];

        [expectation fulfill];
    });
    
    // Wait for the expectation
    [self waitForExpectationsWithTimeout:120.0 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"The task did not complete in time");
    }];
}

- (void) testReadPixels
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"Task on main thread completed"];
    
    // Ensure everything runs on the main thread
    dispatch_async(dispatch_get_main_queue(), ^{
        const char* vertex_shader =
        GLSL(450 core,
             layout(location = 0) in vec3 position;
             layout(location = 1) in vec3 in_color;
             layout(location = 2) in vec2 in_texcords;
             
             layout(location = 0) out vec4 out_color;
             layout(location = 1) out vec2 out_texcoords;
             
             void main() {
                gl_Position = vec4(position, 1.0);
                out_color = vec4(in_color, 1.0);
                out_texcoords = in_texcords;
            }
        );

        const char* fragment_shader =
        GLSL(450 core,
             layout(location = 0) in vec4 in_color;
             layout(location = 1) in vec2 in_texcords;
             
             layout(location = 0) out vec4 frag_colour;
             
             uniform sampler2D image;
             
             void main() {
                vec4 tex_color = texture(image, in_texcords);
            
                frag_colour = in_color * tex_color;
            }
        );
        
        GLuint vbo = 0, col_vbo = 0, tex_vbo = 0;

        float points[] = {
            0.0f,  0.5f,  0.0f,
            0.5f, -0.5f,  0.0f,
            -0.5f, -0.5f,  0.0f
        };

        float color[] = {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
        };

        float texcoords[] = {
            0.5f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
        };

        vbo = bindDataToVBO(GL_ARRAY_BUFFER, 9 * sizeof(float), points, GL_STATIC_DRAW);
        col_vbo = bindDataToVBO(GL_ARRAY_BUFFER, 9 * sizeof(float), color, GL_STATIC_DRAW);
        tex_vbo = bindDataToVBO(GL_ARRAY_BUFFER, 6 * sizeof(float), texcoords, GL_STATIC_DRAW);

        GLuint vao = 0;
        glCreateVertexArrays(1, &vao);
        glBindVertexArray(vao);

        bindAttribute(0, GL_ARRAY_BUFFER, vbo, 3, GL_FLOAT, false, 0, NULL);
        bindAttribute(1, GL_ARRAY_BUFFER, col_vbo, 3, GL_FLOAT, false, 0, NULL);
        bindAttribute(2, GL_ARRAY_BUFFER, tex_vbo, 2, GL_FLOAT, false, 0, NULL);

        GLuint shader_program = compileGLSLProgram(2, GL_VERTEX_SHADER, vertex_shader, GL_FRAGMENT_SHADER, fragment_shader);
        glUseProgram(shader_program);

        GLuint tex;
        tex = createTexture(GL_TEXTURE_2D, 256, 256, 0, genTexturePixels(GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, 0x10, 256, 256));
        glBindTexture(GL_TEXTURE_2D, tex);

        glViewport(0, 0, [self winWidth], [self winHeight]);

        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);

        MGLswapBuffers(NULL);

        bool result;
        result = [self compareResults: "testReadPixels"];

        [expectation fulfill];
    });
    
    // Wait for the expectation
    [self waitForExpectationsWithTimeout:120.0 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"The task did not complete in time");
    }];
}

// =============================================================================
// Task 5: GL Implicit Synchronization Regression Tests
// These tests verify that GL's sequential visibility semantics are preserved
// under Metal's async execution model.
// =============================================================================

// Test 5.1 — Feedback loop: draw writes attachment T → samples T after barrier.
// Verifies that a draw writing to color attachment texture T is visible to a
// subsequent draw that samples T, after glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT).
// Uses FBO-based readback (not default framebuffer) because MGLswapBuffers rotates
// to a fresh drawable, making GL_FRONT readback unreliable for single-frame tests.
- (void)testGLSyncFeedbackLoop
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"testGLSyncFeedbackLoop completed"];

    dispatch_async(dispatch_get_main_queue(), ^{
        const GLsizei rtW = 64, rtH = 64;

        const char* solid_vs =
        GLSL(460,
             layout(location = 0) in vec3 position;
             void main() {
                gl_Position = vec4(position, 1.0);
            }
        );
        const char* solid_fs =
        GLSL(460,
             layout(location = 0) out vec4 frag_colour;
             uniform vec4 u_color;
             void main() {
                frag_colour = u_color;
            }
        );
        const char* sample_vs =
        GLSL(460,
             layout(location = 0) in vec3 position;
             layout(location = 1) in vec2 in_texcoord;
             layout(location = 1) out vec2 out_texcoord;
             void main() {
                gl_Position = vec4(position, 1.0);
                out_texcoord = in_texcoord;
            }
        );
        const char* sample_fs =
        GLSL(460,
             layout(location = 1) in vec2 in_texcoord;
             layout(location = 0) out vec4 frag_colour;
             uniform sampler2D u_tex;
             void main() {
                frag_colour = texture(u_tex, in_texcoord);
            }
        );

        GLuint solid_prog = compileGLSLProgram(2, GL_VERTEX_SHADER, solid_vs, GL_FRAGMENT_SHADER, solid_fs);
        GLuint sample_prog = compileGLSLProgram(2, GL_VERTEX_SHADER, sample_vs, GL_FRAGMENT_SHADER, sample_fs);

        // Fullscreen quad (two triangles) with texcoords
        float quad_pos[] = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
        };
        float quad_tex[] = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
        };

        GLuint pos_vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(quad_pos), quad_pos, GL_STATIC_DRAW);
        GLuint tex_vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(quad_tex), quad_tex, GL_STATIC_DRAW);
        GLuint vao = bindVAO();
        bindAttribute(0, GL_ARRAY_BUFFER, pos_vbo, 3, GL_FLOAT, false, 0, NULL);
        bindAttribute(1, GL_ARRAY_BUFFER, tex_vbo, 2, GL_FLOAT, false, 0, NULL);

        // Create texture T as render target for the first draw
        GLuint texT;
        glGenTextures(1, &texT);
        glBindTexture(GL_TEXTURE_2D, texT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rtW, rtH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Create FBO_A, attach T as color attachment
        GLuint fboA;
        glGenFramebuffers(1, &fboA);
        glBindFramebuffer(GL_FRAMEBUFFER, fboA);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texT, 0);

        glViewport(0, 0, rtW, rtH);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw solid red into T
        glUseProgram(solid_prog);
        GLint color_loc = glGetUniformLocation(solid_prog, "u_color");
        glUniform4f(color_loc, 1.0, 0.0, 0.0, 1.0);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Barrier: ensure draw writes to T are visible to subsequent texture fetch
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        // Create result texture and FBO_B for the sampling draw
        GLuint texResult;
        glGenTextures(1, &texResult);
        glBindTexture(GL_TEXTURE_2D, texResult);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rtW, rtH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        GLuint fboB;
        glGenFramebuffers(1, &fboB);
        glBindFramebuffer(GL_FRAMEBUFFER, fboB);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texResult, 0);

        glViewport(0, 0, rtW, rtH);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        // Sample T into result texture
        glUseProgram(sample_prog);
        GLint tex_loc = glGetUniformLocation(sample_prog, "u_tex");
        glUniform1i(tex_loc, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texT);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Read back center pixel from FBO_B (reliable — reads from texResult, not a drawable)
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        uint8_t pixel[4] = {0, 0, 0, 0};
        glReadPixels(rtW / 2, rtH / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);

        XCTAssertEqual(pixel[0], 255, @"Feedback loop: expected red channel 255, got %d", pixel[0]);
        XCTAssertEqual(pixel[1], 0,   @"Feedback loop: expected green channel 0, got %d", pixel[1]);
        XCTAssertEqual(pixel[2], 0,   @"Feedback loop: expected blue channel 0, got %d", pixel[2]);

        glDeleteFramebuffers(1, &fboA);
        glDeleteFramebuffers(1, &fboB);
        glDeleteTextures(1, &texT);
        glDeleteTextures(1, &texResult);
        glDeleteProgram(solid_prog);
        glDeleteProgram(sample_prog);

        [expectation fulfill];
    });

    [self waitForExpectationsWithTimeout:120.0 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"testGLSyncFeedbackLoop did not complete in time");
    }];
}

// Test 5.2 — Texture upload-after-sample: draw samples T → glTexSubImage2D updates T →
// draw samples T again. Verifies the second draw reads the new data, not stale data.
// Covers GL_TEXTURE_2D target.
// Uses FBO-based readback (not default framebuffer) because MGLswapBuffers rotates
// to a fresh drawable, making GL_FRONT readback unreliable for single-frame tests.
- (void)testGLSyncTextureUploadAfterSample
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"testGLSyncTextureUploadAfterSample completed"];

    dispatch_async(dispatch_get_main_queue(), ^{
        const GLsizei rtW = 64, rtH = 64;

        const char* sample_vs =
        GLSL(460,
             layout(location = 0) in vec3 position;
             layout(location = 1) in vec2 in_texcoord;
             layout(location = 1) out vec2 out_texcoord;
             void main() {
                gl_Position = vec4(position, 1.0);
                out_texcoord = in_texcoord;
            }
        );
        const char* sample_fs =
        GLSL(460,
             layout(location = 1) in vec2 in_texcoord;
             layout(location = 0) out vec4 frag_colour;
             uniform sampler2D u_tex;
             void main() {
                frag_colour = texture(u_tex, in_texcoord);
            }
        );

        GLuint sample_prog = compileGLSLProgram(2, GL_VERTEX_SHADER, sample_vs, GL_FRAGMENT_SHADER, sample_fs);

        float quad_pos[] = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
        };
        float quad_tex[] = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
        };

        GLuint pos_vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(quad_pos), quad_pos, GL_STATIC_DRAW);
        GLuint tex_vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(quad_tex), quad_tex, GL_STATIC_DRAW);
        GLuint vao = bindVAO();
        bindAttribute(0, GL_ARRAY_BUFFER, pos_vbo, 3, GL_FLOAT, false, 0, NULL);
        bindAttribute(1, GL_ARRAY_BUFFER, tex_vbo, 2, GL_FLOAT, false, 0, NULL);

        // Create texture T with initial green data (2x2 so sampling anywhere hits green)
        const GLsizei texW = 2, texH = 2;
        uint8_t green_data[texW * texH * 4];
        for (int i = 0; i < texW * texH; i++) {
            green_data[i * 4 + 0] = 0;
            green_data[i * 4 + 1] = 255;
            green_data[i * 4 + 2] = 0;
            green_data[i * 4 + 3] = 255;
        }
        uint8_t blue_data[texW * texH * 4];
        for (int i = 0; i < texW * texH; i++) {
            blue_data[i * 4 + 0] = 0;
            blue_data[i * 4 + 1] = 0;
            blue_data[i * 4 + 2] = 255;
            blue_data[i * 4 + 3] = 255;
        }

        GLuint texT;
        glGenTextures(1, &texT);
        glBindTexture(GL_TEXTURE_2D, texT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texW, texH, 0, GL_RGBA, GL_UNSIGNED_BYTE, green_data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Create result texture + FBO for readback
        GLuint texResult;
        glGenTextures(1, &texResult);
        glBindTexture(GL_TEXTURE_2D, texResult);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rtW, rtH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texResult, 0);

        glViewport(0, 0, rtW, rtH);
        glUseProgram(sample_prog);
        GLint tex_loc = glGetUniformLocation(sample_prog, "u_tex");
        glUniform1i(tex_loc, 0);
        glActiveTexture(GL_TEXTURE0);

        // First draw: sample T (should be green)
        glBindTexture(GL_TEXTURE_2D, texT);
        glBindVertexArray(vao);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glReadBuffer(GL_COLOR_ATTACHMENT0);
        uint8_t pixel1[4] = {0, 0, 0, 0};
        glReadPixels(rtW / 2, rtH / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel1);

        XCTAssertEqual(pixel1[1], 255, @"Upload-after-sample: first draw expected green channel 255, got %d", pixel1[1]);
        XCTAssertEqual(pixel1[2], 0,   @"Upload-after-sample: first draw expected blue channel 0, got %d", pixel1[2]);

        // Update T to blue via glTexSubImage2D
        glBindTexture(GL_TEXTURE_2D, texT);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texW, texH, GL_RGBA, GL_UNSIGNED_BYTE, blue_data);

        // Second draw: sample T (should now be blue, not stale green)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, rtW, rtH);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glReadBuffer(GL_COLOR_ATTACHMENT0);
        uint8_t pixel2[4] = {0, 0, 0, 0};
        glReadPixels(rtW / 2, rtH / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel2);

        XCTAssertEqual(pixel2[1], 0,   @"Upload-after-sample: second draw expected green channel 0 (stale data), got %d", pixel2[1]);
        XCTAssertEqual(pixel2[2], 255, @"Upload-after-sample: second draw expected blue channel 255, got %d", pixel2[2]);

        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &texT);
        glDeleteTextures(1, &texResult);
        glDeleteProgram(sample_prog);

        [expectation fulfill];
    });

    [self waitForExpectationsWithTimeout:120.0 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"testGLSyncTextureUploadAfterSample did not complete in time");
    }];
}

// Test 5.3 — Buffer readback-after-compute: compute writes to SSBO → glMemoryBarrier →
// glGetBufferSubData reads back. Verifies compute writes are visible to CPU readback.
- (void)testGLSyncBufferReadbackAfterCompute
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"testGLSyncBufferReadbackAfterCompute completed"];

    dispatch_async(dispatch_get_main_queue(), ^{
        const char* compute_shader =
        GLSL(460,
             layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
             layout(std430, binding = 0) buffer OutBuf {
                 int data[];
             };
             void main() {
                 data[0] = 42;
             }
        );

        GLuint compute_prog = compileGLSLProgram(1, GL_COMPUTE_SHADER, compute_shader);
        glUseProgram(compute_prog);

        // Create SSBO with initial zero data
        GLint initial_data[4] = {0, 0, 0, 0};
        GLuint ssbo;
        glGenBuffers(1, &ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initial_data), initial_data, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

        // Dispatch compute
        glDispatchCompute(1, 1, 1);

        // Barrier: ensure compute writes are visible to CPU readback
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        // Read back via glGetBufferSubData
        GLint readback[4] = {0, 0, 0, 0};
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(readback), readback);

        XCTAssertEqual(readback[0], 42, @"Buffer readback-after-compute: expected data[0]=42, got %d", readback[0]);

        glDeleteBuffers(1, &ssbo);
        glDeleteProgram(compute_prog);

        [expectation fulfill];
    });

    [self waitForExpectationsWithTimeout:120.0 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"testGLSyncBufferReadbackAfterCompute did not complete in time");
    }];
}

// Test 5.4 — Fence round-trip: draw → glFenceSync → more draws → glClientWaitSync.
// Verifies the fence actually waits for GPU completion of commands before the fence
// insertion point, and that glGetSynciv status transitions correctly.
- (void)testGLSyncFenceRoundTrip
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"testGLSyncFenceRoundTrip completed"];

    dispatch_async(dispatch_get_main_queue(), ^{
        GLuint winW = [self winWidth];
        GLuint winH = [self winHeight];

        const char* vertex_shader =
        GLSL(460,
             layout(location = 0) in vec3 position;
             void main() {
                gl_Position = vec4(position, 1.0);
            }
        );
        const char* fragment_shader =
        GLSL(460,
             layout(location = 0) out vec4 frag_colour;
             void main() {
                frag_colour = vec4(1.0, 1.0, 0.0, 1.0);
            }
        );

        GLuint shader_program = compileGLSLProgram(2, GL_VERTEX_SHADER, vertex_shader, GL_FRAGMENT_SHADER, fragment_shader);
        glUseProgram(shader_program);

        float points[] = {
           0.0f,  0.5f,  0.0f,
           0.5f, -0.5f,  0.0f,
          -0.5f, -0.5f,  0.0f
        };

        GLuint vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
        GLuint vao = bindVAO();
        bindAttribute(0, GL_ARRAY_BUFFER, vbo, 3, GL_FLOAT, false, 0, NULL);

        glViewport(0, 0, winW, winH);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw before fence
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Insert fence
        GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        XCTAssertNotEqual(fence, (GLsync)NULL, @"Fence round-trip: glFenceSync returned NULL");

        // Query status pre-wait (UNSIGNALED or SIGNALED both acceptable)
        GLint pre_status = 0;
        GLsizei pre_len = 0;
        glGetSynciv(fence, GL_SYNC_STATUS, 1, &pre_len, &pre_status);
        XCTAssertTrue(pre_status == GL_UNSIGNALED || pre_status == GL_SIGNALED,
                      @"Fence round-trip: pre-wait status expected UNSIGNALED or SIGNALED, got 0x%x", pre_status);

        // Issue more draws after fence
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Wait for fence (5 second timeout in nanoseconds)
        GLuint64 timeout_ns = 5000000000ULL;
        GLenum wait_result = glClientWaitSync(fence, 0, timeout_ns);

        XCTAssertTrue(wait_result == GL_ALREADY_SIGNALED || wait_result == GL_CONDITION_SATISFIED,
                      @"Fence round-trip: glClientWaitSync expected ALREADY_SIGNALED or CONDITION_SATISFIED, got 0x%x", wait_result);

        // After wait, status must be GL_SIGNALED
        GLint post_status = 0;
        GLsizei post_len = 0;
        glGetSynciv(fence, GL_SYNC_STATUS, 1, &post_len, &post_status);
        XCTAssertEqual(post_status, (GLint)GL_SIGNALED,
                       @"Fence round-trip: post-wait status expected SIGNALED, got 0x%x", post_status);

        glDeleteSync(fence);
        glDeleteProgram(shader_program);

        [expectation fulfill];
    });

    [self waitForExpectationsWithTimeout:120.0 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"testGLSyncFenceRoundTrip did not complete in time");
    }];
}

// Test 5.5 — glMemoryBarrier compute→fragment visibility: compute writes to image T
// via imageStore → glMemoryBarrier → fragment samples T. Verifies compute image
// writes are visible to subsequent fragment texture fetches.
// Uses FBO-based readback (not default framebuffer) because MGLswapBuffers rotates
// to a fresh drawable, making GL_FRONT readback unreliable for single-frame tests.
- (void)testGLSyncMemoryBarrierComputeToFragment
{
    XCTestExpectation *expectation = [self expectationWithDescription:@"testGLSyncMemoryBarrierComputeToFragment completed"];

    dispatch_async(dispatch_get_main_queue(), ^{
        const GLsizei rtW = 64, rtH = 64;
        const GLsizei texW = 16, texH = 16;

        const char* compute_shader =
        GLSL(460,
             layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
             layout(rgba8, binding = 0) writeonly uniform image2D u_img;
             void main() {
                 ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
                 imageStore(u_img, coord, vec4(0.0, 0.0, 1.0, 1.0));
             }
        );
        const char* sample_vs =
        GLSL(460,
             layout(location = 0) in vec3 position;
             layout(location = 1) in vec2 in_texcoord;
             layout(location = 1) out vec2 out_texcoord;
             void main() {
                gl_Position = vec4(position, 1.0);
                out_texcoord = in_texcoord;
            }
        );
        const char* sample_fs =
        GLSL(460,
             layout(location = 1) in vec2 in_texcoord;
             layout(location = 0) out vec4 frag_colour;
             uniform sampler2D u_tex;
             void main() {
                frag_colour = texture(u_tex, in_texcoord);
            }
        );

        GLuint compute_prog = compileGLSLProgram(1, GL_COMPUTE_SHADER, compute_shader);
        GLuint sample_prog = compileGLSLProgram(2, GL_VERTEX_SHADER, sample_vs, GL_FRAGMENT_SHADER, sample_fs);

        // Create texture T (GL_RGBA8), bound as image and as sampler
        GLuint texT;
        glGenTextures(1, &texT);
        glBindTexture(GL_TEXTURE_2D, texT);
        // Initialize to zero so we can detect compute writes
        std::vector<uint8_t> zero_data(texW * texH * 4, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texW, texH, 0, GL_RGBA, GL_UNSIGNED_BYTE, zero_data.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Bind T as image for compute (write-only)
        glBindImageTexture(0, texT, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        // Dispatch compute: write blue to every pixel of T
        glUseProgram(compute_prog);
        glDispatchCompute(texW, texH, 1);

        // Barrier: ensure compute image writes are visible to subsequent fragment texture fetches
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        // Set up fullscreen quad for sampling
        float quad_pos[] = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
        };
        float quad_tex[] = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
        };

        GLuint pos_vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(quad_pos), quad_pos, GL_STATIC_DRAW);
        GLuint tex_vbo = bindDataToVBO(GL_ARRAY_BUFFER, sizeof(quad_tex), quad_tex, GL_STATIC_DRAW);
        GLuint vao = bindVAO();
        bindAttribute(0, GL_ARRAY_BUFFER, pos_vbo, 3, GL_FLOAT, false, 0, NULL);
        bindAttribute(1, GL_ARRAY_BUFFER, tex_vbo, 2, GL_FLOAT, false, 0, NULL);

        // Create result texture + FBO for readback (FBO-based readback is reliable;
        // default framebuffer + MGLswapBuffers rotates to a fresh empty drawable).
        GLuint texResult;
        glGenTextures(1, &texResult);
        glBindTexture(GL_TEXTURE_2D, texResult);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rtW, rtH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texResult, 0);

        // Draw fullscreen quad sampling T into result texture
        glViewport(0, 0, rtW, rtH);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(sample_prog);
        GLint tex_loc = glGetUniformLocation(sample_prog, "u_tex");
        glUniform1i(tex_loc, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texT);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Read back center pixel — should be blue (compute-written color)
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        uint8_t pixel[4] = {0, 0, 0, 0};
        glReadPixels(rtW / 2, rtH / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);

        XCTAssertEqual(pixel[0], 0,   @"Compute→fragment: expected red channel 0, got %d", pixel[0]);
        XCTAssertEqual(pixel[1], 0,   @"Compute→fragment: expected green channel 0, got %d", pixel[1]);
        XCTAssertEqual(pixel[2], 255, @"Compute→fragment: expected blue channel 255 (compute-written), got %d", pixel[2]);

        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &texT);
        glDeleteTextures(1, &texResult);
        glDeleteProgram(compute_prog);
        glDeleteProgram(sample_prog);

        [expectation fulfill];
    });

    [self waitForExpectationsWithTimeout:120.0 handler:^(NSError * _Nullable error) {
        XCTAssertNil(error, @"testGLSyncMemoryBarrierComputeToFragment did not complete in time");
    }];
}

#if 0
- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}
#endif

@end
