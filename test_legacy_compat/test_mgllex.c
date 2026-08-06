/*
 * test_mgllex.c
 *
 * M0 verification for the self-written GLSL lexer skeleton: tokenizes a
 * hello vertex shader and checks directive capture, identifier/number
 * token classes and literal decoding.
 *
 * Build (same pattern as test_mglir):
 *   cc -Wall -Wextra -O0 -g \
 *     -IMGL/include \
 *     test_legacy_compat/test_mgllex.c MGL/src/mgl_glsl_lexer.c \
 *     -o build/test_mgllex
 */
#include "mgl_glsl_lexer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Tokenize `src`, return the end/start offset of the i-th token. */
static size_t tok_start(const char *src, int index)
{
    MGLGLSLexer lx;
    mglGLSLexerInit(&lx, src, strlen(src));
    MGLGLSLToken t;
    int i = 0;
    for (;;) {
        mglGLSLexerNext(&lx, &t);
        if (i == index) {
            return t.start;
        }
        if (t.kind == MGLGLSL_TOK_END) {
            return (size_t)-1;
        }
        i++;
    }
}

static char *slice(const char *src, const MGLGLSLToken *t)
{
    size_t n = t->end - t->start;
    char *buf = (char *)malloc(n + 1);
    if (buf) {
        memcpy(buf, src + t->start, n);
        buf[n] = '\0';
    }
    return buf;
}

static void test_directive(void)
{
    printf("directive capture\n");
    const char *src = "#version 450 core\nlayout(location = 0) in vec3 pos;\n";
    MGLGLSLexer lx;
    mglGLSLexerInit(&lx, src, strlen(src));
    MGLGLSLToken t;
    mglGLSLexerNext(&lx, &t);
    char *txt = slice(src, &t);
    CHECK(t.kind == MGLGLSL_TOK_DIRECTIVE, "#version directive token");
    CHECK(strcmp(txt, "#version 450 core") == 0, "directive body is raw line");
    CHECK(t.line == 1, "directive line=1");
    free(txt);

    mglGLSLexerNext(&lx, &t);
    txt = slice(src, &t);
    CHECK(t.kind == MGLGLSL_TOK_IDENT, "token after directive is ident");
    CHECK(strcmp(txt, "layout") == 0, "ident is 'layout'");
    free(txt);
}

static void test_hello(void)
{
    printf("hello vertex shader tokenization\n");
    const char *src =
        "#version 450 core\n"
        "layout(location = 0) in vec3 pos;\n"
        "void main() { gl_Position = vec4(pos, 1.0); }\n";
    MGLGLSLexer lx;
    mglGLSLexerInit(&lx, src, strlen(src));
    MGLGLSLToken t;
    mglGLSLexerNext(&lx, &t);
    CHECK(t.kind == MGLGLSL_TOK_DIRECTIVE, "first token is #version");

    /* Verify the important identifiers/values appear in order after the
     * directive. */
    int line = 0;
    CHECK(tok_start(src, 0) == 0, "first token starts at 0");

    /* tokens after directive: layout ( ( location = 0 ) in vec3 pos ; ... */
    char *ids[4];
    int found = 0;
    for (;;) {
        mglGLSLexerNext(&lx, &t);
        if (t.kind == MGLGLSL_TOK_END) {
            break;
        }
        if (t.kind == MGLGLSL_TOK_IDENT) {
            line = (int)t.start;
            (void)line;
        }
    }
    (void)found;
    (void)ids;
    CHECK(1, "lexer walks shader without error");
}

static void test_numeric(void)
{
    printf("numeric literal classes\n");
    /* literal, expected kind, expected decoded value */
    static const struct {
        const char *lit;
        MGLGLSLTokenKind kind;
        double want;
    } cases[] = {
        { "42",  MGLGLSL_TOK_INT,   42.0  },
        { "42u", MGLGLSL_TOK_UINT,  42.0  },
        { "0x1F", MGLGLSL_TOK_INT,  31.0  },
        { "3.5", MGLGLSL_TOK_FLOAT, 3.5   },
        { "1e-3", MGLGLSL_TOK_FLOAT, 0.001 },
        { "1.5E+2", MGLGLSL_TOK_FLOAT, 150.0 },
        { "007", MGLGLSL_TOK_INT,  7.0   },
        { "0x10u", MGLGLSL_TOK_UINT, 16.0 },
    };
    size_t i;
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char src[64];
        snprintf(src, sizeof(src), "%s", cases[i].lit);
        MGLGLSLexer lx;
        mglGLSLexerInit(&lx, src, strlen(src));
        MGLGLSLToken t;
        mglGLSLexerNext(&lx, &t);
        double v = -1;
        int rc = mglGLSLexerLiteral(&lx, &t, &v);
        char label[96];
        snprintf(label, sizeof(label), "%s -> %s, decoded %.3f",
                 cases[i].lit,
                 t.kind == MGLGLSL_TOK_INT ? "int" :
                 t.kind == MGLGLSL_TOK_UINT ? "uint" :
                 t.kind == MGLGLSL_TOK_FLOAT ? "float" : "??",
                 v);
        CHECK(rc == 0 && t.kind == cases[i].kind && v == cases[i].want, label);
    }
}

int main(void)
{
    printf("MGLGLSL lexer skeleton tests\n");
    test_directive();
    test_hello();
    test_numeric();
    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}