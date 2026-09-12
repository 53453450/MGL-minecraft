/* Offline Rule 3 / Rule 4 checker for quad domain edges.
 * Reads MGL's point-mode output and applies ARB_tessellation_shader
 * Appendix A.X rules directly (no CTS involved). */
#include "mgl_tess_domain.h"
#include "glcorearb.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAXP 300000
static MGLTessCoord pts[MAXP];

static int has_pt(float x, float y)
{
    for (int i = 0; pts[i].u || pts[i].v || i == 0; i++) { }
    return 0;
}

int main(int argc, char **argv)
{
    uint32_t spacing = (uint32_t)strtoul(argv[1], 0, 10);
    float o0 = atof(argv[2]), o1 = atof(argv[3]), o2 = atof(argv[4]), o3 = atof(argv[5]);
    float i0 = atof(argv[6]), i1 = atof(argv[7]);
    MGLTessFactorInput in; memset(&in, 0, sizeof in);
    in.gen_mode = GL_QUADS; in.spacing = spacing; in.winding = GL_CCW;
    in.point_mode = 1;
    in.outer[0]=o0; in.outer[1]=o1; in.outer[2]=o2; in.outer[3]=o3;
    in.inner[0]=i0; in.inner[1]=i1;
    uint32_t n = mglTessGenerateDomain(&in, pts, MAXP);

    /* bottom edge interior points (v==0, 0<u<1) and left edge (u==0, 0<v<1) */
    float bot[256], left[256]; int nb = 0, nl = 0;
    for (uint32_t k = 0; k < n; k++) {
        if (pts[k].v == 0.f && pts[k].u > 0.f && pts[k].u < 1.f) {
            int f = 0; for (int j = 0; j < nb; j++) if (bot[j] == pts[k].u) f = 1;
            if (!f && nb < 256) bot[nb++] = pts[k].u;
        }
        if (pts[k].u == 0.f && pts[k].v > 0.f && pts[k].v < 1.f) {
            int f = 0; for (int j = 0; j < nl; j++) if (left[j] == pts[k].v) f = 1;
            if (!f && nl < 256) left[nl++] = pts[k].v;
        }
    }

    printf("spacing=%u outer=(%g,%g,%g,%g) inner=(%g,%g) points=%u bottom=%d left=%d\n",
           spacing, o0,o1,o2,o3, i0,i1, n, nb, nl);

    /* Rule 3 on the bottom edge: for each x, 1-x must also be present */
    int r3 = 0;
    for (int a = 0; a < nb; a++) {
        float c = 1.f - bot[a]; int f = 0;
        for (int b = 0; b < nb; b++) if (bot[b] == c) f = 1;
        if (!f) { if (r3 < 3) printf("  RULE3 miss: 1-%.9g=0x%08x not on bottom edge\n", bot[a], *(unsigned*)&c); r3++; }
    }
    /* Rule 4: if (x,0) and (1-x,0) both exist, (0,x) and (0,1-x) must */
    int r4 = 0;
    for (int a = 0; a < nb; a++) {
        float x = bot[a], c = 1.f - x; int hc = 0;
        for (int b = 0; b < nb; b++) if (bot[b] == c) hc = 1;
        if (!hc) continue;
        int hx = 0, hxc = 0;
        for (int b = 0; b < nl; b++) { if (left[b] == x) hx = 1; if (left[b] == c) hxc = 1; }
        if (!hx || !hxc) { if (r4 < 3) printf("  RULE4 miss: x=%.9g 0x%08x -> (0,x)=%s (0,1-x)=%s\n", x, *(unsigned*)&x, hx?"Y":"N", hxc?"Y":"N"); r4++; }
    }
    printf("  RULE3 violations=%d  RULE4 violations=%d  => %s\n",
           r3, r4, (r3 == 0 && r4 == 0) ? "PASS" : "FAIL");
    return (r3 || r4) ? 1 : 0;
}
