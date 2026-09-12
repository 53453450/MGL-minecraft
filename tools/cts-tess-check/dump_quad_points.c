/* Dump MGL's point-mode quad domain as x y per line, for the CTS simulator. */
#include "mgl_tess_domain.h"
#include "glcorearb.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(int argc, char **argv){
    if (argc < 8) { fprintf(stderr,"usage: %s mode spacing i0 i1 o0 o1 o2 o3\n", argv[0]); return 2; }
    MGLTessFactorInput in; memset(&in,0,sizeof in);
    in.gen_mode = GL_QUADS;
    in.spacing = (uint32_t)strtoul(argv[2],0,10);
    in.winding = GL_CCW; in.point_mode = 1;
    in.inner[0]=(float)atof(argv[3]); in.inner[1]=(float)atof(argv[4]);
    for (int i=0;i<4;i++) in.outer[i]=(float)atof(argv[5+i]);
    static MGLTessCoord buf[300000];
    uint32_t n=mglTessGenerateDomain(&in,buf,300000);
    printf("%u\n",n);
    for(uint32_t i=0;i<n;i++) printf("%.9g %.9g\n",buf[i].u,buf[i].v);
    return 0;}
