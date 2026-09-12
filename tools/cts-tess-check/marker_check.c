#include "mgl_tess_domain.h"
#include "glcorearb.h"
#include <stdio.h>
#include <string.h>
int main(void){
    /* the marker case: set1 inner[0] == 1 */
    MGLTessFactorInput in; memset(&in,0,sizeof in);
    in.gen_mode=GL_QUADS; in.spacing=GL_FRACTIONAL_ODD; in.winding=GL_CCW; in.point_mode=0;
    in.outer[0]=in.outer[1]=in.outer[2]=in.outer[3]=3.f;
    in.inner[0]=1.f; in.inner[1]=3.f;
    static MGLTessCoord c[200000];
    uint32_t n=mglTessGenerateDomain(&in,c,200000);
    printf("inner=(1,3) outer=3 FO: vertices=%u tris=%u\n", n, n/3);
    /* unique y values */
    float ys[64]; int ny=0;
    for(uint32_t i=0;i<n;i++){int f=0;for(int k=0;k<ny;k++)if(ys[k]==c[i].v)f=1; if(!f&&ny<64)ys[ny++]=c[i].v;}
    for(int a=0;a<ny;a++)for(int b=a+1;b<ny;b++)if(ys[b]<ys[a]){float t=ys[a];ys[a]=ys[b];ys[b]=t;}
    printf("unique y(%d): ",ny); for(int k=0;k<ny;k++)printf("%.6g ",ys[k]); printf("\n");
    printf("second_from_top=%.6g second_from_bottom=%.6g\n", ys[1], ys[ny-2]);
    /* does any triangle have two vertices at second_from_top spanning x{0,1}? */
    float y1y2 = ys[1];
    int found=0;
    for(uint32_t t=0;t<n/3;t++){
        const MGLTessCoord *a=&c[t*3], *b=&c[t*3+1], *d=&c[t*3+2];
        const MGLTessCoord *v1=0,*v2=0,*v3=0;
        if(a->v==y1y2){ if(b->v==y1y2&&d->v==0.f){v1=a;v2=b;v3=d;} else if(b->v==0.f&&d->v==y1y2){v1=a;v2=d;v3=b;} }
        else if(b->v==y1y2&&d->v==y1y2&&a->v==0.f){v1=b;v2=d;v3=a;}
        if(v1&&v2&&v3&&((v1->u==0.f&&v2->u==1.f)||(v1->u==1.f&&v2->u==0.f))) found=1;
    }
    printf("marker (y1_y2=%.6g, y3=0) found=%s\n", y1y2, found?"YES":"NO");
    /* where is the full-width inner row? */
    for(int k=0;k<ny;k++){
        int has0=0,has1=0;
        for(uint32_t i=0;i<n;i++) if(c[i].v==ys[k]){ if(c[i].u==0.f)has0=1; if(c[i].u==1.f)has1=1; }
        if(has0&&has1) printf("full-width row at y=%.6g\n", ys[k]);
    }
    return 0;}
