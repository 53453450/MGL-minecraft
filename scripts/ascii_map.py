#!/usr/bin/env python3
import sys
def load(path):
    with open(path,"rb") as f: d=f.read()
    w=d[12]|(d[13]<<8); h=d[14]|(d[15]<<8)
    return w,h,d[18:]
def rgb(w,px,x,y):
    i=(y*w+x)*3; return (px[i+2],px[i+1],px[i])
def sym(c):
    r,g,b=c
    if (r,g,b)==(26,26,26): return '.'
    if r>150 and g<100 and b<100: return 'R'
    if g>150 and r<100 and b<100: return 'G'
    if b>150 and r<100 and g<100: return 'B'
    return '?'
path=sys.argv[1]; w,h,px=load(path)
step=max(1,w//48)
for y in range(0,h,step*2):
    row="".join(sym(rgb(w,px,x,y)) for x in range(0,w,step))
    print(row)
