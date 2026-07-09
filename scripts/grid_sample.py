#!/usr/bin/env python3
import sys
def load(path):
    with open(path,"rb") as f: d=f.read()
    w=d[12]|(d[13]<<8); h=d[14]|(d[15]<<8)
    return w,h,d[18:]
def rgb(w,px,x,y):
    i=(y*w+x)*3; return (px[i+2],px[i+1],px[i])
path=sys.argv[1]; w,h,px=load(path)
# distinct color histogram
from collections import Counter
c=Counter()
for y in range(0,h,2):
    for x in range(0,w,2):
        c[rgb(w,px,x,y)]+=1
print(path)
for col,n in c.most_common(6):
    print(f"  {col}: {n}")
