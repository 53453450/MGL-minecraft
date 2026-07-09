#!/usr/bin/env python3
# Sample pixels from a regression golden TGA (uncompressed BGR, top-left).
import sys, struct

def load(path):
    with open(path, "rb") as f:
        data = f.read()
    w = data[12] | (data[13] << 8)
    h = data[14] | (data[15] << 8)
    bpp = data[16]
    assert bpp == 24, f"expected 24bpp, got {bpp}"
    px = data[18:]
    return w, h, px

def rgb_at(w, h, px, x, y):
    # top-left origin (header[17]=0x20)
    i = (y * w + x) * 3
    b, g, r = px[i], px[i+1], px[i+2]
    return (r, g, b)

path = sys.argv[1]
w, h, px = load(path)
cx, cy = w // 2, h // 2
print(f"{path}  {w}x{h}")
print(f"  center      ({cx},{cy}): {rgb_at(w,h,px,cx,cy)}")
print(f"  left        ({w//4},{cy}): {rgb_at(w,h,px,w//4,cy)}")
print(f"  right       ({3*w//4},{cy}): {rgb_at(w,h,px,3*w//4,cy)}")
print(f"  top         ({cx},{h//4}): {rgb_at(w,h,px,cx,h//4)}")
print(f"  corner      (4,4): {rgb_at(w,h,px,4,4)}")
