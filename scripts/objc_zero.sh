#!/bin/bash
# ObjC 清零度量（docs/OBJC_CATEGORY_DISMANTLE_TODO.md §0.0 的口径）。
# 输出：.m/.mm 文件数、空 TU 数、ObjC 语法出现次数、ObjC 词汇出现次数、MGLRenderer*.m 合计。
set -u
cd "$(dirname "$0")/.." || exit 1

python3 - <<'PY'
import re, glob
SYNTAX = re.compile(r'@(interface|implementation|protocol|end|autoreleasepool|selector|encode|property|synthesize|try|catch|finally|synchronized)\b|\[[A-Za-z_][A-Za-z0-9_\.]*\s+[A-Za-z_]|__bridge|__weak|__strong|#\s*import\b')
VOCAB  = re.compile(r'\b(BOOL|YES|NO|nil|Nil|NSInteger|NSUInteger|CGFloat|SEL|IMP|NSObject|NS[A-Z]\w*|MTL[A-Z]\w*)\b')

def strip(src):
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', src)

files = sorted(f for f in glob.glob('MGL/**/*.m', recursive=True) + glob.glob('MGL/**/*.mm', recursive=True))
empty = 0
syntax_total = vocab_total = line_total = 0
per = []
for f in files:
    src = open(f, errors='ignore').read()
    body = strip(src)
    lines = len(src.split('\n'))
    line_total += lines
    syn = len(SYNTAX.findall(body))
    voc = len(VOCAB.findall(body))
    syntax_total += syn
    vocab_total += voc
    if not body.strip():
        empty += 1
    per.append((syn, voc, lines, f))

print(f"==> ObjC 冻结面（MGL/ 内 .m/.mm）")
print(f"      文件数            : {len(files)}")
print(f"      其中空 TU          : {empty}")
print(f"      文件行数合计       : {line_total}")
print(f"      ObjC 语法出现次数  : {syntax_total}")
print(f"      ObjC 词汇出现次数  : {vocab_total}")
print(f"\n==> 逐文件（ObjC 语法 / 词汇 / 行数）")
for syn, voc, lines, f in sorted(per, key=lambda r: -r[2]):
    tag = 'EMPTY' if (syn == 0 and voc == 0 and lines < 60) else ('PURE' if syn == 0 and voc == 0 else '')
    print(f"      {syn:5d} {voc:5d} {lines:6d}  {f}  {tag}")
PY

if [ -x scripts/objc_renderer_loc.sh ]; then
    scripts/objc_renderer_loc.sh 2>/dev/null | grep "MGLRenderer\*.m total" | sed 's/^/==> /'
fi
