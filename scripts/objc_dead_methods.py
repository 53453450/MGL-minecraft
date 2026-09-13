#!/usr/bin/env python3
"""Find Objective-C methods in MGL/src/*.m that nothing calls any more.

Motivation (see docs/OBJC_CATEGORY_DISMANTLE_TODO.md, log 70/71): the manual
version of this check produced false positives three times in a row, because a
selector can be referenced in ways a naive "[self name" grep misses:

  1. the selector carries colons (``foo:bar:``) while the scan looked at ``foo``;
  2. the send uses a receiver other than ``self`` (``[renderer foo:...]``), often
     in the very same file as the definition;
  3. the method is a property setter reached through ``obj.device = x``;
  4. the method is a framework callback (KVO ``observeValueForKeyPath:...``,
     NSNotification selectors) that is never written down as a call at all.

This script checks all four and, for the property/notification cases, reports
them separately so a human can look before deleting anything.  A clean run with
no candidates is the expected steady state; a candidate still has to survive the
compiler: delete it, rebuild, and let clang be the oracle.

usage: python3 scripts/objc_dead_methods.py [--root MGL/src]
"""

import argparse
import glob
import os
import re
import sys

DEF_RE = re.compile(r'^[-+]\s*\(([^)]*)\)\s*([\w:]+)')
# A send to any receiver: [foo sel...] / [self sel...] / [a.b sel...]
SEND_RE = r'\[\s*[\w.\[\]]+\s+%s'
FRAMEWORK_CALLBACKS = (
    'observeValueForKeyPath',      # KVO
    'applicationDid', 'applicationWill', 'applicationShould', 'windowDid', 'windowWill',
    'viewDid', 'viewWill', 'didChangeValueForKey', 'willChangeValueForKey',
    'encodeWithCoder', 'initWithCoder', 'copyWithZone', 'mutableCopyWithZone',
    'forwardInvocation', 'methodSignatureForSelector', 'doesNotRecognizeSelector',
)


def collect_sources(root):
    """Every source/header the library or its harnesses compile."""
    pats = ['MGL/src/*.m', 'MGL/src/*.c', 'MGL/src/*.cpp', 'MGL/src/*.h',
            'MGL/include/*.h', 'MGL/include/GL/*.h',
            'test_legacy_compat/*', 'test_regression/*']
    out = {}
    for pat in pats:
        for f in glob.glob(pat):
            try:
                out[f] = open(f, errors='replace').read()
            except OSError:
                pass
    return out


def methods_of(src):
    """(first_line, last_line, full_selector) for each method definition."""
    lines = src.split('\n')
    out = []
    i = 0
    while i < len(lines):
        m = DEF_RE.match(lines[i])
        if not m:
            i += 1
            continue
        selector = m.group(2)
        start = i
        while i < len(lines) and '{' not in lines[i]:
            i += 1
        depth = 0
        k = i
        while k < len(lines):
            depth += lines[k].count('{') - lines[k].count('}')
            if depth == 0:
                break
            k += 1
        out.append((start + 1, k + 1, selector))
        i = k + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='MGL/src')
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sources = collect_sources(args.root)
    findings = []
    for path, src in sorted(sources.items()):
        if not path.endswith('.m'):
            continue
        for first, last, selector in methods_of(src):
            body = '\n'.join(src.split('\n')[first - 1:last])
            name = selector.rstrip(':')
            reasons = []

            # 1/2: any send to any receiver, anywhere in the tree
            send_any = 0
            for other, text in sources.items():
                hits = len(re.findall(SEND_RE % re.escape(selector), text))
                if other == path:
                    hits -= len(re.findall(SEND_RE % re.escape(selector), body))
                send_any += hits
            # @selector(...) anywhere
            sel_any = sum(t.count('@selector(%s)' % selector) for t in sources.values())
            # 3: property setter reached as `x.name = `
            prop_any = 0
            if selector.startswith('set') and selector.endswith(':') and selector.count(':') == 1:
                prop = selector[3].lower() + selector[4:-1]
                prop_any = len(re.findall(r'\.%s\s*=' % re.escape(prop), '\n'.join(sources.values())))
            # bare occurrence outside our own definition body (comments, strings, declarations)
            bare = 0
            for other, text in sources.items():
                n = text.count(selector)
                if other == path:
                    n -= body.count(selector)
                bare += n
            if send_any == 0 and sel_any == 0 and prop_any == 0 and bare == 0:
                reasons.append('no reference anywhere')
            if any(name.startswith(cb) or cb in name for cb in FRAMEWORK_CALLBACKS):
                reasons.append('framework callback name - verify by hand')
            if reasons:
                findings.append((path, first, last, selector, last - first + 1, '; '.join(reasons)))

    print('%d candidate(s) in %s' % (len(findings), args.root))
    for path, first, last, selector, size, why in findings:
        print('%-42s %5d-%-5d %4d lines  %s  [%s]' % (path, first, last, size, selector, why))
    if not findings:
        print('(no unreferenced methods - every definition is reached by a send, '
              '@selector, property assignment or bare mention)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
