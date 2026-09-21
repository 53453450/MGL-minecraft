#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
"""
spec_claim_check.py — TypeSafe-assisted OpenGL SPEC conformance check.

Code owns candidate location and quote matching. TypeSafe (Jev) judges whether
the SPEC evidence supports, contradicts, or is silent about the behavioral
claim describing MGL's implementation.

Usage:
  export TYPESAFE_API_KEY=...
  .venv-typesafe/bin/python scripts/spec_claim_check.py \\
      scripts/spec_claims/draw_command.json [--id ID] [--json-out PATH]

Requires Python >= 3.10 and typesafe-sdk (see .venv-typesafe/).
Never pass API keys on the command line; never commit them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "typesafe_sdk missing. Create .venv-typesafe and pip install typesafe-sdk.\n"
        f"{exc}\n"
    )
    sys.exit(2)

DEFAULT_MODEL = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
DEFAULT_AUTO_ACCEPT = float(os.environ.get("SPEC_CLAIM_AUTO_ACCEPT", "0.8"))

QUESTIONS = {
    "relation": Choice(
        instructions={
            "task": (
                "Judge how the OpenGL SPEC evidence relates to the behavioral "
                "claim about MGL's implementation."
            ),
            "rules": [
                "Treat `claim` as a statement of what MGL does.",
                "Treat `spec_evidence` as the sole SPEC authority for this judgment.",
                "Treat `code_excerpt` only as context for what the claim asserts; "
                "do not invent SPEC text from the code.",
                "If the SPEC marks the situation undefined or implementation-"
                "dependent and the claim describes one possible outcome without "
                "requiring a specific error, choose unspecified.",
                "If the SPEC recommends but does not require a behavior, and the "
                "claim matches the recommendation, choose supports.",
                "If the claim asserts MGL generates a required error that the "
                "SPEC demands, and the code excerpt shows that error path, that "
                "is still judged against the SPEC requirement (supports if SPEC "
                "requires it).",
                "If the claim asserts a required SPEC error but the described "
                "behavior omits that error or substitutes a different outcome, "
                "choose contradicts.",
            ],
        },
        criteria={
            "supports": (
                "The SPEC requires or explicitly recommends the behavior "
                "described by the claim (including recommended INVALID_VALUE "
                "for otherwise-undefined cases)."
            ),
            "contradicts": (
                "The SPEC requires a different outcome than the claim "
                "(for example a mandatory error that the claim says is omitted, "
                "or a different error code)."
            ),
            "unspecified": (
                "The SPEC leaves the situation undefined, implementation-"
                "dependent, or silent; the claim is one allowed reading but "
                "not required."
            ),
            "says_nothing": (
                "The provided SPEC excerpt does not address the claim at all."
            ),
        },
    ),
}

VERDICT = {
    "supports": "conforms",
    "contradicts": "violates",
    "unspecified": "unspecified",
    "says_nothing": "no_evidence",
}


def normalize(text: str) -> str:
    table = str.maketrans(
        {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u2212": "-",
            "\ufb01": "fi",
            "\ufb02": "fl",
        }
    )
    return re.sub(r"\s+", " ", text.translate(table)).strip().lower()


def quote_present(haystack: str, quote: str | None) -> bool | None:
    """True/False if quote searchable; None if no quote supplied.

    Accepts a substantial leading prefix when PDF/HTML layout truncates the
    trailing constant-table row from the evidence window.
    """
    if not quote:
        return None
    hay = normalize(haystack)
    q = normalize(quote)
    if q in hay:
        return True
    head = q.split("...")[0].strip()
    if len(head) >= 24 and head in hay:
        return True
    if len(q) >= 72 and q[:72] in hay:
        return True
    return False


def load_suite(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("claims"), list) or not data["claims"]:
        raise SystemExit(f"no claims in {path}")
    return data


def ask(client: TypeSafeClient, claim: dict, model: str) -> dict:
    state = {
        "claim": claim["claim"],
        "code_locus": claim.get("code_locus", ""),
        "code_excerpt": claim.get("code_excerpt", ""),
        "spec_section": claim.get("spec_section", ""),
        "spec_evidence": claim.get("evidence")
        or claim.get("spec_quote", ""),
        "spec_source": claim.get("spec_source", ""),
        "evidence_verified": bool(claim.get("evidence_verified")),
    }
    started = perf_counter()
    last_err: Exception | None = None
    response = None
    for attempt in range(3):
        try:
            response = client.system_one(
                state=state, questions=QUESTIONS, model=model, timeout=60.0
            )
            break
        except Exception as exc:  # network timeouts only
            last_err = exc
            if "Timeout" not in type(exc).__name__ and "timeout" not in str(exc).lower():
                raise
            import time

            time.sleep(1.5 * (attempt + 1))
    else:
        assert last_err is not None
        raise last_err

    answer = response.answers["relation"]
    choice = answer.choice
    return {
        "choice": choice,
        "verdict": VERDICT.get(choice, choice),
        "probabilities": dict(answer.probabilities),
        "confidence": float(answer.confidence),
        "seconds": round(perf_counter() - started, 2),
        "input_tokens": getattr(response.usage, "input_tokens", None) or 0,
        "output_tokens": getattr(response.usage, "output_tokens", None) or 0,
        "request_id": getattr(response, "request_id", None),
    }


def check_one(client: TypeSafeClient, claim: dict, model: str, auto_accept: float) -> dict:
    # Self-contained claims embed the quote as evidence; optional larger blob
    # under claim["evidence"] is preferred when present.
    evidence = claim.get("evidence") or claim.get("spec_quote") or ""
    present = quote_present(evidence, claim.get("spec_quote"))

    row = {
        "id": claim["id"],
        "claim": claim["claim"],
        "spec_section": claim.get("spec_section"),
        "code_locus": claim.get("code_locus"),
        "quote_in_evidence": present,
    }

    if present is False:
        row.update(
            {
                "choice": None,
                "verdict": "fabricated_quote",
                "confidence": None,
                "auto": True,
                "answer": None,
            }
        )
        return row

    answer = ask(client, claim, model)
    auto = answer["confidence"] >= auto_accept
    row.update(
        {
            "choice": answer["choice"],
            "verdict": answer["verdict"],
            "confidence": answer["confidence"],
            "auto": auto,
            "answer": answer,
        }
    )
    return row


def format_table(rows: list[dict]) -> str:
    lines = [
        f"{'id':<42}{'relation':<14}{'conf':>6}  {'verdict':<16}{'action':>8}"
    ]
    lines.append("-" * len(lines[0]))
    for row in rows:
        relation = row.get("choice") or "-"
        conf = (
            f"{row['confidence']:.2f}"
            if isinstance(row.get("confidence"), float)
            else "-"
        )
        action = "auto" if row.get("auto") else "review"
        lines.append(
            f"{row['id']:<42}{relation:<14}{conf:>6}  "
            f"{row['verdict']:<16}{action:>8}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claims_json", type=Path, help="suite JSON path")
    parser.add_argument("--id", action="append", dest="ids", help="only these claim ids")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--auto-accept",
        type=float,
        default=DEFAULT_AUTO_ACCEPT,
        help="confidence threshold for auto action (default env/0.8)",
    )
    args = parser.parse_args()
    auto_accept = args.auto_accept

    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.stderr.write("TYPESAFE_API_KEY is not set.\n")
        return 2

    suite = load_suite(args.claims_json)
    claims = suite["claims"]
    if args.ids:
        want = set(args.ids)
        claims = [c for c in claims if c["id"] in want]
        missing = want - {c["id"] for c in claims}
        if missing:
            sys.stderr.write(f"unknown claim ids: {sorted(missing)}\n")
            return 2

    print(
        f"suite={suite.get('suite')} claims={len(claims)} "
        f"model={args.model} auto_accept={auto_accept}"
    )

    rows: list[dict] = []
    with TypeSafeClient(timeout=60.0) as client:
        for claim in claims:
            print(f"... {claim['id']}", flush=True)
            rows.append(check_one(client, claim, args.model, auto_accept))

    print()
    print(format_table(rows))

    summary = {
        "suite": suite.get("suite"),
        "spec": suite.get("spec"),
        "model": args.model,
        "auto_accept": auto_accept,
        "results": rows,
        "counts": {
            "conforms": sum(1 for r in rows if r["verdict"] == "conforms"),
            "violates": sum(1 for r in rows if r["verdict"] == "violates"),
            "unspecified": sum(1 for r in rows if r["verdict"] == "unspecified"),
            "no_evidence": sum(1 for r in rows if r["verdict"] == "no_evidence"),
            "fabricated_quote": sum(
                1 for r in rows if r["verdict"] == "fabricated_quote"
            ),
            "needs_review": sum(1 for r in rows if not r.get("auto")),
        },
    }
    print()
    print("counts:", json.dumps(summary["counts"], ensure_ascii=False))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")

    hard = [r for r in rows if r["verdict"] == "violates" and r.get("auto")]
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
