"""Scoring helpers: word error rate (jiwer), tool-call matching and outcome checks."""
import re
import statistics

import jiwer

_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, spoken digits -> digits, and merge adjacent digit tokens so
    "five five five" / "555" / "5 5 5" all compare equal."""
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    tokens = [_DIGIT_WORDS.get(t, t) for t in tokens]
    out: list[str] = []
    for t in tokens:
        if out and t.isdigit() and out[-1].isdigit():
            out[-1] += t
        else:
            out.append(t)
    return out


def word_error_detail(reference: str, hypothesis: str) -> dict:
    """jiwer word-level alignment of the normalised texts: errors, ref words, substitutions, deletions, insertions."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref or not hyp:  # jiwer rejects empty strings
        return {"errors": len(ref) + len(hyp), "words": len(ref), "sub": 0, "del": len(ref), "ins": len(hyp)}
    o = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return {"errors": o.substitutions + o.deletions + o.insertions, "words": len(ref),
            "sub": o.substitutions, "del": o.deletions, "ins": o.insertions}


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(edit distance in words, reference length)."""
    d = word_error_detail(reference, hypothesis)
    return d["errors"], d["words"]


# ------------------------------------------------------------------- tools ---
def _args_match(expected: dict, actual: dict) -> bool:
    """Every expected key must match (case-insensitive). A '~' key prefix means 'contains'."""
    for key, want in expected.items():
        contains = key.startswith("~")
        got = actual.get(key.lstrip("~"))
        if got is None:
            return False
        got, want = str(got).lower(), str(want).lower()
        if (want not in got) if contains else (got != want):
            return False
    return True


def match_tools(expected: list[dict], actual: list[dict]) -> dict:
    """Checks that the expected calls occurred in order (extra calls in between are allowed).
    An expected call with "optional": true is neither required nor penalised."""
    pos, hits, required = 0, 0, 0
    missing = []
    for exp in expected:
        found = next((i for i in range(pos, len(actual))
                      if actual[i]["name"] == exp["name"] and _args_match(exp.get("args", {}), actual[i]["args"])), None)
        if found is not None:
            pos = found + 1
        if exp.get("optional"):
            continue
        required += 1
        hits += found is not None
        if found is None:
            missing.append(exp)
    return {"score": hits / required if required else 1.0, "missing": missing}


def _dig(obj: dict, path: str):
    for part in path.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj


def check_outcome(expect: dict | None, tool_calls: list[dict]) -> tuple[bool, str]:
    """expect = {"created": bool, "fields": {"quote.item_id": "AC-100", "~customer_name": "sam"}}"""
    if not expect:
        return True, ""
    created = [t["result"] for t in tool_calls
               if t["name"] == "create_service_request" and isinstance(t["result"], dict) and "request_id" in t["result"]]
    if bool(created) != expect.get("created", True):
        return False, "request created" if created else "no request created"
    for key, want in expect.get("fields", {}).items():
        got = _dig(created[-1], key.lstrip("~")) if created else None
        ok = got is not None and ((str(want).lower() in str(got).lower()) if key.startswith("~")
                                  else str(got).lower() == str(want).lower())
        if not ok:
            return False, f"{key.lstrip('~')}={got!r}, wanted {want!r}"
    return True, ""


# ----------------------------------------------------------------- summary ---
def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate(results: list[dict], key: str) -> list[dict]:
    """Per-group stats (key = "voice" or "scenario") used by both the console table and the HTML report."""
    rows = []
    for name in sorted({r[key] for r in results}):
        rs = [r for r in results if r[key] == name]
        words = sum(r["wer_words"] for r in rs)
        ttfa = [t["ttfa_ms"] for r in rs for t in r["turns"] if t.get("ttfa_ms") is not None]
        tool_ms = [t["tool_ms"] for r in rs for t in r["turns"] if t.get("tool_ms") is not None]
        judged = [r["judge"] for r in rs if r.get("judge") and "scores" in r["judge"]]
        rows.append({
            "judge": mean([j["overall"] for j in judged]), "judge_pass": mean([float(j["pass"]) for j in judged]),
            "name": name, "runs": len(rs), "passed": sum(r["passed"] for r in rs),
            "pass_rate": sum(r["passed"] for r in rs) / len(rs),
            "tool_score": mean([r["tool_score"] for r in rs]),
            "wer": sum(r["wer_errors"] for r in rs) / words if words else None,
            "turns": sum(len(r["turns"]) for r in rs),
            "early_turns": sum(1 for r in rs for t in r["turns"] if t.get("early_ms")),
            "ttfa_p50": pct(ttfa, 0.5), "ttfa_p95": pct(ttfa, 0.95), "tool_ms": mean(tool_ms),
        })
    return rows
