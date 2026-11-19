"""Judge finished calls with an LLM: Claude Haiku on Bedrock by default, or a local Ollama model with --model ollama:qwen2:7b.

    python -m judge.judge_cli results/20261016-114914                 # judge every run in a results folder
    python -m judge.judge_cli results/<run> --model ollama:qwen2:7b --scenario book_ac_inspection --repeat 3
    python -m judge.judge_cli --transcript path/to/call.txt --scenario book_ac_inspection

Reads runs.jsonl (uses each run's "conversation" when present, otherwise parses its transcripts/*.txt file), scores
task_success / accuracy / policy / voice_style 1-5 with the same rubric as judge.py, and writes judge_local.json next to
runs.jsonl. --repeat N runs the judge N times per call and takes the median score per criterion, which steadies a
small local model. Needs `ollama serve` running and the model pulled (`ollama pull qwen2:7b`).
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from . import llm_judge as judge  # noqa: E402

DEFAULT_MODEL = judge.MODEL  # Haiku on Bedrock; pass --model ollama:qwen2:7b for the local model


def parse_transcript(path: Path) -> list[dict]:
    """Rebuild a conversation from a transcripts/*.txt file written by run_evals.py."""
    conv: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[.*?\]\s+CALLER[^:]*:\s*(.*)", line)
        if m:
            conv.append({"role": "caller", "text": m.group(1).strip()})
            continue
        m = re.match(r"\s+AGENT HEARD\s*:\s*(.*?)(\s+\(WER.*\))?$", line)
        if m:
            conv.append({"role": "heard", "text": m.group(1).strip()})
            continue
        m = re.match(r"\[.*?\]\s+AGENT\s*:\s*(.*)", line)
        if m:
            conv.append({"role": "agent", "text": m.group(1).strip()})
    return conv


def judge_once_or_median(scenario: dict, run: dict, model: str, repeat: int) -> dict:
    results = [judge.evaluate(scenario, run, model) for _ in range(repeat)]
    ok = [r for r in results if "scores" in r]
    if not ok:
        return results[0]
    scores = {k: int(statistics.median(r["scores"][k] for r in ok)) for k in judge.CRITERIA}
    spread = {k: max(r["scores"][k] for r in ok) - min(r["scores"][k] for r in ok) for k in judge.CRITERIA}
    return {
        "scores": scores, "overall": round(sum(scores.values()) / len(scores), 2),
        "pass": min(scores.values()) >= judge.PASS_AT,
        "issues": sorted({i for r in ok for i in r["issues"]})[:6], "reason": ok[0]["reason"],
        "model": model, "samples": len(ok), "unstable": [k for k, v in spread.items() if v >= 2],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", help="results folder containing runs.jsonl")
    ap.add_argument("--transcript", help="judge a single transcript .txt instead of a results folder")
    ap.add_argument("--scenario", help="scenario id (required with --transcript; otherwise filters runs)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Bedrock model id, or ollama:<name> (default %(default)s)")
    ap.add_argument("--repeat", type=int, default=1, help="judge each call N times, take the median per criterion")
    args = ap.parse_args()

    scenarios = {s["id"]: s for s in json.loads((HERE.parent / "simulator" / "scenarios.json").read_text(encoding="utf-8"))}
    jobs: list[tuple[str, dict, dict]] = []  # (label, scenario, run)
    if args.transcript:
        if not args.scenario or args.scenario not in scenarios:
            sys.exit("--transcript needs --scenario <id> from scenarios.json")
        p = Path(args.transcript)
        jobs.append((p.name, scenarios[args.scenario], {"conversation": parse_transcript(p)}))
        out = p.with_suffix(".judge_local.json")
    elif args.results:
        folder = Path(args.results)
        out = folder / "judge_local.json"
        for line in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            run = json.loads(line)
            if args.scenario and run["scenario"] != args.scenario:
                continue
            if not run.get("conversation") and run.get("transcript_file"):
                run["conversation"] = parse_transcript(folder / run["transcript_file"])
            if run["scenario"] in scenarios:
                jobs.append((f"{run.get('run_id')}:{run['scenario']}:{run['voice']}", scenarios[run["scenario"]], run))
    else:
        ap.error("give a results folder or --transcript")

    rows = []
    for label, scenario, run in jobs:
        v = judge_once_or_median(scenario, run, args.model, max(1, args.repeat))
        rows.append({"call": label, "rule_pass": run.get("passed"), **v})
        if "scores" in v:
            flag = "PASS" if v["pass"] else "FAIL"
            print(f"{flag}  {v['overall']:.2f}  {label}  {v['scores']}  {'; '.join(v['issues'])}")
        else:
            print(f"ERR   {label}  {v['error']}")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    scored = [r for r in rows if "scores" in r]
    if scored:
        agree = sum(1 for r in scored if r["rule_pass"] is not None and r["rule_pass"] == r["pass"])
        print(f"\n{len(scored)}/{len(rows)} judged, judge pass rate "
              f"{100 * sum(r['pass'] for r in scored) / len(scored):.0f}%, agrees with rule PASS/FAIL on {agree}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
