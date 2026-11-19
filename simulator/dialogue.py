"""Caller scripts for a scenario: a fixed list of turns, or a reactive dialogue.

Fixed (the original form):
    "turns": ["Hi, I would like to book ...", "Yes, that price is fine. Please go ahead."]
    The caller says each line in order, whatever the agent says.

Reactive:
    "dialogue": {
      "start": "Hi, I would like to book an inspection for my air conditioner.",
      "replies": [
        {"when": "name and (?:your )?phone", "say": "My name is Sam Rivera and my phone number is five five five, zero one zero zero."},
        {"when": "phone|number",             "say": "My phone number is five five five, zero one zero zero."},
        {"when": "go ahead|confirm|price",   "say": "Yes, that price is fine. Please go ahead and book it.", "once": true}
      ],
      "fallback": "Sorry, could you say that again?",     # optional: said when no rule matches
      "max_turns": 8                                       # optional cap on caller lines (default 8)
    }
    After each agent turn, the first rule whose `when` regex (case-insensitive) matches what the agent just said, and that
    has not been used yet (every rule is `once` unless "once": false), supplies the next caller line. Rules are tried
    against the end of the agent's reply first (last sentence, then last two, then all of it), because replies often
    restate details before asking their question. A rule with "wrong_digits": "5550100" only applies when the agent has
    just read a phone number back that is NOT that one (use it with "when": ".", "once": false, listed first, to make the
    caller correct a bad read-back instead of agreeing to it). With no match and no fallback the caller hangs up. Every possible line is a stored clip or a TTS render, so reactive calls work with
    clip:... voices too: record them with `python evals/clips.py record`.
"""
import re


def _wrong_readback(agent_text: str, expected: str) -> bool:
    """True if the agent read a phone number back (5+ digits in a sentence about the number) that is not `expected`."""
    from . import metrics  # merges "five five five" / "5-55" style digit runs into one token

    for sent in re.split(r"(?<=[.?!])\s+", agent_text):
        if re.search(r"number|phone", sent, re.I):
            digits = "".join(t for t in metrics.normalize(sent) if t.isdigit())
            if len(digits) >= 5 and expected not in digits:
                return True
    return False


class Script:
    def __init__(self, scenario: dict) -> None:
        d = scenario.get("dialogue")
        self.reactive = d is not None
        self._used: set[int] = set()
        self._n = 0  # lines said so far
        if self.reactive:
            self.start: str = d["start"]
            self.rules: list[dict] = d.get("replies", [])
            self.fallback: str | None = d.get("fallback")
            self.max_turns: int = d.get("max_turns", 8)
            raw = [self.start] + [r["say"] for r in self.rules] + ([self.fallback] if self.fallback else [])
        else:
            self.turns: list[str] = list(scenario["turns"])
            raw = self.turns
        self._lines = list(dict.fromkeys(raw))  # every line the caller might say, in a stable order

    def lines(self) -> list[str]:
        """All lines to render or look up before the call (index = the line number used for clip ids)."""
        return self._lines

    def planned(self) -> list[str]:
        """What the caller would say with no agent in the loop; used to score a call that never got going."""
        return [self.start] if self.reactive else self.turns

    def first(self) -> str:
        self._n = 1
        return self.start if self.reactive else self.turns[0]

    def next(self, agent_text: str) -> str | None:
        """The caller's next line given what the agent just said, or None when the caller is done."""
        if not self.reactive:
            self._n += 1
            return self.turns[self._n - 1] if self._n <= len(self.turns) else None
        if self._n >= self.max_turns:
            return None
        # A reply often restates things before asking its question, so look at the end of it first: the last
        # sentence, then the last two, then everything.
        sentences = [x for x in re.split(r"(?<=[.?!])\s+", agent_text.strip()) if x]
        for window in (sentences[-1:], sentences[-2:], sentences):
            text = " ".join(window)
            for k, rule in enumerate(self.rules):
                if k in self._used and rule.get("once", True):
                    continue
                if "wrong_digits" in rule and not _wrong_readback(agent_text, rule["wrong_digits"]):
                    continue
                if re.search(rule["when"], text, re.I):
                    self._used.add(k)
                    self._n += 1
                    return rule["say"]
        if self.fallback:
            self._n += 1
            return self.fallback
        return None
