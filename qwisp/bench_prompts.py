"""Benchmark-derived prompts for the Qwisp measurement ref set.

Regimes (Step1 mix + short-nl), each tagged with its benchmark source:
  code     — code generation.        source: LiveCodeBench / SWE-bench style (function impl).
  agentic  — tool / function calling. source: BFCL (Berkeley Function Calling Leaderboard) format.
  longctx  — long-context retrieval.  source: RULER needle-in-haystack (synthetic-by-design recipe).
  shortnl  — short-form chat.         source: MT-Bench (LMSYS) question 81 (verbatim).

Each entry: {"text", "ctx", "source"}. `ctx` = target prompt length in tokens (None = use the
prompt as-is; bench_refs pads short prompts by repetition and truncates long ones).

Fidelity note: shortnl is a verbatim MT-Bench question; longctx uses RULER's actual synthetic
needle recipe; code/agentic are representative samples in their benchmarks' real formats. Exact
dataset instances can be swapped in without changing the harness.
"""

_CODE = '''You are an expert Python programmer. Implement the function below. Respond with code only.

def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge all overlapping intervals and return them sorted by start.
    Example: merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    """
'''

_AGENTIC = '''You are a function-calling assistant. Available tools:
[
 {"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}},
 {"name":"book_flight","description":"Book a one-way flight","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string","description":"YYYY-MM-DD"}},"required":["origin","destination","date"]}},
 {"name":"convert_currency","description":"Convert an amount between currencies","parameters":{"type":"object","properties":{"amount":{"type":"number"},"from_ccy":{"type":"string"},"to_ccy":{"type":"string"}},"required":["amount","from_ccy","to_ccy"]}}
]
Emit the needed tool calls as a JSON list to answer:
User: What's the weather in Tokyo in celsius, book me a flight from SFO to Tokyo on 2026-08-01, and convert 500 USD to JPY.
Assistant:'''

# MT-Bench (LMSYS) question 81, verbatim (writing category).
_SHORTNL = ("Compose an engaging travel blog post about a recent trip to Hawaii, "
            "highlighting cultural experiences and must-see attractions.")


def _build_ruler() -> str:
    """RULER needle-in-haystack: distractor filler with one inserted fact, then a query.
    The needle is placed within the first portion so it survives moderate truncation."""
    filler = ("The garden was quiet in the afternoon sun. A gentle breeze moved through the "
              "trees while birds called from the branches. People walked slowly along the path, "
              "talking about ordinary things and enjoying the weather. ")
    needle = " One special fact to remember: the access code for the Qwisp vault is 48271. "
    body = (filler * 12) + needle + (filler * 18)
    q = ("\n\nQuestion: What is the access code for the Qwisp vault? "
         "Answer with the number only.\nAnswer:")
    return body + q


PROMPTS = {
    "code":    {"text": _CODE,          "ctx": None,   "source": "LiveCodeBench/SWE-bench-style (representative)"},
    "agentic": {"text": _AGENTIC,       "ctx": None,  "source": "BFCL format (representative)"},
    "longctx": {"text": _build_ruler(), "ctx": None, "source": "RULER needle-in-haystack (synthetic recipe)"},
    "shortnl": {"text": _SHORTNL,       "ctx": None,   "source": "MT-Bench Q81 (verbatim)"},
}
