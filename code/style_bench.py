"""
Bench for Undetectable.ai's writing-style replication endpoint.

Compares several writing styles, each built from a different sample of the
corresponding author's own verified-human prose, over the same test
paragraphs. Every candidate is scored on three axes:

  integrity  - protected spans (numbers, citations, refs, defined terms)
               survive intact and in the same order
  semantics  - the rules Codex and Kimi proposed after the first pilot:
               no new or dropped negations, no new epistemic absolutes, no
               measurement turned into recommendation, defined glossary
               terms preserved verbatim
  register   - proxies for sounding machine-written: mean word length,
               count of inflated connectives, mean sentence length

Nothing here writes to the manuscript.
"""
import json
import os
import re
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from humanize_undetectable import protect, restore, api_key  # noqa: E402

STYLE = "https://style.undetectable.ai"

# Terms the paper defines and must keep verbatim. Protected like numbers so
# the model cannot reword them, and checked again afterwards.
GLOSSARY = [
    "threshold rule", "stratified test", "permutation test", "false-alarm rate",
    "minimum fold overlap", "covariate support", "leave-one-facility-out",
    "facility-level bootstrap", "ground truth", "confounded", "universal",
]

# Words that mark inflated academic register; the first pilot's outputs were
# full of them ("Consequently", "demonstrates superior performance").
INFLATED = ["consequently", "furthermore", "moreover", "demonstrates",
            "methodolog", "utilize", "leverage", "facilitate", "deployment",
            "superior performance", "in order to", "it is important to",
            "comprehensive", "robustly", "significantly enhance"]

NEGATIONS = ["not", "no", "never", "without", "cannot", "fails", "neither", "nor"]
ABSOLUTES = ["exactly", "always", "never", "guarantees", "proves", "precisely",
             "certainly", "definitely", "all cases"]
PRESCRIPTIVE = ["should", "must", "needs to", "ought to", "requires that"]


def protect_glossary(text):
    """Lift defined terms out before the text leaves, longest first."""
    store, out, i = {}, text, 0
    for term in sorted(GLOSSARY, key=len, reverse=True):
        for m in list(re.finditer(re.escape(term), out, flags=re.I)):
            i += 1
            tag = "Gx" + chr(ord("a") + i % 26) + chr(ord("a") + (i // 26) % 26) + "Gx"
            store[tag] = m.group(0)
            out = out[:m.start()] + tag + out[m.end():]
            break
    return out, store


def count(words, text):
    low = text.lower()
    return sum(low.count(w) for w in words)


def semantic_flags(original, proposed):
    """Rules from the post-pilot consultation. Each hit is a reason to look."""
    flags = []
    o, p = original.lower(), proposed.lower()
    if count(NEGATIONS, p) != count(NEGATIONS, o):
        flags.append(f"negations {count(NEGATIONS,o)}->{count(NEGATIONS,p)}")
    if count(ABSOLUTES, p) > count(ABSOLUTES, o):
        flags.append("new absolute")
    if count(PRESCRIPTIVE, p) > count(PRESCRIPTIVE, o):
        flags.append("measurement turned prescriptive")
    for term in GLOSSARY:
        if term in o and term not in p:
            flags.append(f"lost term '{term}'")
    return flags


def register_score(text):
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return {}
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return dict(
        mean_word_len=round(sum(len(w) for w in words) / len(words), 2),
        inflated=count(INFLATED, text),
        mean_sent_len=round(len(words) / max(1, len(sents)), 1),
    )


# ------------------------------------------------------------------- API
def create_style(name, sample, key):
    r = httpx.post(f"{STYLE}/writing-styles", json={"name": name, "input": sample},
                   headers={"x-api-key": key}, timeout=120)
    r.raise_for_status()
    return r.json()["id"]


def run_style(style_id, text, key, humanized=False, timeout=400):
    r = httpx.post(f"{STYLE}/requests",
                   json={"writingStyleId": style_id, "input": text,
                         "humanized": humanized},
                   headers={"x-api-key": key}, timeout=120)
    r.raise_for_status()
    rid = r.json()["id"]
    end = time.time() + timeout
    while time.time() < end:
        g = httpx.get(f"{STYLE}/requests/{rid}", headers={"x-api-key": key},
                      timeout=60).json()
        if g.get("output"):
            return g["output"]
        if g.get("status") in ("failed", "error"):
            return None
        time.sleep(5)
    return None


def evaluate(original_tex, output, num_store, glo_store):
    """Returns (restored_text, integrity_problem, flags, register)."""
    if output is None:
        return None, "timeout", [], {}
    flat = " ".join(output.split())
    missing = [k for k in list(num_store) + list(glo_store) if flat.count(k) != 1]
    if missing:
        return None, f"{len(missing)} placeholders lost", [], {}
    # order check: a reordered sentinel keeps the count but breaks the claim
    order_in = [k for k in re.findall(r"(?:Zq[a-z]+Zq|Gx[a-z]{2}Gx)", original_tex)]
    order_out = [k for k in re.findall(r"(?:Zq[a-z]+Zq|Gx[a-z]{2}Gx)", flat)]
    reordered = order_in != order_out
    restored = restore(restore(flat, glo_store), num_store)
    return restored, ("placeholders reordered" if reordered else None), \
        semantic_flags(original_tex, flat), register_score(flat)
