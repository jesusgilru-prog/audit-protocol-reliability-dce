"""
Humanization pass through Undetectable.ai, LaTeX-aware.

Two departures from the sentence-level approach used on earlier papers,
both aimed at the failure mode recorded there (the API returning degenerate
output on technical prose: clauses repeated, words truncated, parentheses
broken, "et al." severed):

1. PARAGRAPH granularity instead of sentence. Short technical sentences
   give the model almost no context to work with, which is the likeliest
   cause of the fragmenting. Paragraphs give it room.

2. PLACEHOLDER PROTECTION. Everything fragile is lifted out before the text
   leaves: inline maths, \\citep/\\ref/\\texttt/\\emph arguments, and every
   numeric literal. Each becomes an opaque sentinel the model has no reason
   to alter, and the originals are restored afterwards. A response that
   loses or mangles a sentinel is rejected outright, so no number, citation
   or cross-reference can be corrupted by a substitution that slips through.

Nothing is written back to the manuscript automatically. The script emits a
review file of ORIGINAL/PROPOSED pairs; accepting them is a separate,
deliberate step.
"""
import json
import os
import re
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = "/home/jesus/paper_ai_checker/.env"
SUBMIT = "https://humanize.undetectable.ai/submit"
DOCUMENT = "https://humanize.undetectable.ai/document"

# Config confirmed against the vendor's own API reference: v11sr is the
# slowest and best English model; University/Article match the register.
CFG = dict(readability="University", purpose="Article",
           model="v11sr", strength="More Human")

MIN_LEN = 50        # API minimum for `content`
MAX_LEN = 12000     # keep a paragraph in one request


def api_key():
    for line in open(ENV):
        if line.startswith("UNDETECTABLE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no API key")


# --------------------------------------------------------------- protection
# Order matters: maths and braced commands first, bare numbers last, so a
# number already inside a protected span is not double-substituted.
PATTERNS = [
    r"\\(?:citep|citet|cite|ref|eqref|label)\{[^}]*\}",
    r"\\(?:texttt|emph|textbf|textit|textsc)\{[^}]*\}",
    r"\$[^$]*\$",
    r"\\[a-zA-Z]+\*?",
    r"\d+(?:[.,]\d+)*\s*\\?%?",
]


def protect(text):
    store, out, idx = {}, text, 0
    for pat in PATTERNS:
        def sub(m):
            nonlocal idx
            idx += 1
            # letters only: a sentinel containing digits would be re-matched
            # by the numeric pattern below and nested inside another sentinel
            tag = ""
            n = idx
            while True:
                tag = chr(ord("a") + n % 26) + tag
                n //= 26
                if n == 0:
                    break
            key = f"Zq{tag}Zq"
            store[key] = m.group(0)
            return key
        out = re.sub(pat, sub, out)
    return out, store


def restore(text, store):
    for key, val in store.items():
        text = text.replace(key, val)
    return text


def all_sentinels_intact(text, store):
    return all(text.count(k) == 1 for k in store)


# ------------------------------------------------------------------ quality
def degenerate(original, proposed):
    """Reject the known failure modes rather than trusting the output."""
    o, p = original.split(), proposed.split()
    if not p:
        return "empty"
    ratio = len(p) / max(1, len(o))
    if ratio < 0.6 or ratio > 1.8:
        return f"length ratio {ratio:.2f}"
    low = proposed.lower()
    # a content trigram repeated verbatim inside one paragraph
    toks = [w for w in re.findall(r"[a-z]{4,}", low)]
    tri = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
    if tri and max(tri.count(t) for t in set(tri)) > 2:
        return "trigram repetition"
    if re.search(r"\b\w{1,3}\.\.\.|\w-\s*$", proposed):
        return "truncated word"
    if proposed.count("(") != proposed.count(")"):
        return "unbalanced parentheses"
    return None


# --------------------------------------------------------------------- API
def submit(text, key):
    body = dict(content=text, **CFG)
    r = httpx.post(SUBMIT, json=body, headers={"apikey": key}, timeout=60)
    r.raise_for_status()
    return r.json()["id"]


def fetch(doc_id, key, timeout=420):
    end = time.time() + timeout
    while time.time() < end:
        r = httpx.post(DOCUMENT, json={"id": doc_id},
                       headers={"apikey": key}, timeout=60)
        r.raise_for_status()
        d = r.json()
        out = (d.get("output") or "").strip()
        if out:
            return out
        time.sleep(6)
    return None


def humanize_paragraph(par, key):
    """Returns (proposed, reason_rejected). Never raises on bad output."""
    flat = " ".join(par.split())            # the API cannot take line breaks
    prot, store = protect(flat)
    if len(prot) < MIN_LEN or len(prot) > MAX_LEN:
        return None, f"length {len(prot)}"
    doc = submit(prot, key)
    out = fetch(doc, key)
    if not out:
        return None, "timeout"
    if not all_sentinels_intact(out, store):
        return None, "sentinel lost"
    restored = restore(" ".join(out.split()), store)
    bad = degenerate(flat, restored)
    if bad:
        return None, bad
    return restored, None
