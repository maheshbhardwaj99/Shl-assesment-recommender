# Approach Document — SHL Assessment Recommender

## 1. Problem framing

The agent needs to go from a vague hiring intent to a grounded shortlist of
**Individual Test Solutions only**, over a *stateless* multi-turn API, inside
tight limits (8 turns, 30s/call). That constrains the design more than the
recommendation logic itself: every `/chat` call re-derives all state from the
message history, and the LLM can only be called ~once per turn to stay inside
the timeout with retrieval + validation on either side of it.

## 2. Data

**What I used.** SHL's live catalog listing page did not return a scrapable
item table when I built this (the URL resolved to a generic marketing page
rather than the filterable table), so I sourced a pre-scraped CSV
(name/url/duration/test_type/remote/adaptive) and re-derived the
**Individual Test Solutions** subset myself rather than trusting its scope
labeling.

**Filtering logic.** SHL's own type taxonomy (A/B/C/D/E/K/P/S) assigns one
code per construct. Composite products (Pre-packaged Job Solutions, "Short
Form" batteries) carry *multiple* codes because they bundle several
constructs. So: de-duplicate by (name, url) → keep only single-type-code rows
→ drop anything still literally named "... Solution" as a safety net. That
took 511 raw rows to 258 genuine individual tests (K:168, P:39, A:20, S:19,
D:5, C:4, B:3). `scripts/rebuild_catalog.py` documents/reproduces this so a
fresher scrape can be dropped in.

**Known gap.** The source CSV had no duration data (all `N/A`) — the
original scraper didn't extract it. I did not re-scrape 258 individual pages
given the time budget. The agent is instructed to say "not available in my
catalog data" rather than fabricate a duration, and this is the single
biggest thing I'd fix with more time (see §6).

## 3. Retrieval: TF-IDF, not embeddings

I used a TF-IDF vectorizer (word 1-2 grams) over `name + test_type_label`,
not a neural embedding index. Reasoning:
- Assessment names are short and keyword-dense ("Java 8 (New)", "SQL Server",
  "OPQ32r") — lexical overlap is already a strong signal for this catalog.
- No embedding-API round trip inside the 30s/call budget, no vector-DB
  dependency, no large model download slowing cold starts on free hosting.
- The LLM re-ranks/selects from the top-40 TF-IDF candidates each turn, so
  semantic understanding happens at *selection*, not at *retrieval*, which is
  cheaper and still grounded (see §4).

For compare queries ("difference between X and Y"), I additionally run a
fuzzy substring/ratio match (`difflib`) against acronym-like tokens in the
latest message (e.g. "OPQ") and merge those into a separate "compare
candidates" pool, since a raw name like "OPQ" won't score well against the
tokenized "opq32r" in TF-IDF alone.

## 4. Grounding / anti-hallucination

The model only ever sees two catalog slices per turn (retrieved candidates +
compare candidates) and is instructed to use their *exact* name/url. After
the call, every returned URL is checked against the full catalog
independently of the prompt — if a URL doesn't exist, it's dropped rather
than passed through. This is a hard backstop: even if the prompt fails or
the model drifts, the API contract ("every URL comes from the scraped
catalog") can't be violated by construction.

## 5. Agent behavior (single LLM call per turn)

One Gemini call per `/chat` request returns a small structured JSON object
(`action`, `reply`, `recommended_urls`, `conversation_complete`) via
`response_schema`, which is then mapped onto the assignment's exact response
shape:
- **clarify** — recommendations forced empty regardless of what the model
  returns, so an eager model can't short-circuit the "don't recommend on
  turn 1 for a vague query" behavior probe.
- **recommend / refine** — both go through the same validation path; refine
  isn't special-cased beyond "the model sees the full history," which is
  what lets accumulated constraints (e.g. "also add personality tests")
  compound instead of resetting.
- **compare** — recommendations forced empty; answer is grounded in the
  compare-candidates metadata only, with an explicit instruction not to use
  prior knowledge about named products.
- **refuse** — for out-of-scope (legal/general hiring advice) and
  prompt-injection attempts. Injection attempts are additionally caught by a
  regex pre-filter (common patterns like "ignore previous instructions",
  "reveal your system prompt") *before* the LLM call, so that failure mode
  doesn't depend on the model behaving — it's deterministic.

**Turn cap.** `end_of_conversation` is forced `true` whenever this response
would be the 8th turn, regardless of what the model outputs, with an extra
system-prompt hint on that turn telling the model to give its best-effort
shortlist rather than ask another question. It's also set true early if the
user's message reads as closing ("thanks", "that's all") after a shortlist
was already given.

**Failure fallback.** If the Gemini call itself fails (timeout/quota/network),
`run_agent` returns a schema-valid `clarify` response instead of a 500, so a
transient LLM outage degrades gracefully rather than failing the hard
schema-compliance eval outright.

## 6. Evaluation

I didn't have access to the official 10-trace zip (the link in the
assignment PDF wasn't resolvable from my environment), so I wrote 9 synthetic
scripted traces (`tests/traces.json`) covering: clarify-then-recommend,
immediate-detail recommend, refine growing a shortlist, compare, refuse
(legal advice / general hiring advice / prompt injection), "no preference"
handling, and a turn-cap stress test. `tests/run_eval.py` runs them against a
live server and checks schema compliance, zero-hallucination (every returned
url exists in the catalog), and the specific behavior each trace probes.

**What I'd do with more time:** (1) re-scrape per-item pages for real
duration data and add it as a hard filter, not just prompt context; (2) once
the real trace set is available, compute actual Recall@10 and tune the
TF-IDF top-k / the "enough context to recommend" threshold against it —
right now that threshold is prompt-specified, not empirically tuned; (3)
add a second-pass re-ranker (cross-encoder or a cheap Gemini call scoring
top-40 → top-10) if lexical-only retrieval turns out to miss semantically-
related-but-lexically-different matches (e.g. "leadership" query missing
"OPQ32r").

## 7. Stack

FastAPI + scikit-learn (TF-IDF) + Gemini 2.5 Flash via raw REST calls
(no SDK, to keep the free-tier deployment footprint small). No vector DB —
unnecessary at 258 items. AI-assisted: used an AI pair-programmer for
boilerplate (Pydantic models, FastAPI scaffolding) and for triaging which
public GitHub scrape to reuse as a starting dataset; the retrieval design,
grounding/validation logic, guardrails, and agent state-handling were
authored and reasoned through directly.
