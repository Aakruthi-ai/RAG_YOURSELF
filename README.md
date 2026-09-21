# 🧑‍💻 RAG-over-Myself — Agentic GraphRAG Career Assistant

> **Most portfolios are static pages you scroll. This one is a system you can interrogate.**



An agentic GraphRAG system built over my own resume data — skills, projects, courses, and certifications — that answers natural-language questions about my background, performs job-description gap analysis, and grows itself by ingesting new resume PDFs and GitHub repositories on the fly.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Deploy to Hugging Face Spaces](#deploy-to-hugging-face-spaces)
- [Project Structure](#project-structure)
- [Engineering Decisions](#engineering-decisions)
- [Evaluation](#evaluation)
- [Limitations & Roadmap](#limitations--roadmap)

---

## Why this exists

I didn't want a portfolio that just *lists* what I know. I wanted something that demonstrates it. So instead of a static "About Me" page, I built a retrieval-augmented system where the retrieval corpus is my actual work history, the graph is the relationships between it, and the interface is a conversation.

This project is as much a demonstration of engineering judgment as it is a career artifact — every design decision below exists because a simpler approach broke.

---

## Features

### 🔍 Ask about me
Natural-language Q&A over my background. Ask *"What NLP projects have you built?"* or *"Do you have cloud deployment experience?"* and get a grounded, first-person answer with retrieved context.

### 🎯 JD Match with session memory
Paste any job description and get a structured analysis:
- **Matches** — skills and projects that directly align, with reasons
- **Gaps** — requirements not currently covered, stated honestly
- **Reframing suggestions** — how to rephrase existing bullet points to echo the JD's language *without fabricating experience*
- **Compared to previous JDs** — the app remembers every JD pasted this session and flags recurring gaps (*"this is the third JD asking for Docker"*)

### 📄 Resume PDF ingestion
Upload a resume PDF; the LLM extracts structured skills, projects, courses, and certifications, merges them into the knowledge base, deduplicates against existing entries by name *and* fuzzy match, and rebuilds the retrieval indexes instantly. No manual JSON editing.

### 🐙 GitHub auto-sync
Paste a GitHub username; the system pulls public repos and their READMEs, summarizes each as a project entry with a skill list, and merges the results the same way.

### 📊 RAGAS-style evaluation scorecard
For each test question, an LLM judge scores:
- **Faithfulness** — is every claim in the answer grounded in the retrieved context? (1–5)
- **Relevancy** — does the answer actually address the question? (1–5)

Most student RAG projects skip evaluation entirely. This one doesn't.

### 🕸️ Interactive knowledge graph
A drag/zoom/hover `pyvis` visualization of every skill, project, course, and certification — and every relationship between them. Includes an orphan-node detector that flags disconnected data and auto-links what it can.

---

## Architecture

```
User query
    │
    ▼
┌───────────────────────────────────────────────────────┐
│                  Hybrid Retrieval                      │
│                                                        │
│   BM25 keyword search ──┐                              │
│                          ├──▶ min-max normalize ──▶    │
│   Semantic embeddings ───┘        + blend (0.4/0.6) ──▶│
│                                                        │
│   ──▶ 1-hop graph expansion (NetworkX) ──▶ Context     │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│              Generation (Groq API)                     │
│   openai/gpt-oss-120b with native JSON mode for        │
│   structured extraction tasks                          │
└───────────────────────────────────────────────────────┘
    │
    ▼
Answer + cited context
```

### Why hybrid retrieval?

Pure embedding search smooths over exact tokens — a tool name like `cuGraph`, an acronym like `TF-IDF`, a course code, a model version. BM25 catches those. Blending the two, with min-max normalization so neither dominates the score, gave a measurable jump in recall on acronym-heavy queries.

### Why graph expansion?

If you ask about *"NLP"* and I have an `NLP` skill node connected to three projects that used it, plain vector retrieval returns just the skill node. Expanding 1-hop into the graph pulls the projects, the courses that taught it, and any certifications that validated it — all for free, with no extra LLM calls.

### Why an LLM-as-judge scorecard?

Because "it looks like it works" is not an engineering claim. The scorecard gives a numerical answer to *"is this actually grounded?"* and *"is this actually relevant?"* — and it caught real regressions during development (e.g. answers that confidently said things not in the context).

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 90 MB, runs on CPU in milliseconds, no GPU cost |
| Keyword search | `rank_bm25` | Exact-token recall that embeddings miss |
| Graph | `networkx` | Flexible DiGraph, no DB needed for personal-scale corpus |
| Generation | Groq API (`openai/gpt-oss-120b`) | Sub-second latency, free tier, native JSON mode |
| PDF parsing | `pypdf` | Pure-Python, no system deps |
| UI | Gradio | Fast to build, deploys to HF Spaces in one click |
| Graph viz | `pyvis` | Interactive, embeddable, no JS framework required |
| Hosting | Hugging Face Spaces | Free tier is sufficient for a personal demo |



---


## Deploy to Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK.
2. Upload `app.py`, `profile_data.json`, and `requirements.txt`.
3. Add `GROQ_API_KEY` (and optionally `GITHUB_TOKEN`) under **Settings → Repository secrets**.
4. If your Space runs on **ZeroGPU hardware**, the `@spaces.GPU` stub at the top of `app.py` satisfies the startup probe. If it runs on `cpu-basic`, the stub is a harmless no-op.

---

## Project Structure

```
.
├── app.py                  # Everything: graph, retrieval, agents, UI
├── profile_data.json       # The knowledge base (edit this)
├── requirements.txt
└── knowledge_graph.html    # Generated on first run
```

The whole app lives in one file on purpose. It's a personal-scale system with a small enough surface area that splitting it into `retrieval.py`, `agents.py`, `ui.py` would add import ceremony without buying anything. If it ever grows past ~3,000 lines, that changes.

---

## Engineering Decisions

<details>
<summary><strong>1. Native JSON mode + truncation repair for LLM extraction</strong></summary>

LLMs will emit malformed JSON at the worst moment — often truncated mid-generation when they hit the token limit. Three layers of defense:

1. **Groq's `response_format={"type": "json_object"}`** constrains the decoder so only syntactically valid JSON is emitted.
2. **Multi-strategy extraction** — try the whole string, then the first `{...}` block, then strip code fences.
3. **Truncation repair** — a character-walker that tracks open string literals and bracket depth, then closes any dangling `"` and appends the missing `]`/`}` in the right order.

Without the third layer, a resume with more than ~20 skills would fail extraction entirely.
</details>

<details>
<summary><strong>2. Name-based deduplication, not just ID-based</strong></summary>

The first version of resume ingestion slugified every extracted entity's name into an ID and checked that ID for collisions. Problem: a hand-written skill with ID `skill_cloud` and name `"Cloud Deployment"` wouldn't be recognized as existing when the extractor later produced the same name with a slugified ID of `skill_cloud_deployment`.

Fix: deduplicate by **normalized name first** (case-insensitive, whitespace-collapsed), then fall back to fuzzy matching (`difflib.get_close_matches` with 0.92 cutoff) for near-misses. Now re-running extraction is idempotent.
</details>

<details>
<summary><strong>3. Compound-skill splitting as a safety net</strong></summary>

Extraction prompts tell the model to emit atomic skills. It mostly complies — but "mostly" isn't good enough when a compound like `"AI & Machine Learning"` becomes an orphaned node with no edges.

Every extracted skill name is split on `&`, `,`, `/`, and `and` before insertion. Genuinely atomic names pass through unchanged. Compound names get split, and each part is deduplicated independently.
</details>

<details>
<summary><strong>4. Graph orphan detection as data-quality feedback</strong></summary>

The graph view flags every node with zero edges. This started as a UI feature and turned into the most useful debugging tool in the project — disconnected nodes almost always meant a broken `skills_used` reference or a duplicate skill that should have been merged. Surfacing them in the UI made data hygiene visible instead of silent.
</details>

---

## Evaluation

The **Evaluation** tab runs an LLM-as-judge over a configurable set of test questions. Typical scores on the current profile:

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | 4.5–5.0 / 5 | Answers stay grounded in retrieved context |
| Relevancy | 4.0–4.7 / 5 | Occasionally over-explains before answering |

Observed failure modes:

- **Multi-hop questions** (*"which of my projects used both NLP and cloud deployment?"*) get partial answers because 1-hop graph expansion isn't deep enough.
- **Negation** (*"which skills do I *not* have?"*) works, but the judge sometimes penalizes the answer for correctly saying "no."
- **Very short queries** (*"Python?"*) retrieve well but the answer lacks the specificity the judge wants.

These are documented limitations, not hidden bugs.

---

## Limitations & Roadmap

**Known limitations**

- `jd_history` is in-memory; it resets on Space restart.
- Graph expansion is 1-hop only.
- At scale (10,000+ nodes), the in-memory cosine-similarity matrix would become the bottleneck.

**Planned improvements**

- [ ] Deeper graph expansion (2-hop) with an LLM re-ranker for multi-hop questions
- [ ] Persist JD history to SQLite
- [ ] Stream responses via server-sent events to reduce perceived latency
- [ ] Swap to a vector DB (`chromadb` / `qdrant`) before node count exceeds ~5,000
- [ ] Add a "compare against target role" mode that suggests which skills to acquire next

---

