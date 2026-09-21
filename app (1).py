# %% [markdown]
# # RAG-over-Myself: Agentic GraphRAG Career Assistant (v2)
#
# Builds a GraphRAG system over your own resume data (skills, projects,
# courses, certifications) with several agentic extras:
#
# 1. **Ask about me** — hybrid retrieval (BM25 keyword search + semantic
#    embeddings + graph expansion) answers questions about your background.
# 2. **JD Match** — paste a job description; get a matches/gaps/reframing
#    breakdown, and it remembers previous JDs you've pasted this session.
# 3. **Resume PDF ingestion** — upload a resume PDF; the LLM extracts
#    structured skills/projects/courses and merges them into your profile.
# 4. **GitHub auto-sync** — pulls your public repos + READMEs and turns them
#    into project nodes automatically.
# 5. **Evaluation scorecard** — a small RAGAS-style faithfulness/relevancy
#    eval using an LLM-as-judge over a handful of test questions.
# 6. **Knowledge graph** — interactive, drag/zoom/hover pyvis visualization.
#
# **Before running:** edit `profile_data.json` with your real information,
# and set `GROQ_API_KEY` as an environment variable (free key at
# https://console.groq.com/keys).

# %%
# ---- FIX #1: import `spaces` FIRST, before any CUDA-initializing library ----
# On Hugging Face Spaces, `spaces` MUST be imported before torch /
# sentence-transformers. Otherwise HF's background file-watcher thread raises:
#   "RuntimeError: CUDA has been initialized before importing the `spaces` package"
# Wrapped in try/except so this notebook still runs fine outside Spaces
# (e.g. locally, in Colab, or on a plain VM) where `spaces` isn't installed.
try:
    import spaces
    _SPACES_AVAILABLE = True
except ImportError:
    _SPACES_AVAILABLE = False

# ---- FIX #2: register a @spaces.GPU stub so ZeroGPU's startup probe passes ----
# ZeroGPU hardware (zero-a10g) requires at least one @spaces.GPU-decorated
# function to exist at module level, otherwise the container crashes at boot
# with "No @spaces.GPU function detected during startup".
# This app does no local GPU compute (sentence-transformers runs on CPU; all
# LLM calls go to the Groq API), so the stub below is never actually called.
# It exists purely to satisfy ZeroGPU's startup handshake. On cpu-basic
# hardware, @spaces.GPU is a no-op, so this block is harmless there too.
if _SPACES_AVAILABLE:
    @spaces.GPU(duration=1)
    def _zerogpu_probe():
        return "ready"

# Also silence torch's "Can't initialize NVML" noise on CPU-only Spaces.
import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# %%
# ---- Install dependencies (run once) ----
# !pip install gradio sentence-transformers networkx groq scikit-learn pyvis rank_bm25 pypdf pandas requests --quiet

# %%
import json
import re
import base64
import difflib
import textwrap
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx
import requests
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from pyvis.network import Network

# %%
# ---- Config ----
# Get a free Groq API key at https://console.groq.com/keys
# Set it as an environment variable before running:
#   export GROQ_API_KEY="your-key-here"
# or, just to test quickly, paste it directly below (do NOT commit a real
# key to GitHub — remove this line again before pushing your repo public):
# os.environ["GROQ_API_KEY"] = "your-key-here"

GROQ_MODEL = "openai/gpt-oss-120b"  # check console.groq.com/docs/models for the current lineup — Groq deprecates models periodically
DATA_PATH = "profile_data.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs locally/free

LLM_AVAILABLE = bool(os.environ.get("GROQ_API_KEY"))
if not LLM_AVAILABLE:
    print(
        "WARNING: GROQ_API_KEY not set. The app will still run, but will "
        "show raw retrieved context instead of a generated answer. Set the "
        "key to enable full LLM-generated responses."
    )
else:
    # Quick sanity check: confirm GROQ_MODEL is actually available before
    # the app tries to use it mid-answer. Groq deprecates models periodically,
    # so this catches a stale model name early with a clear message instead
    # of a 404 buried in a traceback later.
    try:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
        _available_ids = {m.id for m in _client.models.list().data}
        if GROQ_MODEL not in _available_ids:
            print(
                f"WARNING: GROQ_MODEL='{GROQ_MODEL}' was not found in your "
                f"account's available models. Groq may have deprecated it. "
                f"Available models include: {sorted(_available_ids)[:10]} ... "
                f"Update GROQ_MODEL in the config cell to one of these."
            )
    except Exception as e:
        print(f"Note: couldn't verify GROQ_MODEL availability at startup ({e}). Proceeding anyway.")

# %% [markdown]
# ## 1. Load profile data and build the knowledge graph
#
# `rebuild_all()` is the single function that reloads data, rebuilds the
# graph, and rebuilds both retrieval indexes (embeddings + BM25). It's called
# once at startup, and again any time resume ingestion or GitHub sync adds
# new nodes — so the app never needs a manual restart to pick up new data.

# %%
def load_profile_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_profile_data(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_graph(data: dict) -> nx.DiGraph:
    """Builds a directed graph: skills, courses, certifications, projects as
    nodes; edges represent 'uses' / 'covers' relationships between them."""
    G = nx.DiGraph()

    for skill in data.get("skills", []):
        G.add_node(skill["id"], type="skill", name=skill["name"], text=skill["description"])

    for course in data.get("courses", []):
        G.add_node(course["id"], type="course", name=course["name"], text=course["description"])
        for skill_id in course.get("skills_covered", []):
            if skill_id in G:
                G.add_edge(course["id"], skill_id, relation="covers")

    for cert in data.get("certifications", []):
        G.add_node(cert["id"], type="certification", name=cert["name"], text=cert["description"])
        for skill_id in cert.get("skills_related", []):
            if skill_id in G:
                G.add_edge(cert["id"], skill_id, relation="validates")

    for proj in data.get("projects", []):
        G.add_node(proj["id"], type="project", name=proj["name"], text=proj["description"])
        for skill_id in proj.get("skills_used", []):
            if skill_id in G:
                G.add_edge(proj["id"], skill_id, relation="uses")
        for course_id in proj.get("courses_used", []):
            if course_id in G:
                G.add_edge(proj["id"], course_id, relation="built_on")

    return G


def slugify(text: str, prefix: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return f"{prefix}_{s}"[:60]


def split_compound_skill_name(name: str) -> list:
    """Splits a category-style compound name (e.g. 'AI & Machine Learning',
    'Cloud & Deployment', 'Data, Programming & Tools') into atomic skill
    names. Genuinely atomic names with no separators pass through unchanged.
    This is a safety net — the extraction prompts are also instructed not to
    produce compounds in the first place, but LLMs occasionally slip."""
    parts = re.split(r"\s*(?:&|,|/|\band\b)\s*", name, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
    return parts if len(parts) > 1 else [name]


def _normalize_name(name: str) -> str:
    """Loose normalization for de-duplication: case/whitespace-insensitive,
    strips common punctuation, so 'Cloud Deployment' and 'Cloud & Deployment'
    are still treated as different (that's a real distinct label a person
    might mean differently) but 'Cloud Deployment' and 'cloud   deployment'
    are recognized as the same thing."""
    return re.sub(r"\s+", " ", name.strip().lower())


def find_existing_id_by_name(data: dict, category: str, name: str):
    """Looks for an existing entry in the given category whose name matches
    (case/whitespace-insensitive). Returns its id if found, else None. This
    catches duplicates that slugified-id matching alone misses — e.g. a
    hand-written skill with id 'skill_cloud' and name 'Cloud Deployment'
    wouldn't be recognized as existing by ID matching alone if a later
    extraction also produces the name 'Cloud Deployment' but generates the
    id 'skill_cloud_deployment' via slugify."""
    target = _normalize_name(name)
    for entry in data.get(category, []):
        if _normalize_name(entry["name"]) == target:
            return entry["id"]
    return None


# ---- Globals populated by rebuild_all() ----
profile_data = {}
graph = None
embedder = None
node_ids = []
node_texts = []
node_embeddings = None
bm25_index = None
bm25_tokenized_corpus = []


def rebuild_all():
    """Reloads profile_data.json, rebuilds the graph, and rebuilds both the
    embedding index and the BM25 index. Call this after any change to the
    underlying data (resume ingestion, GitHub sync, manual edits)."""
    global profile_data, graph, node_ids, node_texts, node_embeddings
    global bm25_index, bm25_tokenized_corpus, embedder

    profile_data = load_profile_data(DATA_PATH)
    graph = build_graph(profile_data)

    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL_NAME)

    node_ids = list(graph.nodes())
    node_texts = [f"{graph.nodes[n]['name']}. {graph.nodes[n]['text']}" for n in node_ids]
    node_embeddings = embedder.encode(node_texts, normalize_embeddings=True) if node_texts else np.zeros((0, 384))

    bm25_tokenized_corpus = [t.lower().split() for t in node_texts]
    bm25_index = BM25Okapi(bm25_tokenized_corpus) if bm25_tokenized_corpus else None

    print(f"Rebuilt: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
          f"{len(node_ids)} indexed (embeddings + BM25).")


rebuild_all()

# %% [markdown]
# ## 2. Hybrid retrieval: BM25 keyword search + semantic embeddings + graph expansion
#
# Pure embedding search can miss exact terms (a tool name, a course code,
# an acronym) because semantic similarity smooths over exact wording. BM25
# catches those. We min-max normalize both score arrays and blend them, then
# expand 1-hop into the graph so related nodes (e.g. a project's skills)
# come along for context.

# %%
def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def retrieve(query: str, top_k: int = 6, expand_neighbors: bool = True, bm25_weight: float = 0.4):
    """Hybrid retrieval: blends BM25 keyword scores with semantic embedding
    similarity, then does 1-hop graph expansion for richer context."""
    if not node_ids:
        return []

    query_emb = embedder.encode([query], normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_emb, node_embeddings)[0]

    bm25_scores = np.array(bm25_index.get_scores(query.lower().split())) if bm25_index else np.zeros(len(node_ids))

    combined = (bm25_weight * _minmax_norm(bm25_scores)) + ((1 - bm25_weight) * _minmax_norm(semantic_scores))
    top_idx = np.argsort(combined)[::-1][:top_k]

    retrieved_ids = {node_ids[i] for i in top_idx}

    if expand_neighbors:
        expanded = set(retrieved_ids)
        for nid in list(retrieved_ids):
            expanded.update(graph.successors(nid))
            expanded.update(graph.predecessors(nid))
        retrieved_ids = expanded

    results = []
    for nid in retrieved_ids:
        node = graph.nodes[nid]
        results.append(f"[{node['type'].upper()}] {node['name']}: {node['text']}")
    return results


print("\n".join(retrieve("What NLP projects have you built?", top_k=3)))

# %% [markdown]
# ## 3. LLM wrapper (Groq API)

# %%
def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1000, json_mode: bool = False) -> str:
    """Calls the Groq API. Falls back to returning the raw prompt context if
    no API key is set, so the notebook still runs end-to-end without a key.

    `json_mode=True` enables Groq's native JSON output mode, which forces the
    model to emit a syntactically valid JSON object — this dramatically
    reduces the "model returned prose / broken JSON" failure rate."""
    if not LLM_AVAILABLE:
        return (
            "[No GROQ_API_KEY set — showing retrieved context instead of a "
            "generated answer]\n\n" + user_prompt
        )

    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    kwargs = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        # Some models don't support response_format — retry without it once.
        if json_mode:
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    return response.choices[0].message.content


def _repair_truncated_json(text: str) -> str:
    """Best-effort repair for a JSON object that was cut off mid-generation
    (the model hit its max_tokens limit). Walks the string tracking whether
    we're inside a string literal and which brackets are open, then closes
    the string and appends the missing `]`/`}` in the correct order."""
    in_string = False
    escape = False
    stack = []
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if in_string:
        text += '"'
    # A dangling trailing comma would make the closed JSON invalid.
    text = re.sub(r",\s*$", "", text.rstrip())
    for opener in reversed(stack):
        text += "}" if opener == "{" else "]"
    return text


def _extract_json_object(text: str):
    """Tries several increasingly permissive strategies to pull a JSON object
    out of an LLM response that may include stray prose, code fences, or
    text before/after the JSON — and, as a last resort, repairs a JSON object
    truncated by a max_tokens cutoff."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    # Strategy 1: the whole cleaned string is valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: grab the first {...} block (handles stray prose around the JSON)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 3: repair a truncated JSON object (model hit max_tokens mid-object)
    start = cleaned.find("{")
    if start != -1:
        candidate = cleaned[start:]
        for attempt in (candidate, _repair_truncated_json(candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    return None


def call_llm_json(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> dict:
    """Calls the LLM and parses the response as JSON. Uses Groq's native JSON
    mode for reliability, tries a couple of extraction strategies (see
    _extract_json_object), and on failure retries once with an extra
    instruction emphasizing raw-JSON-only output before giving up."""
    strict_system_prompt = system_prompt + "\n\nIMPORTANT: Output ONLY the raw JSON object. No prose before or after it, no markdown code fences."

    raw = call_llm(strict_system_prompt, user_prompt, max_tokens=max_tokens, json_mode=True)
    parsed = _extract_json_object(raw)
    if parsed is not None:
        return parsed

    # one retry, with even more explicit instruction and the same generous budget
    retry_prompt = user_prompt + "\n\n(Your previous response wasn't valid JSON. Respond with ONLY the JSON object this time.)"
    raw_retry = call_llm(strict_system_prompt, retry_prompt, max_tokens=max_tokens, json_mode=True)
    parsed_retry = _extract_json_object(raw_retry)
    if parsed_retry is not None:
        return parsed_retry

    return {"_parse_error": True, "_raw": raw_retry}

# %% [markdown]
# ## 4. Agent tool 1 — Ask about me

# %%
ASK_SYSTEM_PROMPT = (
    "You are a career assistant that answers questions about a student's "
    "background (skills, projects, courses, certifications) using ONLY the "
    "context provided below. Be specific, concise, and honest — if the "
    "context doesn't contain the answer, say so rather than guessing. "
    "Speak in first person, as if you were the student answering directly."
)


def retrieve_and_answer(query: str, top_k: int = 6):
    """Returns (answer, context_string) — used by both the chat tab and the
    evaluation tab, since eval needs to inspect the context an answer used."""
    context_chunks = retrieve(query, top_k=top_k)
    context = "\n".join(context_chunks)
    user_prompt = f"Context about me:\n{context}\n\nQuestion: {query}"
    answer = call_llm(ASK_SYSTEM_PROMPT, user_prompt)
    return answer, context


def ask_about_me(query: str) -> str:
    answer, _ = retrieve_and_answer(query)
    return answer


print(ask_about_me("What experience do you have with deep learning?"))

# %% [markdown]
# ## 5. Agent tool 2 — JD match with session memory
#
# `jd_history` persists for the life of the running app (i.e. across
# messages in one Gradio session) so the agent can compare each new JD
# against what it's seen before. Note: this is a single shared list, which
# is fine for a personal portfolio demo running as one instance; if you ever
# deploy this for multiple simultaneous users, swap it for `gr.State` so
# each user gets their own history instead of sharing one.

# %%
jd_history = []  # list of {"jd_snippet": str, "gaps_summary": str}

JD_SYSTEM_PROMPT = (
    "You are a career coach helping a student match their background against "
    "a job description. You will be given (1) a job description, (2) "
    "retrieved context about the student's actual skills/projects/courses, "
    "and (3) an optional summary of gaps identified in previous job "
    "descriptions this session. Produce a structured response with three "
    "sections:\n"
    "1. MATCHES — skills/projects that directly align with the JD, with brief reasons.\n"
    "2. GAPS — requirements in the JD not currently covered, stated honestly.\n"
    "3. REFRAMING SUGGESTIONS — concrete phrasing tweaks to existing resume "
    "bullet points so they better reflect the JD's language, without "
    "fabricating experience the student doesn't have.\n"
    "If previous JD gaps are provided, add a short 4th section, COMPARED TO "
    "PREVIOUS JDs, noting any recurring gap (e.g. 'this is the second JD in a "
    "row asking for cloud deployment experience'). Keep it concrete and "
    "actionable, not generic advice."
)


def jd_match(jd_text: str) -> str:
    context_chunks = retrieve(jd_text, top_k=8)
    context = "\n".join(context_chunks)

    history_context = ""
    if jd_history:
        recent = jd_history[-3:]
        history_context = "\n\nPrevious JD gap summaries this session:\n" + "\n".join(
            f"- ({h['jd_snippet']}): {h['gaps_summary']}" for h in recent
        )

    user_prompt = (
        f"Job description:\n{jd_text}\n\nMy background (retrieved context):\n{context}"
        f"{history_context}"
    )
    result = call_llm(JD_SYSTEM_PROMPT, user_prompt, max_tokens=1800)

    # crude extraction of the GAPS section to seed memory for next time
    gaps_match = re.search(r"GAPS(.*?)(REFRAMING|COMPARED|$)", result, re.DOTALL | re.IGNORECASE)
    gaps_summary = gaps_match.group(1).strip()[:300] if gaps_match else "(no gaps parsed)"
    jd_history.append({"jd_snippet": jd_text[:60].replace("\n", " ") + "...", "gaps_summary": gaps_summary})

    return result


sample_jd = "Looking for a Data Science intern with NLP and deep learning experience, Python, and cloud deployment skills."
print(jd_match(sample_jd))

# %% [markdown]
# ## 6. Agent tool 3 — Resume PDF ingestion
#
# Upload a resume PDF; the LLM extracts skills/projects/courses/certifications
# as structured JSON, which gets merged into `profile_data.json` (new entries
# only — it won't duplicate anything whose slugified id already exists), then
# `rebuild_all()` picks up the new data immediately.

# %%
RESUME_EXTRACT_SYSTEM_PROMPT = (
    "You extract structured career data from resume text. Respond ONLY with "
    "valid JSON (no markdown fences, no commentary) matching this schema:\n"
    "{\n"
    '  "skills": [{"name": "...", "description": "..."}],\n'
    '  "projects": [{"name": "...", "description": "...", "skills_used": ["..."]}],\n'
    '  "courses": [{"name": "...", "description": "..."}],\n'
    '  "certifications": [{"name": "...", "description": "...", "skills_related": ["..."]}]\n'
    "}\n"
    "IMPORTANT — list atomic, individual skills only. Never group multiple "
    "skills into a single entry using '&', 'and', or commas. For example, "
    "output 'Python' and 'NLP' as two separate skill entries — do NOT output "
    "a single entry like 'AI & Machine Learning' or 'Data & Programming' "
    "covering several tools at once. Prefer specific tool/technique names "
    "(e.g. 'Docker', 'Random Forest') over vague category headers (e.g. "
    "'Cloud Skills', 'ML Skills'). For certifications, skills_related should "
    "list the (atomic) skill names the certification demonstrates. Only "
    "include information actually present in the resume text — do not invent "
    "skills, projects, or dates that aren't there."
)


def extract_pdf_text(pdf_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _ensure_skill(data: dict, existing_ids: set, skill_name: str, description: str) -> list:
    """Given a possibly-compound skill name from LLM extraction (e.g. 'AI &
    Machine Learning'), splits it into atomic parts first, then for each
    part: reuses an existing skill if one with a matching name (exact
    case/whitespace-insensitive, or a close fuzzy match) already exists,
    otherwise creates a new one. Returns a list of ids (usually one, more if
    the input was a compound name) — this is what prevents both exact
    re-extractions AND category-header groupings from becoming duplicate or
    orphaned nodes."""
    result_ids = []
    for part in split_compound_skill_name(_normalize_name_preserve_case(skill_name)):
        existing = find_existing_id_by_name(data, "skills", part)
        if not existing:
            existing_names = [s["name"] for s in data.get("skills", [])]
            close = difflib.get_close_matches(part, existing_names, n=1, cutoff=0.92)
            if close:
                existing = find_existing_id_by_name(data, "skills", close[0])

        if existing:
            if existing not in result_ids:
                result_ids.append(existing)
            continue

        sid = slugify(part, "skill")
        if sid in existing_ids:
            if sid not in result_ids:
                result_ids.append(sid)
            continue

        data["skills"].append({"id": sid, "name": part, "description": description})
        existing_ids.add(sid)
        result_ids.append(sid)

    return result_ids


def _normalize_name_preserve_case(name: str) -> str:
    """Collapses whitespace but keeps original casing — used before splitting
    compound names, since we want to preserve the real display casing of each
    resulting atomic skill (unlike _normalize_name, which lowercases for
    comparison purposes only)."""
    return re.sub(r"\s+", " ", name).strip()


def _fuzzy_possible_duplicate(data: dict, category: str, name: str):
    """For projects specifically: catches the common case where a GitHub
    repo name ('TruthMate') and a resume title for the same project
    ('TruthMate: AI Fake Content Detector') would otherwise be added as two
    separate nodes, since neither exact-name nor id matching would flag
    them as the same thing. Returns the matched existing name if one string
    contains the other (case-insensitive), else None. This only WARNS — it
    doesn't auto-skip, since two genuinely different projects could share a
    short common prefix, and silently merging them could lose real data."""
    if category != "projects":
        return None
    target = _normalize_name(name)
    if len(target) < 4:
        return None
    for entry in data.get(category, []):
        existing = _normalize_name(entry["name"])
        if len(existing) >= 4 and (existing in target or target in existing) and existing != target:
            return entry["name"]
    return None


def merge_extracted_data(extracted: dict) -> str:
    """Merges LLM-extracted entries into profile_data.json. De-duplicates by
    NAME first (case/whitespace-insensitive) — not just by generated id — so
    re-running resume ingestion or GitHub sync doesn't keep creating near-
    duplicate nodes for the same skill/project under slightly different
    generated ids. Returns a short human-readable summary of what was added,
    and flags anything skipped as a likely duplicate so you can sanity-check."""
    data = load_profile_data(DATA_PATH)
    added = {"skills": [], "projects": [], "courses": [], "certifications": []}
    skipped_as_duplicate = []
    possible_duplicates = []

    existing_ids = {
        e["id"] for category in ("skills", "projects", "courses", "certifications")
        for e in data.get(category, [])
    }

    for category, prefix in [("skills", "skill"), ("courses", "course"),
                              ("certifications", "cert"), ("projects", "proj")]:
        for item in extracted.get(category, []):
            if find_existing_id_by_name(data, category, item["name"]):
                skipped_as_duplicate.append(item["name"])
                continue

            new_id = slugify(item["name"], prefix)
            if new_id in existing_ids:
                skipped_as_duplicate.append(item["name"])
                continue

            fuzzy_match = _fuzzy_possible_duplicate(data, category, item["name"])
            if fuzzy_match:
                possible_duplicates.append(f"'{item['name']}' (looks similar to existing '{fuzzy_match}')")

            entry = {"id": new_id, "name": item["name"], "description": item.get("description", "")}
            if category == "projects":
                skill_id_lists = [
                    _ensure_skill(data, existing_ids, skill_name, f"Skill used in project: {item['name']}.")
                    for skill_name in item.get("skills_used", [])
                ]
                entry["skills_used"] = list(dict.fromkeys(sid for group in skill_id_lists for sid in group))
                entry["courses_used"] = []
            elif category == "certifications":
                skill_id_lists = [
                    _ensure_skill(data, existing_ids, skill_name, f"Skill validated by certification: {item['name']}.")
                    for skill_name in item.get("skills_related", [])
                ]
                entry["skills_related"] = list(dict.fromkeys(sid for group in skill_id_lists for sid in group))
            data[category].append(entry)
            existing_ids.add(new_id)
            added[category].append(item["name"])

    save_profile_data(data, DATA_PATH)
    rebuild_all()

    summary_lines = [f"- {cat.capitalize()}: {', '.join(names)}" for cat, names in added.items() if names]
    result = "**Added to your profile:**\n" + ("\n".join(summary_lines) if summary_lines else "Nothing new found.")
    if skipped_as_duplicate:
        result += (
            f"\n\n_Skipped as likely duplicates (name already exists): "
            f"{', '.join(skipped_as_duplicate)}_"
        )
    if possible_duplicates:
        result += (
            f"\n\n⚠️ _Possible duplicates added anyway (review manually — one of "
            f"these might be the same project under a different name): "
            f"{'; '.join(possible_duplicates)}_"
        )
    return result


def ingest_resume_pdf(pdf_file) -> str:
    if pdf_file is None:
        return "Please upload a PDF file first."
    resume_text = extract_pdf_text(pdf_file.name if hasattr(pdf_file, "name") else pdf_file)
    if not resume_text.strip():
        return "Couldn't extract any text from that PDF — it may be a scanned image without a text layer."

    extracted = call_llm_json(RESUME_EXTRACT_SYSTEM_PROMPT, resume_text[:6000], max_tokens=8000)
    if extracted.get("_parse_error"):
        return f"The model's output wasn't valid JSON. Raw response:\n\n{extracted['_raw'][:500]}"

    return merge_extracted_data(extracted)

# %% [markdown]
# ## 7. Agent tool 4 — GitHub auto-sync
#
# Pulls your public repos and their READMEs from the GitHub API, asks the LLM
# to summarize each as a project + skill list, and merges the results the
# same way resume ingestion does.

# %%
def fetch_github_repos(username: str, max_repos: int = 5, token: Optional[str] = None) -> list:
    headers = {"Accept": "application/vnd.github+json"}
    token = token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(
        f"https://api.github.com/users/{username}/repos",
        params={"sort": "updated", "per_page": max_repos},
        headers=headers, timeout=15,
    )
    resp.raise_for_status()

    results = []
    for repo in resp.json():
        readme_text = ""
        readme_resp = requests.get(
            f"https://api.github.com/repos/{username}/{repo['name']}/readme",
            headers=headers, timeout=15,
        )
        if readme_resp.status_code == 200:
            content = readme_resp.json().get("content", "")
            try:
                readme_text = base64.b64decode(content).decode("utf-8", errors="ignore")
            except Exception:
                readme_text = ""
        results.append({
            "name": repo["name"],
            "description": repo.get("description") or "",
            "readme": readme_text[:3000],
            "language": repo.get("language") or "",
        })
    return results


GITHUB_SUMMARIZE_SYSTEM_PROMPT = (
    "You summarize a GitHub repo into a resume-style project entry. Respond "
    "ONLY with valid JSON (no markdown fences): "
    '{"description": "one to two sentence summary of what the project does '
    'and what it demonstrates", "skills": ["skill1", "skill2", ...]}. '
    "List atomic, individual skills only — never group multiple skills into "
    "a single entry using '&', 'and', or commas (e.g. output 'Python' and "
    "'Docker' as two separate list items, not 'Python & Docker' as one). "
    "Base the skills list only on what's evidenced in the repo info given."
)


def sync_github_projects(username: str, max_repos: int = 5) -> str:
    if not username.strip():
        return "Please enter a GitHub username."

    try:
        repos = fetch_github_repos(username.strip(), max_repos=max_repos)
    except requests.exceptions.RequestException as e:
        return f"Couldn't reach GitHub: {e}"

    if not repos:
        return f"No public repos found for '{username}'."

    data = load_profile_data(DATA_PATH)
    existing_ids = {
        e["id"] for category in ("skills", "projects", "courses", "certifications")
        for e in data.get(category, [])
    }

    added_projects = []
    skipped_as_duplicate = []
    possible_duplicates = []

    for repo in repos:
        if find_existing_id_by_name(data, "projects", repo["name"]):
            skipped_as_duplicate.append(repo["name"])
            continue

        proj_id = slugify(repo["name"], "proj_gh")
        if proj_id in existing_ids:
            skipped_as_duplicate.append(repo["name"])
            continue

        fuzzy_match = _fuzzy_possible_duplicate(data, "projects", repo["name"])
        if fuzzy_match:
            possible_duplicates.append(f"'{repo['name']}' (looks similar to existing '{fuzzy_match}')")

        repo_info = f"Repo: {repo['name']}\nDescription: {repo['description']}\nLanguage: {repo['language']}\nREADME excerpt: {repo['readme'][:1500]}"
        summary = call_llm_json(GITHUB_SUMMARIZE_SYSTEM_PROMPT, repo_info)
        if summary.get("_parse_error"):
            continue

        skill_id_lists = [
            _ensure_skill(data, existing_ids, skill_name, f"Skill used in GitHub project: {repo['name']}.")
            for skill_name in summary.get("skills", [])
        ]
        skill_ids = list(dict.fromkeys(sid for group in skill_id_lists for sid in group))

        data["projects"].append({
            "id": proj_id, "name": repo["name"],
            "description": summary.get("description", repo["description"]),
            "skills_used": skill_ids, "courses_used": [],
        })
        existing_ids.add(proj_id)
        added_projects.append(repo["name"])

    save_profile_data(data, DATA_PATH)
    rebuild_all()

    if not added_projects:
        msg = "No new repos to add (all matched existing project names)."
        if skipped_as_duplicate:
            msg += f" Skipped: {', '.join(skipped_as_duplicate)}."
        return msg

    result = "**Synced from GitHub:**\n" + "\n".join(f"- {name}" for name in added_projects)
    if skipped_as_duplicate:
        result += f"\n\n_Skipped as likely duplicates: {', '.join(skipped_as_duplicate)}_"
    if possible_duplicates:
        result += (
            f"\n\n⚠️ _Possible duplicates added anyway (review manually): "
            f"{'; '.join(possible_duplicates)}_"
        )
    return result

# %% [markdown]
# ## 8. Evaluation tab — RAGAS-style scorecard (LLM-as-judge)
#
# A lightweight, dependency-free alternative to the full `ragas` library:
# for each test question, we retrieve context, generate an answer, then ask
# the LLM to score **faithfulness** (is the answer grounded in the retrieved
# context, not hallucinated?) and **relevancy** (does it actually address the
# question?) on a 1–5 scale. This is the kind of evaluation almost no
# student RAG project includes — worth highlighting in your README/LinkedIn post.

# %%
DEFAULT_TEST_QUESTIONS = [
    "What programming languages do you know?",
    "Tell me about a project involving retrieval or search.",
    "What courses have you taken related to machine learning?",
    "Do you have any cloud deployment experience?",
]

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator for a RAG system. Given a question, the retrieved "
    "context used to answer, and the generated answer, score two things on a "
    "1-5 scale:\n"
    "1. faithfulness: is every claim in the answer supported by the context? "
    "5 = fully grounded, 1 = largely unsupported or hallucinated.\n"
    "2. relevancy: does the answer directly address the question asked? "
    "5 = fully relevant, 1 = off-topic.\n"
    'Respond ONLY with JSON: {"faithfulness": <int 1-5>, "relevancy": <int 1-5>, "notes": "<one short sentence>"}'
)


def judge_answer(question: str, context: str, answer: str) -> dict:
    user_prompt = f"Question: {question}\n\nRetrieved context:\n{context}\n\nGenerated answer:\n{answer}"
    result = call_llm_json(JUDGE_SYSTEM_PROMPT, user_prompt, max_tokens=300)
    if result.get("_parse_error"):
        return {"faithfulness": None, "relevancy": None, "notes": "judge output unparseable"}
    return result


def run_evaluation(test_questions: Optional[list] = None):
    """Returns (pandas.DataFrame, summary_markdown) for display in Gradio."""
    questions = test_questions or DEFAULT_TEST_QUESTIONS
    rows = []
    for q in questions:
        answer, context = retrieve_and_answer(q)
        scores = judge_answer(q, context, answer)
        rows.append({
            "question": q,
            "answer": textwrap.shorten(answer, width=140, placeholder="..."),
            "faithfulness": scores.get("faithfulness"),
            "relevancy": scores.get("relevancy"),
            "notes": scores.get("notes", ""),
        })

    df = pd.DataFrame(rows)
    valid_faith = df["faithfulness"].dropna()
    valid_rel = df["relevancy"].dropna()
    avg_faith = valid_faith.mean() if len(valid_faith) else float("nan")
    avg_rel = valid_rel.mean() if len(valid_rel) else float("nan")

    summary = (
        f"**Average faithfulness: {avg_faith:.1f}/5**  |  **Average relevancy: {avg_rel:.1f}/5**\n\n"
        f"_Scored via LLM-as-judge over {len(questions)} test questions._"
    )
    return df, summary


def run_evaluation_gradio(questions_text: str):
    questions = [q.strip() for q in questions_text.split("\n") if q.strip()] or DEFAULT_TEST_QUESTIONS
    df, summary = run_evaluation(questions)
    return summary, df

# %% [markdown]
# ## 9. Knowledge graph visualization (interactive, pyvis)

# %%
NODE_COLORS = {
    "skill": "#4C72B0",
    "project": "#DD8452",
    "course": "#55A868",
    "certification": "#C44E52",
}

LAYER_X = {
    "skill": -300,
    "project": 0,
    "course": 300,
    "certification": 300,
}


def build_pyvis_html() -> str:
    net = Network(height="650px", width="100%", directed=True, bgcolor="#ffffff", notebook=False)
    net.toggle_physics(True)
    net.repulsion(node_distance=180, spring_length=200, damping=0.9)

    type_counters = {t: 0 for t in NODE_COLORS}
    for node_id, data in graph.nodes(data=True):
        node_type = data["type"]
        y = type_counters[node_type] * 90
        type_counters[node_type] += 1
        net.add_node(
            node_id, label=data["name"], title=f"[{node_type.upper()}] {data['text']}",
            color=NODE_COLORS[node_type], x=LAYER_X[node_type], y=y, physics=True,
        )

    for source, target, edge_data in graph.edges(data=True):
        net.add_edge(source, target, title=edge_data.get("relation", ""), color="#999999")

    return net.generate_html(notebook=False)


def draw_graph() -> str:
    """Encodes the pyvis HTML as a base64 data URI inside the iframe `src`
    instead of stuffing it into `srcdoc`. This keeps `<script>` tags out of
    the raw gr.HTML content (silencing Gradio's warning) while still
    executing them inside the iframe's own document. Full interactivity
    (drag, zoom, hover) is preserved."""
    html_content = build_pyvis_html()
    with open("knowledge_graph.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    b64 = base64.b64encode(html_content.encode("utf-8")).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{b64}" '
        f'style="width:100%;height:670px;border:none;"></iframe>'
    )


def graph_stats() -> str:
    counts = {}
    for _, d in graph.nodes(data=True):
        counts[d["type"]] = counts.get(d["type"], 0) + 1
    lines = [f"- {k.capitalize()}s: {v}" for k, v in counts.items()]

    orphans = find_orphan_nodes()
    summary = "**Profile summary**\n" + "\n".join(lines)
    if orphans:
        orphan_names = [graph.nodes[n]["name"] for n in orphans]
        summary += (
            f"\n\n⚠️ **{len(orphans)} disconnected node(s)** (not linked to any "
            f"project/course): {', '.join(orphan_names)}. Link these in "
            f"`profile_data.json` (via `skills_used`/`courses_used`/`skills_covered`) "
            f"so the graph reads as fully connected."
        )
    return summary


def find_orphan_nodes() -> list:
    """Returns node ids with no edges at all (neither incoming nor outgoing) —
    these are skills/certs/courses that exist in profile_data.json but aren't
    referenced by any project or course, so they show up as disconnected dots
    in the graph visualization."""
    return [n for n in graph.nodes() if graph.degree(n) == 0]


SKILL_LINK_SUGGEST_SYSTEM_PROMPT = (
    "Given a course or certification's name and description, and a list of "
    "the person's already-declared skills, return ONLY the skill names from "
    "that list that this course/certification genuinely relates to or "
    "teaches. Respond with JSON: {\"related_skills\": [\"...\"]}. If none "
    "clearly relate, return an empty list. Do not invent skills that are not "
    "in the provided list — only choose from what's given."
)


def suggest_skill_links_for_node(node_name: str, node_description: str, existing_skill_names: list) -> list:
    """Asks the LLM which of the person's ALREADY-DECLARED skills a
    disconnected course/certification relates to, based on its own name and
    description. This only re-derives structure (graph edges) from content
    the person already wrote — it cannot invent a new skill, since the model
    is constrained to picking from the existing skill list."""
    if not LLM_AVAILABLE or not existing_skill_names:
        return []
    user_prompt = (
        f"Course/Certification: {node_name}\nDescription: {node_description}\n\n"
        f"Available skills to choose from: {', '.join(existing_skill_names)}"
    )
    result = call_llm_json(SKILL_LINK_SUGGEST_SYSTEM_PROMPT, user_prompt, max_tokens=300)
    if result.get("_parse_error"):
        return []
    return [s for s in result.get("related_skills", []) if s in existing_skill_names]


def cleanup_profile_data() -> str:
    """One-time migration for profile_data.json entries created before the
    atomic-skill-splitting fix existed (e.g. from GitHub sync / resume
    ingestion runs before this update). Does two passes:
    1. SPLIT & MERGE — any existing skill whose name is a compound (e.g.
       'AI & Machine Learning') gets split into atomic parts. Each part is
       matched against other existing skills by name (exact or close fuzzy
       match) and reused if found, or kept as a new atomic skill otherwise.
       Every project/course/certification that referenced the old compound
       skill's id gets rewired to point at the new split ids instead. The
       old compound entry is then removed.
    2. LINK REMAINING ORPHANS — for any course or certification that's still
       disconnected afterward (no skills_covered/skills_related), asks the
       LLM to pick which of the person's ALREADY-DECLARED skills it relates
       to, based on its own description. This never invents a new skill —
       only rewires existing ones — so nothing is fabricated.
    Returns a human-readable summary of everything changed. Safe to run
    multiple times — it's a no-op on data that's already clean.
    """
    data = load_profile_data(DATA_PATH)
    changes = []

    # ---- Pass 1: split & merge compound skill names ----
    old_skills = data["skills"]
    new_skills = []
    id_remap = {}  # old_skill_id -> [new_skill_id, ...]

    # process non-compound skills first so they're available for reuse-matching
    compound_skills = [s for s in old_skills if len(split_compound_skill_name(s["name"])) > 1]
    atomic_skills = [s for s in old_skills if s not in compound_skills]
    new_skills.extend(atomic_skills)
    data["skills"] = new_skills  # so _ensure_skill's lookups see the atomic ones already

    existing_ids = {
        e["id"] for category in ("skills", "projects", "courses", "certifications")
        for e in data.get(category, [])
    }

    for old_skill in compound_skills:
        new_ids = _ensure_skill(data, existing_ids, old_skill["name"], old_skill["description"])
        id_remap[old_skill["id"]] = new_ids
        changes.append(f"Split '{old_skill['name']}' → {', '.join(new_ids)}")

    # rewire every reference to a remapped old skill id
    def _rewire(id_list):
        result = []
        for sid in id_list:
            result.extend(id_remap.get(sid, [sid]))
        return list(dict.fromkeys(result))

    for proj in data.get("projects", []):
        proj["skills_used"] = _rewire(proj.get("skills_used", []))
    for course in data.get("courses", []):
        course["skills_covered"] = _rewire(course.get("skills_covered", []))
    for cert in data.get("certifications", []):
        cert["skills_related"] = _rewire(cert.get("skills_related", []))

    save_profile_data(data, DATA_PATH)
    rebuild_all()

    # ---- Pass 2: LLM-assisted linking of remaining orphan courses/certs ----
    orphans = find_orphan_nodes()
    existing_skill_names = [s["name"] for s in data["skills"]]
    skill_name_to_id = {s["name"]: s["id"] for s in data["skills"]}

    for node_id in orphans:
        node = graph.nodes[node_id]
        if node["type"] not in ("course", "certification"):
            continue  # don't guess-link orphan skills/projects — only courses/certs have a field for this

        suggested_names = suggest_skill_links_for_node(node["name"], node["text"], existing_skill_names)
        if not suggested_names:
            continue

        suggested_ids = [skill_name_to_id[n] for n in suggested_names]
        field = "skills_covered" if node["type"] == "course" else "skills_related"
        category = "courses" if node["type"] == "course" else "certifications"
        for entry in data[category]:
            if entry["id"] == node_id:
                entry[field] = suggested_ids
                changes.append(f"Linked '{node['name']}' → {', '.join(suggested_names)}")
                break

    save_profile_data(data, DATA_PATH)
    rebuild_all()

    remaining_orphans = find_orphan_nodes()
    summary = "**Cleanup complete.**\n\n"
    summary += ("\n".join(f"- {c}" for c in changes) if changes else "Nothing needed changing.")
    if remaining_orphans:
        names = [graph.nodes[n]["name"] for n in remaining_orphans]
        summary += (
            f"\n\n⚠️ Still disconnected (couldn't confidently auto-link — "
            f"link these manually in `profile_data.json`): {', '.join(names)}"
        )
    else:
        summary += "\n\n✅ No disconnected nodes remain."
    return summary


print(graph_stats())

# %% [markdown]
# ## 9b. Orphan resolver (called from the UI)

# %%
# Skills that are vague category headers rather than real technologies.
# These duplicate more specific skills you already have, so they'll never
# get a meaningful edge and just clutter the graph.
VAGUE_CATEGORY_SKILLS = [
    "AI", "GenAI", "Data", "Programming", "Cloud", "Deployment",
    "LLM Tools", "Database Management Systems",
]


def auto_fix_orphans() -> str:
    """One-shot orphan cleanup, called from the UI button:
      1. Auto-links orphan skills to any project/course/cert whose text
         literally contains the skill name (substring match).
      2. Deletes the names in VAGUE_CATEGORY_SKILLS (and removes any
         references to them elsewhere in the data).
      3. Returns a markdown summary of what changed plus what's still
         disconnected.
    Safe to run repeatedly."""
    data = load_profile_data(DATA_PATH)
    link_changes = []
    removed = []

    # ---- Step 1: auto-link by substring match ----
    orphans = [n for n in graph.nodes() if graph.degree(n) == 0]
    for nid in orphans:
        node = graph.nodes[nid]
        ntype = node["type"]
        nname = node["name"].strip()
        nname_lower = nname.lower()
        if len(nname_lower) < 4:
            continue  # too short to match safely (e.g. 'AI', 'CSS')

        if ntype == "skill":
            for proj in data["projects"]:
                haystack = (proj["name"] + " " + proj.get("description", "")).lower()
                if nname_lower in haystack:
                    used = proj.setdefault("skills_used", [])
                    if nid not in used:
                        used.append(nid)
                        link_changes.append(f"`{nname}` → project **{proj['name']}**")
            for course in data["courses"]:
                haystack = (course["name"] + " " + course.get("description", "")).lower()
                if nname_lower in haystack:
                    cov = course.setdefault("skills_covered", [])
                    if nid not in cov:
                        cov.append(nid)
                        link_changes.append(f"`{nname}` → course **{course['name']}**")
            for cert in data["certifications"]:
                haystack = (cert["name"] + " " + cert.get("description", "")).lower()
                if nname_lower in haystack:
                    rel = cert.setdefault("skills_related", [])
                    if nid not in rel:
                        rel.append(nid)
                        link_changes.append(f"`{nname}` → certification **{cert['name']}**")

        elif ntype in ("course", "certification"):
            haystack = (nname + " " + node["text"]).lower()
            field = "skills_covered" if ntype == "course" else "skills_related"
            category = "courses" if ntype == "course" else "certifications"
            for skill in data["skills"]:
                sname = skill["name"].strip().lower()
                if len(sname) < 4:
                    continue
                if sname in haystack:
                    for entry in data[category]:
                        if entry["id"] == nid:
                            rel = entry.setdefault(field, [])
                            if skill["id"] not in rel:
                                rel.append(skill["id"])
                                link_changes.append(f"**{nname}** → skill `{skill['name']}`")
                            break

    # ---- Step 2: purge vague category-header skills ----
    for name in VAGUE_CATEGORY_SKILLS:
        sid = find_existing_id_by_name(data, "skills", name)
        if not sid:
            continue
        data["skills"] = [s for s in data["skills"] if s["id"] != sid]
        for proj in data.get("projects", []):
            proj["skills_used"] = [s for s in proj.get("skills_used", []) if s != sid]
        for course in data.get("courses", []):
            course["skills_covered"] = [s for s in course.get("skills_covered", []) if s != sid]
        for cert in data.get("certifications", []):
            cert["skills_related"] = [s for s in cert.get("skills_related", []) if s != sid]
        removed.append(name)

    # ---- Persist ----
    save_profile_data(data, DATA_PATH)
    rebuild_all()

    # ---- Report ----
    lines = ["### Orphan fix complete\n"]
    if link_changes:
        lines.append(f"**Added {len(link_changes)} link(s):**")
        lines += [f"- {c}" for c in link_changes]
    else:
        lines.append("_No safe auto-links found._")
    if removed:
        lines.append(f"\n**Removed {len(removed)} vague skill(s):** {', '.join(f'`{r}`' for r in removed)}")

    remaining = find_orphan_nodes()
    if remaining:
        names = [graph.nodes[n]["name"] for n in remaining]
        lines.append(
            f"\n⚠️ **{len(remaining)} still disconnected** (need manual linking "
            f"in `profile_data.json`): {', '.join(names)}"
        )
    else:
        lines.append("\n✅ **Graph is now fully connected.**")

    return "\n".join(lines)

# %% [markdown]
# ## 10. Gradio UI

# %%
import gradio as gr

with gr.Blocks(title="RAG Over Myself — Career Assistant") as demo:
    gr.Markdown("# 🧑‍💻 RAG-over-Myself: Agentic GraphRAG Career Assistant")
    gr.Markdown(
        "Ask questions about my background, paste a job description for a "
        "gap analysis, upload my resume or sync my GitHub to grow the "
        "knowledge base, or check the eval scorecard."
    )

    with gr.Tab("Ask about me"):
        chat_input = gr.Textbox(label="Ask a question about my background")
        chat_output = gr.Markdown()
        chat_btn = gr.Button("Ask")
        chat_btn.click(fn=ask_about_me, inputs=chat_input, outputs=chat_output)

    with gr.Tab("JD Match"):
        jd_input = gr.Textbox(label="Paste a job description", lines=8)
        jd_output = gr.Markdown()
        jd_btn = gr.Button("Analyze fit")
        jd_btn.click(fn=jd_match, inputs=jd_input, outputs=jd_output)
        gr.Markdown("_Remembers previous JDs pasted this session and flags recurring gaps._")

    with gr.Tab("Resume Upload"):
        gr.Markdown("Upload a resume PDF to auto-extract skills/projects/courses into your profile.")
        resume_input = gr.File(label="Resume PDF", file_types=[".pdf"])
        resume_output = gr.Markdown()
        resume_btn = gr.Button("Extract & merge into profile")
        resume_btn.click(fn=ingest_resume_pdf, inputs=resume_input, outputs=resume_output)

    with gr.Tab("GitHub Sync"):
        gr.Markdown("Pull your public repos + READMEs and add them as projects automatically.")
        gh_username = gr.Textbox(label="GitHub username")
        gh_max = gr.Slider(1, 10, value=5, step=1, label="Max repos to sync")
        gh_output = gr.Markdown()
        gh_btn = gr.Button("Sync from GitHub")
        gh_btn.click(fn=sync_github_projects, inputs=[gh_username, gh_max], outputs=gh_output)

    with gr.Tab("Evaluation"):
        gr.Markdown(
            "RAGAS-style scorecard: for each test question, an LLM judge "
            "scores faithfulness (grounded in retrieved context?) and "
            "relevancy (answers the question?) on a 1-5 scale."
        )
        eval_questions_input = gr.Textbox(
            label="Test questions (one per line, or leave default)",
            lines=4, value="\n".join(DEFAULT_TEST_QUESTIONS),
        )
        eval_summary = gr.Markdown()
        eval_table = gr.Dataframe(headers=["question", "answer", "faithfulness", "relevancy", "notes"])
        eval_btn = gr.Button("Run evaluation")
        eval_btn.click(fn=run_evaluation_gradio, inputs=eval_questions_input, outputs=[eval_summary, eval_table])

    with gr.Tab("Knowledge Graph"):
        stats_output = gr.Markdown(value=graph_stats())
        gr.Markdown("_Drag nodes to untangle, scroll to zoom, hover for details._")
        graph_html = gr.HTML(value=draw_graph())
        refresh_btn = gr.Button("Refresh graph")
        refresh_btn.click(fn=draw_graph, outputs=graph_html)

        gr.Markdown(
            "---\n### Fix disconnected nodes\n"
            "Links skills to any project/course/cert whose text mentions them, "
            "and removes vague category headers (`AI`, `GenAI`, `Data`, "
            "`Programming`, `Cloud`, `Deployment`, `LLM Tools`, "
            "`Database Management Systems`) that duplicate your more specific "
            "skills. Safe to click repeatedly."
        )
        orphan_output = gr.Markdown()
        orphan_btn = gr.Button("🔧 Fix disconnected nodes", variant="primary")

        def _run_orphan_fix():
            summary = auto_fix_orphans()
            return summary, graph_stats(), draw_graph()

        orphan_btn.click(fn=_run_orphan_fix, outputs=[orphan_output, stats_output, graph_html])

        gr.Markdown(
            "---\n### Compound-skill cleanup\n"
            "Splits old compound skill names (e.g. `AI & Machine Learning`) "
            "into atomic skills and asks the LLM to link remaining "
            "disconnected courses/certs to your existing skills where the "
            "description supports it."
        )
        cleanup_output = gr.Markdown()
        cleanup_btn = gr.Button("Run compound-skill cleanup")

        def _cleanup_and_refresh():
            summary = cleanup_profile_data()
            return summary, graph_stats(), draw_graph()

        cleanup_btn.click(fn=_cleanup_and_refresh, outputs=[cleanup_output, stats_output, graph_html])

if __name__ == "__main__":
    demo.launch()