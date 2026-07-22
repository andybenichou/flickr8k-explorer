# Flickr8k Explorer

A local tool for exploring the [Flickr8k](https://huggingface.co/datasets/jxie/flickr8k) dataset
(8,000 images, 5 captions each): browse the samples, search them by meaning, inspect any example in
detail, and see the shape of the whole dataset on a 2D map of its embedding space.

Everything runs on one machine. No cloud services, no managed database, no paid APIs.

---

## TL;DR for the reviewer

Four commands. About 12 minutes total, most of it downloads that only happen once.

```bash
uv sync --extra dev                     # ~3 min   (downloads PyTorch)
npm --prefix frontend install           # ~15 s
uv run python -m backend.ingest         # ~7 min   (one time; see the breakdown below)
uv run uvicorn backend.app.api:app      # starts in ~2 s
```

Then, in a second terminal:

```bash
npm --prefix frontend run dev           # open http://localhost:5173
```

**In a hurry?** Replace the ingestion with `uv run python -m backend.ingest --limit 200`. It ingests
600 images instead of 8,000 and finishes in about 90 seconds after the dataset download. Every
feature works, just on a smaller set.

**Want to check it works before committing to the full run?** `uv run pytest` takes under a second
and needs neither the dataset nor the model weights.

---

## What to look at once it is running

The tool has five views. Suggested two-minute tour:

1. **Browse** — the grid, 8,000 images, scroll to load more. Filter by split in the dropdown.
2. **Search, semantic mode** — type `a dog jumping over a fence`. The top hits are dogs leaping over
   tree stumps, agility hurdles and a pole with fire on the ends. None of those captions contain the
   word "fence". This is the difference between embedding search and keyword search, and it is the
   core of what makes the tool useful.
3. **Search, caption-text mode** — run the same query with the toggle set to *Caption text*. BM25
   only matches the literal words. Comparing the two result sets tells you something real about
   annotation coverage.
4. **Click any image** — the detail panel opens: full image, all five captions with word counts,
   resolution and file metadata, and its nearest neighbours in CLIP space. Near-identical
   neighbours are duplicates or over-represented scenes.
5. **Map** — the UMAP projection of all 8,000 embeddings, coloured by cluster. Hover for a
   thumbnail, click to open the detail panel. Type a search query while on this view: the hits stay
   coloured and everything else dims, which turns the query into "where does this concept live in
   the dataset".
6. **Stats** — split sizes, caption-length distribution, resolutions, most frequent words. Note that
   "dog" appears 8,111 times across 40,000 captions. That is the dataset's content bias in one line.

One thing worth knowing: **the map shows a small detached group far to the left of the main cloud.**
That is not a rendering bug. It is a genuinely isolated group in CLIP space, and surfacing that kind
of structure is exactly what the view is for.

---

## Requirements

- **Python 3.11 or 3.12** via [uv](https://docs.astral.sh/uv/). Install uv with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`. You do not need to install Python yourself, uv
  fetches the right version.
- **Node.js 18+**
- **About 4 GB of free disk space**, broken down below
- An internet connection for the first run

### Disk usage

| Location | Size | Note |
| --- | --- | --- |
| `.venv/` | 1.1 GB | Mostly PyTorch |
| `~/.cache/huggingface` (dataset) | 1.0 GB | Parquet shards, shared across runs |
| `~/.cache/huggingface` (CLIP weights) | 577 MB | Downloaded on the first embedding pass |
| `data/` | 1.2 GB | 1.0 GB images, 169 MB thumbnails, 11 MB SQLite, 16 MB vectors |

Only `data/` lives in the repo, and it is gitignored. Deleting it and re-running the ingestion
rebuilds everything from the caches in about 1.5 minutes.

---

## Setup

```bash
git clone <this-repo> && cd Mobileye
uv sync --extra dev
npm --prefix frontend install
```

`uv sync` creates `.venv` and installs the pinned dependencies from `uv.lock`. The slow part is
PyTorch, roughly 3 minutes on a normal connection.

---

## Step 1: ingest the dataset

```bash
uv run python -m backend.ingest
```

This is a **one-time** step. Measured on an Apple Silicon laptop, CPU only:

| Stage | Time | What it does |
| --- | --- | --- |
| download | 2 to 5 min | Pulls the parquet shards from HuggingFace. Cached, so it is free on re-runs. |
| decode | 20 s | Writes 8,000 full images and 320 px thumbnails to disk, fills SQLite (images, captions, FTS5 index). |
| embed | 59 s | CLIP ViT-B/32 over every image, ~135 images/s, into `data/embeddings.npy` (16 MB). Adds a one-off 577 MB model download the first time. |
| project | 15 s | UMAP to 2D plus KMeans clusters, written back into SQLite. |

**First run: about 7 minutes**, dominated by the two downloads. **Any later re-run: about 1.5
minutes**, since both caches are warm.

The pipeline prints a progress bar per stage, so you can see where it is.

### Flags

| Flag | Effect |
| --- | --- |
| `--limit N` | Only ingest N images per split. `--limit 200` gives 600 images in ~90 s. |
| `--only {decode,embed,project}` | Re-run a single stage against the existing database. |
| `--skip-embeddings` | Browse and caption search only, no CLIP, no semantic search, no map. |
| `--skip-projection` | Everything except the map. |

Stages are separable on purpose: embedding is the expensive one, and a failure in a later stage
should never force you to redo it.

Re-running the full ingestion is safe. It rebuilds the tables from scratch and reuses both caches.

---

## Step 2: run the application

Two terminals.

```bash
# terminal 1 - API on http://127.0.0.1:8000, interactive docs at /docs
uv run uvicorn backend.app.api:app --reload
```

```bash
# terminal 2 - UI on http://localhost:5173
npm --prefix frontend run dev
```

Open <http://localhost:5173>.

**Startup and response times:**

- The API starts in about 2 seconds. It memory-maps the vectors rather than loading them.
- Browsing, caption search and stats respond in a few milliseconds.
- **The first semantic search takes 3 to 5 seconds**, because that is when the CLIP text encoder
  loads. Every search after that is a few milliseconds. This is deliberate: a reviewer who never
  runs a semantic query never waits for the model.

### Single-process alternative

If you would rather run one process, build the frontend and let the API serve it:

```bash
npm --prefix frontend run build
uv run uvicorn backend.app.api:app     # UI and API together on http://127.0.0.1:8000
```

---

## Tests

```bash
uv run pytest
```

19 tests, under a second. The suite builds a miniature 4-image dataset on disk and drives the HTTP
API against it with a stub embedder, so it needs neither the real dataset nor the model weights.
That is also the proof that the `Embedder` seam is real and not decorative.

---

## Project layout

```
backend/
  ingest.py           one-shot pipeline: parquet -> images + SQLite + vectors + projection
  app/
    api.py            FastAPI routes (HTTP only, no logic)
    services.py       application state, search orchestration, lazy model loading
    repository.py     every SQL statement in the project
    embeddings.py     Embedder interface, CLIP implementation, vector index
    db.py             schema and connection helpers
    models.py         response schemas (the contract with the frontend)
    config.py         settings, overridable via F8K_* environment variables
  tests/              fixture dataset + stub embedder
frontend/src/
  App.tsx             view and query state
  api.ts              typed client
  hooks.ts            data-fetching hooks
  components/         ImageGrid, SearchBar, DetailPanel, ProjectionMap, StatsPanel
data/                 generated by the ingestion, gitignored
```

---

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/dataset` | Counts, splits, embedding model in use |
| `GET /api/images?split=&offset=&limit=` | Paginated browse |
| `GET /api/images/{id}` | Full detail with all captions |
| `GET /api/images/{id}/similar?limit=` | Nearest neighbours in embedding space |
| `GET /api/search?q=&mode=semantic\|text&split=&limit=` | Ranked search |
| `GET /api/projection?split=` | 2D coordinates and cluster labels |
| `GET /api/stats` | Dataset composition |

Interactive documentation at <http://127.0.0.1:8000/docs>.

---

## Configuration

Any setting in `backend/app/config.py` can be overridden with an `F8K_`-prefixed environment
variable or a `.env` file. Swapping the embedding model is one command:

```bash
F8K_CLIP_MODEL=ViT-L-14 F8K_CLIP_PRETRAINED=laion2b_s32b_b82k uv run python -m backend.ingest --only embed
```

---

## Troubleshooting

**"Backend unavailable" in the UI, or a 503 from the API.** The dataset has not been ingested yet.
Run step 1.

**Semantic search is greyed out.** The database has no embeddings. Run
`uv run python -m backend.ingest --only embed`.

**The map says no projection is stored.** Run `uv run python -m backend.ingest --only project`
(15 seconds).

**The first semantic search seems to hang.** It is loading CLIP, 3 to 5 seconds, once per API
process. The uvicorn log prints `Loading CLIP model ...` when it starts.

**`uv sync` fails on Python version.** The project needs 3.11 or 3.12. `uv python install 3.11` then
retry; uv manages the interpreter, your system Python is not used.

---

## Design notes

The reasoning behind the main choices, what was deliberately left out, and where this design would
stop scaling: [DESIGN.md](DESIGN.md).
