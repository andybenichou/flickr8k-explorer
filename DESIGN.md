# Design notes

What this tool is for, why it is built this way, and where the design would break.

## What a dataset tool has to answer

Flickr8k is an image-captioning dataset: 8,000 photos, five human captions each. A researcher
opening it for the first time is not asking "show me the files". They are asking:

- Does this dataset contain the concepts I care about, and how many examples of each?
- Are there near-duplicates, or scenes so over-represented that a model will overfit them?
- What does the caption annotation actually look like, and how much does its style vary?
- Given one interesting example, what else in the set resembles it?

Those questions drove the feature set. Everything in the tool exists to answer one of them; nothing
was added because it was easy.

## The central choice: one embedding model, used everywhere

A single CLIP ViT-B/32 pass over the images produces the vectors behind three of the five views:

- **semantic search** compares a text embedding against the image embeddings, so a query is matched
  by meaning rather than by whether the annotator happened to use that word;
- **nearest neighbours** in the detail panel surface duplicates and over-represented scenes;
- **the map** is a UMAP projection of the same vectors, so a search can be highlighted directly on it,
  turning a query into "where does this concept live in the dataset".

One artefact, three capabilities. The alternative (keyword search, a perceptual hash for duplicates,
a separate feature extractor for the map) would be more code, three things to keep consistent, and
strictly less useful.

ViT-B/32 was picked over a larger CLIP because it embeds the full dataset in about a minute on a
laptop CPU. Retrieval quality on 8k photos of everyday scenes is not the bottleneck here; the
setup time a reviewer has to sit through is. `F8K_CLIP_MODEL` swaps it in one command.

Caption search is kept alongside semantic search rather than replaced by it. They fail differently:
BM25 finds the literal word an annotator used, CLIP finds the concept. Comparing the two result sets
for the same query is itself informative about annotation coverage, so both are one click apart.

## Storage: SQLite and a numpy array

The structured data (images, captions, FTS5 index, projection coordinates) lives in one SQLite file.
The vectors live in a `float32` `.npy` array, memory-mapped at startup, where row *i* corresponds to
`images.row_index = i`.

Search is an exhaustive cosine scan: `matrix @ query`, then a partial sort. For 8,000 x 512 that is
16 MB and a few milliseconds per query. An ANN index (FAISS, hnswlib) or an external vector database
would add a dependency, a build step, and a second source of truth to keep in sync with SQLite, in
exchange for a speedup below the threshold of perception. It would start to pay off somewhere around
10^6 vectors; `VectorIndex` is a small enough interface (`search`, `vector`, `__len__`) that swapping
the backend is a local change when that day comes.

The same reasoning applies to filtering. Rather than encoding metadata into the index, a filtered
search passes the matching row indices as `candidates` and the scan runs over that subset. Metadata
filters and semantic ranking compose without either side knowing about the other.

## Precompute at ingestion, serve from SQL

UMAP and KMeans run once during ingestion and their output is written back into the `images` table.
The map endpoint is then a plain `SELECT`, and the frontend receives coordinates already normalised
to `[0, 1]`. Nothing heavy happens per request.

The same principle governs thumbnails: 320 px JPEGs are generated at ingestion, so the grid never
transfers a full-resolution image. This is the difference between a grid that scrolls smoothly and
one that saturates the connection.

The one deliberate exception is the CLIP text encoder, loaded lazily on the first semantic query.
Browsing, caption search and stats therefore work on a cold start, and a reviewer who never runs a
semantic query never waits for the model.

## Ingestion as separable stages

`ingest.py` runs four stages: download, decode, embed, project. Each can be run alone via `--only`,
and `--limit` caps the number of images per split.

The reason is the failure mode. Embedding is the expensive stage; a crash in UMAP after it should not
force a re-run. `--limit 200` also makes the whole pipeline verifiable in under a minute, which is
how it was developed and how a reviewer can sanity-check it before committing to the full run.

## Layering

```
api.py          HTTP: routing, validation, status codes. No logic.
services.py     application state, search orchestration, lazy model loading
repository.py   every SQL statement in the project
embeddings.py   Embedder protocol + CLIP implementation; VectorIndex
```

The boundaries are drawn where the substitutions are plausible. `Embedder` is a protocol because
swapping CLIP for DINOv2, SigLIP, or a domain-specific model is the most likely extension of this
tool; the test suite already exercises that seam with a stub embedder, which is also why the tests
need neither the dataset nor the model weights. `repository.py` holds all SQL because the storage
engine is the second most likely thing to change. `models.py` is the shared contract with the
frontend, mirrored in `types.ts`.

Free-text queries are quoted and stripped before reaching FTS5. Raw user input in an FTS5 MATCH
expression is a syntax error waiting to happen (`"`, `*`, `NEAR`), and quoting each token also gives
predictable AND-of-terms behaviour instead of accidental operators.

## Frontend

React with Vite, and no other runtime dependency. Two decisions worth naming:

**Infinite scroll instead of windowing.** A virtualisation library would mean fixed row heights or
measurement logic fighting a responsive CSS grid. Native `loading="lazy"` already keeps offscreen
thumbnails off the network, and an `IntersectionObserver` appends the next page of 60. The DOM stays
cheap without the library or the layout constraints it imposes.

**The map is a canvas.** 8,000 SVG or DOM nodes would stutter on hover; a canvas redraws the whole
cloud in one pass, and hit-testing is a linear scan over 8,000 points, which is far below a frame
budget. Rendering-heavy, interaction-light views are what canvas is for.

Search is debounced, and every fetch is cancellable, so the results shown always correspond to the
query currently in the box.

## What is deliberately absent

- **Authentication, multi-user state, deployment config.** The brief is a single local machine.
- **A router.** Views are top-level state. Deep links would be the first thing to add if this tool
  were shared between researchers, since "look at this image" is a natural thing to send a colleague.
- **Writes of any kind.** No labelling, no annotation editing, no favourites. The dataset is
  read-only here, which is what keeps the whole backend a pure query layer.
- **Caption embeddings.** Embedding the 40,000 captions as well would enable caption-to-caption
  clustering and image-text alignment scoring (finding examples whose caption poorly matches its
  image, a genuinely useful data-quality signal). It is the single most interesting extension, and it
  was left out to keep the scope honest rather than because it is hard: the `Embedder` interface
  already exposes `embed_texts`.

## Where this breaks

| Scale | What gives | What to do |
| --- | --- | --- |
| ~10^5 images | Thumbnail directory and browse pagination stay fine; exhaustive search stays under ~50 ms | Nothing |
| ~10^6 images | Exhaustive cosine scan becomes noticeable; UMAP on the full set gets slow | Swap `VectorIndex` for hnswlib; project a sample |
| ~10^7 images | SQLite writes during ingestion and single-file storage become the constraint | Postgres + pgvector, object storage, distributed ingestion |
| Multiple datasets | `config.py` assumes one dataset per data directory | Add a `datasets` table and scope every query by dataset id |

The ingestion is also single-process. Parallelising the decode stage across cores would be the first
optimisation if the dataset were ten times larger; at 8k images it finishes in the time it takes to
read this file.
