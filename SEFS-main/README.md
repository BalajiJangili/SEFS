# Semantic Entropy File System (SEFS)

SEFS watches a raw folder, semantically understands PDF/TXT content, auto-groups files into real OS folders using symlinks (or copy fallback on Windows), and renders a live 2D semantic map.

## Features

- Real-time monitoring of `<root>/_raw` for create/modify/delete/move events
- Multi-format extraction: `.pdf`, `.txt`, `.md`, `.py`, `.docx`, `.pptx`, `.csv`
- Local semantic embeddings (`all-MiniLM-L6-v2`) with persistent ChromaDB storage
- Automatic cluster discovery with HDBSCAN (unknown cluster count)
- Cluster folder naming using TF-IDF keywords (optional Ollama naming hook)
- Semantic search + RAG answers over indexed files
- Conflict resolution workflow for ambiguous files (yellow nodes + manual choice)
- In-dashboard file operations: open, rename, delete selected `_raw` file
- Semantic diff event feed on updates (cluster movement + drift score)
- Near-duplicate detection (`similarity >= 0.95`)
- Similarity edges between related files in the semantic map
- Timeline tab showing file/cluster evolution and recent semantic events
- Manual override controls (drag-and-drop intent equivalent in Dash UI)
- Auto-generated cluster summaries
- Safe filesystem materialization: originals in `_raw` are never moved
- Interactive Dash + Plotly dashboard (`http://localhost:8050`)
- Graceful shutdown on `SIGINT`/`SIGTERM`

## Runtime Layout

When you run SEFS with `--root <path>`, it manages:

```text
<root>/
├── .sefs/
│   ├── vector_store/
│   ├── config.yaml
│   └── sefs.lock
├── _raw/
├── Uncategorized/
└── <semantic folders...>/
```

- `_raw` is the source of truth.
- Semantic folders contain symlinks to `_raw` files.
- On Windows, symlink creation can require admin/developer mode. If symlink creation fails, SEFS copies files into semantic folders.

## Installation

1. Use Python 3.10+.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --root ./my_files
```

Expected startup log:

```text
SEFS running. Monitoring: <root>/_raw | Dashboard: http://127.0.0.1:8050
```

Drop supported files into `<root>/_raw`: `.pdf`, `.txt`, `.md`, `.py`, `.docx`, `.pptx`, `.csv`.

## Dashboard

Open:

- `http://localhost:8050`

Behavior:

- `Semantic Map` tab:
  - Dot = one file
  - Color = semantic cluster/folder
  - Yellow diamond = ambiguity candidate (near two clusters)
  - Lines = high-similarity file edges
  - File operations: open, rename, delete selected file
  - Duplicate alerts panel + cluster summary panel
  - Manual override selector to force cluster assignment
- `Timeline` tab:
  - File count and cluster count over time
  - Recent semantic-diff events
- Search panel:
  - Semantic retrieval + RAG answer + cited context files

Use search:

1. Enter a natural-language query in the dashboard search box.
2. Click `Search`.
3. SEFS retrieves top matching files and uses them as context for generation.
4. The answer and cited retrieved context are shown under the search panel.

## Optional Ollama Cluster Naming

By default, TF-IDF naming is used.

Enable optional Ollama naming:

```bash
export SEFS_ENABLE_OLLAMA_NAMING=1
export SEFS_OLLAMA_MODEL=llama3
export SEFS_OLLAMA_TIMEOUT_SECONDS=8
```

If Ollama is unavailable or times out, SEFS silently falls back to TF-IDF naming.

## RAG Query Generation (Provider Options)

SEFS uses full retrieval + generation for dashboard queries.

Provider selection:

```bash
# auto | ollama | huggingface
export SEFS_RAG_PROVIDER=auto
```

### Option A: Ollama

```bash
export SEFS_ENABLE_OLLAMA_RAG=1
export SEFS_OLLAMA_RAG_MODEL=llama3
export SEFS_OLLAMA_RAG_TIMEOUT_SECONDS=20
```

Notes:
- Default Ollama RAG timeout is `45s`.
- If you see timeout errors on first run/model warm-up, set `SEFS_OLLAMA_RAG_TIMEOUT_SECONDS=90`.

If `ollama` is not in PATH, SEFS also checks common install paths:
- `/opt/homebrew/bin/ollama`
- `/usr/local/bin/ollama`
- `~/.local/bin/ollama`

### Option B: HuggingFace Local Model

```bash
export SEFS_RAG_PROVIDER=huggingface
export SEFS_HF_RAG_MODEL=google/flan-t5-small
export SEFS_HF_RAG_MAX_NEW_TOKENS=220
```

Offline behavior:

```bash
# Default in this project: local-only (no remote downloads, fail-fast if uncached)
export SEFS_HF_LOCAL_ONLY=1
```

If you want first-time download from HuggingFace, enable network mode:

```bash
export SEFS_HF_LOCAL_ONLY=0
```

## Logging

Set log level with:

```bash
export SEFS_LOG_LEVEL=DEBUG
```

Default is `INFO`.

## Smoke Checks

After installing dependencies:

1. Import smoke:

```bash
python -c "import config, extractor, semantic_engine, folder_organizer, watcher, dashboard, main; print('ok')"
```

2. Syntax smoke:

```bash
python -m compileall .
```

3. Organizer smoke (manual):
- Start SEFS.
- Put 2-3 small `.txt` files in `_raw`.
- Confirm semantic folders appear with symlinks/copies.
4. Winning features smoke:
- In dashboard map, verify similarity edges and cluster summaries show up.
- Edit an indexed file so topic changes, verify recent event includes semantic drift.
- Pick a yellow ambiguous node and apply override; verify folder assignment persists.
- Open timeline tab and verify file/cluster curves update.

## Full Manual End-to-End Protocol

1. Start SEFS with an empty root.
2. Create these 7 test files in `_raw`:
- `quantum_mechanics.txt`
- `particle_physics.txt`
- `chocolate_cake.txt`
- `pasta_recipe.txt`
- `machine_learning.txt`
- `deep_learning.txt`
- `random_notes.txt`
3. Wait ~30 seconds.
4. Verify semantic folders plus `Uncategorized` exist.
5. Verify each semantic folder entry points to `_raw` originals (or copied fallback on Windows).
6. Open dashboard and verify ~7 points split into semantic groups.
7. Add another physics file and verify it joins the physics cluster.
8. Delete a cooking file and verify folder/dashboard updates.
9. Confirm no crashes, no event loops, and no orphaned entries.

## Troubleshooting

- If startup reports existing lock file, ensure no other SEFS process is running for the same root, then remove `<root>/.sefs/sefs.lock`.
- If no clusters are formed, files may be too short (`< 50` chars) or unrelated; they go to `Uncategorized`.
- If PDFs appear empty, they may be scanned/image-only or protected.
- On Linux, ensure `xdg-open` is available for click-to-open.

## Notes

- SEFS watches only `_raw`, not semantic folders, to avoid self-trigger loops.
- Cluster labels are recomputed globally on each relevant event.
- SEFS never auto-moves files out of `_raw`; rename/delete only happen on explicit user action.
