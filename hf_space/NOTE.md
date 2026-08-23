# This folder is a deployment copy

Everything here except `README.md`, `requirements.txt` and this file is a
**copy** of the repository root. It exists so the Hugging Face Space can be
uploaded as a self-contained directory.

## If you change the root, change this too

`app.py` and `src/` here will silently drift out of sync with the root copies.
That is the same class of problem the code audit was about, so:

    Copy-Item ..\app.py .\app.py    # then re-apply the two edits below
    Copy-Item -Recurse -Force ..\src .\src

## The three deliberate differences in app.py

1. `import spaces` added to the import block.
2. `_zerogpu_probe()` added after `DEFAULT_CONFIDENCE_FLOOR`. ZeroGPU crashes
   at startup if a Space on ZeroGPU hardware has no `@spaces.GPU`-decorated
   function anywhere in its code. Inference itself stays on CPU — see the
   docstring for why.
3. A `gr.Examples` block after `clear_btn`, wiring up `data/samples/`.

Both `import spaces` and the decorator are no-ops outside ZeroGPU, so this
file still runs locally unchanged.

## Other differences

- `requirements.txt` adds `spaces` and drops `ultralytics`, `scikit-learn`
  and `tqdm` (unused in the Space). Do not add a CPU-only torch index URL —
  ZeroGPU expects standard CUDA wheels.
- `README.md` carries the Space YAML frontmatter and a shorter summary. The
  root README is the full one.
- `data/samples/` holds 15 images, one per class, rather than the root's
  full set — `gr.Examples` gets unusable past a dozen or so.
- No `.pth`, `notebooks/`, `reports/` or `scripts/` — the Space fetches the
  checkpoint from the Hub via `src/weights.py` and does not evaluate.

## Upload

    cd hf_space
    hf upload PushkarKumar/plant-disease-densenet169 . --repo-type space
