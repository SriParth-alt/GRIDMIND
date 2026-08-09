# Deploying the live dashboard

The app is deployment-ready as-is. It carries everything it needs:

| Artifact | Size | Purpose |
|---|---|---|
| `data/simulation_data.csv` | 1.0 MB | Precomputed hourly demand / solar / price / carbon |
| `data/external/*.csv` | 1.5 MB | Raw measured inputs (NASA POWER, NYISO) |
| `microgrid_dqn.pth` | 80 KB | Trained agent — served, never retrained on the host |
| `Dockerfile` | — | CPU-only torch, listens on `$PORT` |

The 700 MB ASHRAE dump is **not** needed at runtime — only to regenerate the
cache (`python ashrae_pipeline.py --rebuild`).

---

## Option A — Render (fastest path)

Zero config beyond the repo: Render reads `render.yaml` and builds the Dockerfile.

1. Sign in at [render.com](https://render.com) with GitHub.
2. **New → Blueprint**, pick the `GRIDMIND` repo, confirm.
3. Wait for the first build (~5 min, mostly the torch download).

You get `https://gridmind.onrender.com`. WebSockets work on the free tier.

**Caveat:** the free tier sleeps after 15 minutes idle and takes ~50 s to wake.
Fine for a portfolio link; if you're putting the URL in a paper, either upgrade
to the $7/mo tier or use Option B.

---

## Option B — Hugging Face Spaces (best for a research audience)

Free, WebSocket-capable, and the natural home for an ML demo — a Spaces link
reads as native on a paper or CV.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space):
   name `gridmind`, SDK **Docker**, hardware **CPU basic (free)**.
2. Add it as a remote and push:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/gridmind
git push hf main
```

3. A Space needs YAML front matter in its `README.md`. To keep the GitHub README
   clean, add it only on the branch you push to HF:

```bash
git checkout -b hf-deploy
```

Prepend this to `README.md` on that branch, commit, and `git push hf hf-deploy:main`:

```yaml
---
title: GRIDMIND
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
```

---

## Option C — Any Docker host / locally

```bash
docker build -t gridmind .
docker run -p 8000:7860 gridmind
```

Then open <http://localhost:8000>. Set `PORT` to change the listening port.

---

## Before you deploy

Confirm the container path works without the raw dataset — this is the failure
mode that only shows up in production:

```bash
mv dataset dataset_off && python -c "
from ashrae_pipeline import load_ashrae_data
print(load_ashrae_data(n_days=5).shape)" ; mv dataset_off dataset
```

If it prints a shape, the deployment will work. If it tries to read
`dataset/train.csv`, the cache is missing — regenerate it with
`python ashrae_pipeline.py --rebuild` and commit the result.

## Cost note

Everything above is free. The app is a single CPU process serving a 128-unit
MLP; a free instance handles a live demo comfortably. Only sustained concurrent
traffic would justify a paid tier.
