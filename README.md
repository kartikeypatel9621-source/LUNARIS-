# LUNARIS frontend

Single-file, self-contained frontend for ISRO Problem Statement 26166 —
multi-modal, Sun-angle and scale invariant image correspondence using
Chandrayaan‑2 optical images (OHRC, TMC‑2, IIRS), registered against
LRO NAC / SELENE reference imagery.

## Run it

No build step. Just open `index.html` in a browser, or serve it:

```bash
python3 -m http.server 8080
# then visit http://localhost:8080
```

## What's inside

- `index.html` — everything: markup, styling, the 3D Moon hero (Three.js,
  loaded from cdnjs), charts (Chart.js, loaded from cdnjs), and the full
  app logic. No npm install, no bundler.

## Sections

Mission Overview · Dataset Explorer · Image Registration ·
Correspondence Analysis · AI Models · Experiments · Evaluation ·
Results · About/Documentation — matching the brief's required navigation.

## How the pipeline currently runs

Out of the box, this frontend runs a **real, in-browser classical
baseline** on uploaded images: Harris-response corner detection → grid-based
spatially-uniform keypoint selection → normalized-patch descriptors →
ratio-test matching → RANSAC affine estimation → least-squares refit →
canvas-based warping and difference view. Nothing about that path is
hard-coded — it genuinely analyses the pixels you upload (or the "Load
synthetic validation pair" button, which also embeds a known ground-truth
transform so you can sanity-check the estimated one against it).

The three deep-learning engines (SuperPoint+LightGlue, LoFTR, Hybrid) are
clearly labelled **"Simulated engine"** in this standalone build — their
numbers are derived from the live classical run rather than fabricated
from nothing, but they are not real model inference. That's what the
companion backend is for.

## Wiring this up to the real backend

A matching FastAPI backend (`lunaris-backend`) implements all four engines
for real, including actual SuperPoint+LightGlue and LoFTR inference. To
connect them:

1. Run the backend (see its own README): `uvicorn app.main:app --port 8000`.
2. In `index.html`, find the `runBtn` click handler (search for
   `document.getElementById('runBtn').addEventListener`).
3. Replace the in-browser pipeline call with a fetch to the backend, e.g.:

```js
const form = new FormData();
form.append('source', sourceFileBlob);
form.append('reference', referenceFileBlob);
form.append('model', selectedModel); // 'orb' | 'superpoint_lightglue' | 'loftr' | 'hybrid'

const res = await fetch('http://localhost:8000/api/register', { method: 'POST', body: form });
const data = await res.json();
// data.matches, data.metrics, data.registered_image_png_b64, data.difference_image_png_b64
```

You'll need to keep a reference to the original uploaded `File` objects
(the current code only keeps decoded grayscale arrays for the in-browser
path) and render `registered_image_png_b64` / `difference_image_png_b64`
by setting them as `data:image/png;base64,...` image sources instead of
drawing the client-computed canvases.

4. Update `allow_origins` in the backend's `app/main.py` CORS config to
   match wherever you serve this frontend from.

## Notes

- Fonts: Google Fonts (Space Grotesk, JetBrains Mono, Inter) via a
  `<link>` tag — requires internet access on first load, or self-host
  the font files if you need a fully offline deployment.
- Libraries: Three.js r128 and Chart.js 4.4.0, both loaded from
  `cdnjs.cloudflare.com`.
- Everything else — the pipeline, the 3D scene, the synthetic test-pair
  generator, the charts — is inline vanilla JS with no dependencies beyond
  those two.
