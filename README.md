# Splat Object Analyzer

Input a Gaussian splat (`.spz` or standard 3DGS `.ply`) and get an object inventory
with labels, 3D positions, bounding boxes, supporting images, and parent/child
relationships. Object names are discovered automatically.

This is the standalone research prototype developed from
[Nigel Hartman's Splat Analyzer](https://github.com/nigelhartman/splat_analyzer),
with automatic discovery and mask-based instance resolution added.

## Setup

Use **Python 3.11 or 3.12** (tested with 3.12), Git, and a supported GPU. Rendering
requires Apple Silicon/Metal or NVIDIA/CUDA; there is no CPU renderer.

### Apple Silicon

Install Xcode command-line tools if needed (`xcode-select --install`), then:

```bash
./setup.sh mac
```

This creates `.venv`, installs the tested dependency versions, and fetches the
pinned RAM++ and Metal renderer sources. The renderer compiles on its first use.
`uv` is used if available; otherwise setup uses Python's `venv` and `pip`.

### NVIDIA / GCP

Install the NVIDIA driver, CUDA toolkit (including `nvcc`), and CUDA-enabled
PyTorch/torchvision. Choose the build for your machine using the
[official PyTorch installer](https://pytorch.org/get-started/locally/).
For a CUDA 12.8 environment, for example:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
./setup.sh cuda
```

Setup checks that PyTorch can see CUDA and that `nvcc` exists. It detects the GPU
compute capability for compiling gsplat (A100: `8.0`). Keep the PyTorch CUDA build
and compiler toolkit compatible. The CUDA backend is included from the upstream
project; this packaged version has been exercised on Apple Silicon, not on an
NVIDIA machine yet.

## Analyze a scene

```bash
./analyze-splat.sh /path/to/scene.spz
```

No object-name list is required. Results go to
`out_<scene-name>_<input-sha256-prefix>/` in the current directory.

```bash
./analyze-splat.sh /path/to/scene.ply --quality high --job_dir ./outputs/my-scene
```

Quality controls the maximum view budget: `low` = 24 views, `medium` = 90
(default), `high` = 192. Candidate cameras are checked with low-resolution
previews; only clear final views are sent to detection. The accepted count can
be lower than the budget. For thorough coverage, use `--quality high`.
An optional targeted search remains available:

```bash
./analyze-splat.sh /path/to/scene.spz --prompt "painting, chair, table"
```

The first analysis downloads the pretrained RAM++, OWLv2, SAM, and tokenizer
files (several GB). They stay in `.cache/` and are excluded from Git.
Set `HF_HOME` to reuse an existing Hugging Face cache. After the models are
cached, set `HF_HUB_OFFLINE=1` for an offline analysis run. Model inference runs
locally; the input scene is not uploaded to a model service.

## View results

Start the interactive local viewer:

```bash
.venv/bin/python view-result.py
```

Open **http://127.0.0.1:8088/** and choose a `.ply` or `.spz` file. Uploading
starts automatic analysis at **high** coverage by default. The scene is prepared
for viewing while the status panel reports the real analysis stage and view
counts. Completed annotations load automatically, and **Download results** saves
the JSON. The menu retains completed scenes during the server session so you can
switch between them. Processing continues when the browser page is refreshed.

Use **WASD** to move, **Q/E** for down/up, and **Shift** for faster movement.
Click the scene to focus navigation; drag to look around. Keyboard movement is
disabled while typing or using menu fields. This is free movement without collision.
The home button resets the view to an automatically selected camera.

Uploads and analysis run on the machine hosting the viewer. Jobs are saved in
`outputs/viewer/` (excluded from Git); use `--jobs-dir /path/to/jobs` to change it.
One analysis runs at a time. The upload limit is **2 GiB**; supported analysis
formats remain standard 3DGS PLY and the inherited SPZ v1–v3 reader. Unsupported
files and failed jobs produce a visible error. Models must be installed/cached
or downloadable by the server's analysis environment.

You can also start the viewer with an existing result:

```bash
.venv/bin/python view-result.py ./out_scene_HASH/interactions.json
```

Open **http://127.0.0.1:8088/**. The server automatically loads the scene named in
the result, with an initial camera chosen from its supporting views. Use
`--port 8089` if needed. After moving results to another machine, supply
`--scene /path/to/scene.spz` to replace the original input path.

The viewer displays top-level objects. In Settings, **Show parts and contents**
also displays books, drawers, pillows, and other linked records. Hover a marker
to inspect its 3D box. You can also load a splat and JSON manually through
Settings. The viewer uses the pinned Three.js/Spark browser libraries from their
CDNs, so the browser needs internet access for those scripts.

For a GCP VM, forward the port over SSH and run the same viewer command on the VM:

```bash
ssh -L 8088:127.0.0.1:8088 USER@VM
```

## Output files

| File | Contents |
| --- | --- |
| `interactions.json` | Resolved records, labels, alternatives, positions, boxes, evidence, and parent/child links |
| `candidates.json` | Original candidates before mask-based instance resolution |
| `discovery.json` | Automatically generated labels and tags for every view |
| `transforms.json` | Camera poses and intrinsics |
| `camera_diagnostics.json` | Candidate positions, preview measurements, and selected camera indices |
| `frames/` | Rendered RGB views and depth maps |
| `instance_masks/` | Cached segmentation masks |
| `object_surfaces.npz` | Observed 3D surface points keyed by resolved object ID |
| `input.ply` | SPZ conversion, when needed; the source input is unchanged |

Each record has one primary `label`. Other names for the same resolved instance
are retained in `alternative_labels`. A top-level record has `parent_id: null`;
child records reference their parent, which lists them in `children`.
`source_candidate_ids` and `mask_evidence` let you trace every merge back to its
source detections. IDs are local to one result file, not persistent across runs.

Positions and sizes use the input splat's coordinates. Units are **not assumed
to be meters**. Refined boxes bound observed segmented surfaces; hidden geometry
is not measured. Detector confidence scores are not calibrated probabilities.

`bounding_box.min/max`, the top-level `position`, and `scale` retain the legacy
axis-aligned estimate. Segmented records with enough surface samples also contain `oriented_box` with its
own `position`, `scale`, and XYZW `rotation`. The viewer uses this oriented box
for markers, outlines, and highlighting. It fits yaw around source Y, the same
up axis currently used for camera placement; arbitrary tilted scenes need an
inferred up direction. Boxes remain estimates of the observed surfaces.
Surface fusion compares each mask observation with the strongest observation of
that candidate; inconsistent neighboring-object surfaces are excluded. The
`geometry_supporting_view_count` counts retained geometry views separately from
the original detection votes. Opposite faces without sufficient overlap can be
excluded, so boxes may cover only the supported visible part of an object.

## How it works

1. Sample clear positions inside robust scene bounds, ignoring almost transparent
   Gaussians for this initial centre-clearance check. Render four low-resolution
   RGB/depth previews per candidate using the full splat geometry. Prefer
   candidates with at least three usable horizontal views and sufficient image
   detail relative to the scene's clearer previews, then select for
   visible-surface coverage and spatial separation. Render the final 90-degree
   sweeps and reject views dominated by near surfaces or empty background.
2. RAM++ generates object labels from the views.
3. OWLv2 locates the labels in each image; depth lifts detections into 3D.
4. SAM segments candidate regions. Visible 3D surfaces associate repeated views
   and compatible names with an instance, instead of relying on box centres alone.
5. Parts and contents are linked to parent objects where geometry supports it.
   There is no fixed one-object or three-object cap per category.

No original reconstruction cameras or manual starting positions are required.
If too few cameras pass validation, coverage is reduced and reported instead
of filling the remaining slots with unchecked cameras. If no cameras pass,
the analysis stops with an explicit coverage error.

This remains a prototype. Missed objects, incorrect names, duplicate instances,
and over-merges are possible. RAM++ uses its pretrained vocabulary. Camera
placement uses room-oriented preview heuristics, not a watertight occupancy map.
Open outdoor scenes, disconnected rooms, isolated solids, and incomplete scans
can be rejected or poorly covered. Image/depth checks cannot guarantee that a
view is inside a room or physically unobstructed. SPZ support is
the inherited v1–v3 converter; newer SPZ variants and nonstandard PLY layouts may
need conversion before use.

On the reviewed bedroom, 98 initial candidates became 45 top-level records and
18 linked parts/contents: two bed candidates became one bed, 21 shelf/bookcase
candidates became six storage units, and all 12 book candidates were retained as
contents. These are model outputs, not verified ground-truth counts.

## Checks

```bash
.venv/bin/python check-auto.py
.venv/bin/python check-instances.py
.venv/bin/python check-cameras.py
.venv/bin/python check-boxes.py
.venv/bin/python check-server.py
node check-coordinates.mjs
```

Append the path to an `interactions.json` file to the first two commands to validate a
real output. Checks cover distinct neighboring instances, repeated observations,
more than three objects of one type, parent/child links, provenance, and box bounds.
For camera diagnostics, pass an output directory to `check-cameras.py`.
Pass an `interactions.json` path to `check-boxes.py` to check fitted orientations
and containment of the saved surface points. Server checks use temporary fake
analysis processes; the actual model pipeline must also be exercised on scenes.

## Publishing this folder

Push this folder as its own repository. Keep `.gitignore`: environments, model
weights, input scenes, generated outputs, and fetched dependency checkouts are
excluded. No credentials or remote Git repository are configured here.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and
dependency license references. The original upstream MIT license is preserved in
[LICENSE.md](LICENSE.md).

