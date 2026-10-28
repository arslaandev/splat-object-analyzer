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

Quality controls view coverage: `low` = 24 views, `medium` = 90 (default),
`high` = 192. Low coverage can miss objects; medium is a useful starting point.
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

## How it works

1. Render RGB/depth views from multiple positions, preferring clear space inside scene bounds.
2. RAM++ generates object labels from the views.
3. OWLv2 locates the labels in each image; depth lifts detections into 3D.
4. SAM segments candidate regions. Visible 3D surfaces associate repeated views
   and compatible names with an instance, instead of relying on box centres alone.
5. Parts and contents are linked to parent objects where geometry supports it.
   There is no fixed one-object or three-object cap per category.

This remains a prototype. Missed objects, incorrect names, duplicate instances,
and over-merges are possible. RAM++ uses its pretrained vocabulary. Camera
placement currently suits room scans better than isolated solids. SPZ support is
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
```

Append the path to an `interactions.json` file to either command to validate a
real output. Checks cover distinct neighboring instances, repeated observations,
more than three objects of one type, parent/child links, provenance, and box bounds.

## Publishing this folder

Push this folder as its own repository. Keep `.gitignore`: environments, model
weights, input scenes, generated outputs, and fetched dependency checkouts are
excluded. No credentials or remote Git repository are configured here.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and
dependency license references. The original upstream MIT license is preserved in
[LICENSE.md](LICENSE.md).

