# Source and dependency notices

This project includes and modifies source from
[nigelhartman/splat_analyzer](https://github.com/nigelhartman/splat_analyzer)
at commit `a3cd884`. Its MIT license and copyright notice are preserved verbatim
in `LICENSE.md`.

Changes to that source include automatic label discovery, mask-based 3D instance
association, alternative names and parent/child relationships, output provenance,
clearance-first cameras, removal of the per-category instance cap, regression
checks, standalone setup scripts, and a generic local result viewer.

Fetched dependencies and model weights are not vendored into this source folder:

| Dependency | Source / version | License reference |
| --- | --- | --- |
| RAM++ inference source | [recognize-anything](https://github.com/xinyu1205/recognize-anything), `7cb804a8609e9f4b1a50b7f31436d2df40bb9481` | Apache 2.0; copy in `licenses/recognize-anything-LICENSE.txt` |
| Apple Metal renderer | [iffyloop/gsplat-mps](https://github.com/iffyloop/gsplat-mps), `10372c263f8e54427183de8a5855a62f94c2f206` | AGPL v3; copy in `licenses/gsplat-mps-LICENSE.txt` |
| CUDA renderer | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat), `1.5.3` | Upstream project's license |
| RAM++ weights | [recognize-anything-plus-model](https://huggingface.co/xinyu1205/recognize-anything-plus-model), `84d4aee3a0265c4e0df1f714f0572011d1bf2ec3` | Model repository's license and model card |
| SAM weights | [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base), `d0d9250bd6a38ebbb901aa29e8d37787e366f7be` | Model repository's license and model card |
| OWLv2 weights | [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble), `cfd3195ba4ea9592eec887ded089f4c08eff231d` | Model repository's license and model card |
| Three.js viewer runtime | [Three.js](https://github.com/mrdoob/three.js), `0.179.0` | Upstream project's license |
| Spark viewer runtime | [Spark](https://github.com/sparkjsdev/spark), `2.0.0` | Upstream project's license |

Python dependency versions are recorded in the requirements files. Dependencies
and model weights retain their respective licenses; the top-level license does
not replace them.
