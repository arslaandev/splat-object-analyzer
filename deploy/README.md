# Deploy the API on the GCP A100 VM

Target: one NVIDIA A100 40 GB, Linux amd64, one GPU worker. The container uses
Python 3.12, CUDA 12.8.1, PyTorch 2.10.0 / torchvision 0.25.0, and the existing
pinned analysis/model versions. The frontend is excluded from the image.

The API and queue have been tested locally with fixture subprocesses and existing
scene results. The CUDA image and full pipeline still need a real A100 smoke test;
they have not been run on the remote VM from this workspace.

## Prepare the VM

SSH into the instance. Install Docker Engine with Compose and a compatible NVIDIA
driver. If they are missing, follow the official instructions for your VM OS:

- [GCP GPU drivers](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu)
- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Configure the NVIDIA runtime as described in that toolkit guide, then verify:

```bash
nvidia-smi
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Keep Docker's data directory on the large persistent SSD. The named volumes below
store job inputs, renders, SQLite state, results, model weights, and compiled CUDA
extensions. Check the actual filesystem behind Docker's storage rather than
assuming it uses every disk attached to the VM:

```bash
docker info --format '{{.DockerRootDir}}'
df -h /var/lib/docker
```

## Build and start

Clone your GitHub repository on the VM and enter its root. Create a private env
file with a generated key:

```bash
cp deploy/api.env.example deploy/.env
python3 - <<'PY'
from pathlib import Path
import secrets
p = Path('deploy/.env')
p.write_text(p.read_text().replace(
    'replace-with-a-random-secret-of-at-least-32-characters', secrets.token_urlsafe(32)))
p.chmod(0o600)
PY

docker compose -f deploy/compose.gcp.yaml up --build -d
docker compose -f deploy/compose.gcp.yaml logs -f api
```

The image contains the CUDA development toolchain and compiles gsplat for the
A100's `sm_80` architecture. Building does not require a GPU inside Docker's build
step. The running service verifies CUDA is available before accepting jobs.
The first real job also downloads model weights; subsequent jobs reuse them.

```bash
curl --fail http://127.0.0.1:8000/health/ready
```

The API binds to the VM's loopback interface through Docker. Test uploads on the
VM with the calls in [API.md](../API.md), after loading the key into your shell:

```bash
set -a
. deploy/.env
set +a
```

Start with a known small scene and `quality=low` to validate the whole CUDA path,
then test `quality=high` and semantic enrichment. Inspect its `worker.log` if the
job fails. A model stage has a configurable six-hour timeout. Model processes run
one at a time and exit between analysis and semantic enrichment to release memory.

To test from your laptop before public hosting, forward the port through SSH:

```bash
ssh -N -L 8000:127.0.0.1:8000 YOUR_USER@VM_IP
```

Use `http://127.0.0.1:8000/docs` locally and authorize with the same API key.

## Optional public HTTPS endpoint

Point a DNS name such as `api.example.com` at the VM's static external IP. Allow
inbound TCP 80 and 443 for this VM in GCP's firewall. Keep port 8000 private.
Add `API_DOMAIN=api.example.com` to `deploy/.env`, then start the included Caddy
reverse proxy:

```bash
docker compose --env-file deploy/.env \
  -f deploy/compose.gcp.yaml -f deploy/compose.public.yaml up -d
```

Caddy obtains and renews HTTPS certificates for that domain. Call the API at
`https://api.example.com`; bearer authentication still applies. Use the API key
from your application backend rather than shipping it in a public frontend.
The Caddy and API volumes persist across normal Compose restarts/rebuilds.

## Update and inspect

```bash
git pull --ff-only
docker compose -f deploy/compose.gcp.yaml up --build -d
docker compose -f deploy/compose.gcp.yaml ps
docker compose -f deploy/compose.gcp.yaml logs --tail 100 api
```

If using HTTPS, keep both Compose files and `--env-file deploy/.env` in update
commands. Do not remove the named volumes when updating. Queue state and results
are under `/data` inside the API container; per-job diagnostics are
`/data/jobs/JOB_ID/worker.log`. Completed jobs are retained; plan disk cleanup and
backups for your usage. Keep one API instance for this GPU and data volume.

Configuration in `deploy/.env`:

| Variable | Default / purpose |
| --- | --- |
| `SPLAT_API_KEY` | Required generated bearer secret; startup rejects the placeholder |
| `SPLAT_MAX_UPLOAD_BYTES` | `2147483648` (2 GiB) |
| `SPLAT_MAX_PENDING_JOBS` | `8` running plus queued jobs |
| `SPLAT_STAGE_TIMEOUT_SECONDS` | `21600` per model subprocess |
| `SPLAT_CORS_ORIGINS` | Empty; optional comma-separated browser origins |
| `API_DOMAIN` | Only required for the optional HTTPS proxy |

Implementation references: [Docker GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/),
[official PyTorch wheel combinations](https://pytorch.org/get-started/previous-versions/),
[FastAPI worker deployment](https://fastapi.tiangolo.com/deployment/server-workers/).
