# Splat Object Analyzer API v1

The API runs the existing pipeline without serving the viewer. Upload one Gaussian
splat and receive an asynchronous job. By default the job includes automatic
object discovery, segmentation, positions, boxes, richer descriptions, functional
groups, scene context, hero subjects, relationships, and suggested still views.

Interactive documentation: `/docs`. Machine-readable contract: `/openapi.json`.
A generated copy is checked in as `openapi.json` for frontend/client development.
All data endpoints require `Authorization: Bearer YOUR_API_KEY`.

## Start locally

After the normal GPU setup (`./setup.sh mac` or `./setup.sh cuda`), install the
API dependencies into the same environment:

```bash
uv pip install --python .venv/bin/python -r requirements-api.txt
export SPLAT_API_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
./serve-api.sh
```

The default address is `http://127.0.0.1:8000`. Keep the generated key for client
requests. `SPLAT_DATA_DIR` defaults to `outputs/api/`. For GCP and HTTPS, see
[the deployment guide](deploy/README.md).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/analyses?filename=room.spz` | Upload a file and queue the full analysis; returns 202 |
| GET | `/v1/jobs/{job_id}` | Poll either an analysis or a prompt-query job |
| GET | `/v1/analyses/{analysis_id}` | Fetch the structured analysis result |
| POST | `/v1/analyses/{analysis_id}/queries` | Queue a prompt such as “show off my desk setup”; returns 202 |
| GET | `/v1/queries/{query_id}` | Fetch selected object/group IDs and suggested views |
| GET | `/v1/analyses/{analysis_id}/frames/{frame_index}` | Fetch a supporting rendered PNG |
| GET | `/v1/analyses/{analysis_id}/source` | Download the original uploaded file |
| GET | `/health/live` | Public HTTP liveness check |
| GET | `/health/ready` | Public queue-worker readiness check |

Analyses and prompt queries share one persistent FIFO queue and one GPU worker.
Do not run multiple Uvicorn workers. The service rejects a second process using
the same data directory. Separate data directories are not a GPU isolation mechanism.

## Upload, poll, fetch

The upload body is the file's raw bytes, **not multipart/form-data**. Supported
formats are standard 3DGS PLY and SPZ v1–v3. The default upload limit is 2 GiB.
`quality=high` is the default; `low` and `medium` are also accepted. Set
`understanding=false` to omit the slower description/grouping stage and return
the object inventory and geometry. Such an analysis cannot accept prompt queries.

```bash
export API_URL=http://127.0.0.1:8000

curl --fail-with-body "$API_URL/v1/analyses?filename=room.spz&quality=high&understanding=true" \
  -H "Authorization: Bearer $SPLAT_API_KEY" \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @/path/to/room.spz
```

The 202 response is a `Job`. The analysis ID is the job's `id`. Both `status_url`
and `result_url` are relative to your API base URL. `Location` points to the status
endpoint and `Retry-After` suggests the initial polling interval in seconds.

```json
{
  "id": "11111111111111111111111111111111",
  "kind": "analysis",
  "analysis_id": "11111111111111111111111111111111",
  "status": "queued",
  "created_at": "2026-10-28T00:00:00.000Z",
  "started_at": null,
  "finished_at": null,
  "progress": {"stage": "Queued", "completed": null, "total": null},
  "error": null,
  "status_url": "/v1/jobs/11111111111111111111111111111111",
  "result_url": "/v1/analyses/11111111111111111111111111111111"
}
```

Use the returned ID in subsequent calls:

```bash
curl --fail-with-body "$API_URL/v1/jobs/$ANALYSIS_ID" \
  -H "Authorization: Bearer $SPLAT_API_KEY"

curl --fail-with-body "$API_URL/v1/analyses/$ANALYSIS_ID" \
  -H "Authorization: Bearer $SPLAT_API_KEY"
```

Job states are `queued`, `running`, `succeeded`, and `failed`. Stop polling at a
terminal state. Progress counters apply to the current stage, not to an overall
percentage. A result request before success returns 409. A failed job includes a
structured error; operator diagnostics are in that job's `worker.log` on disk.

The result includes:

| Field | Contents |
| --- | --- |
| `schema_version` | Public result format, currently `1.0` |
| `source` | Original filename, format, bytes, SHA-256, download URL |
| `coordinates` | Geometry/camera spaces, units, quaternion order, transform |
| `rendered_view_count` | Number of accepted rendered analysis views |
| `understanding_included` | Whether semantic enrichment was requested and completed |
| `scene` | Inferred space type, description, confidence; null without enrichment |
| `objects` | IDs, labels, categories, descriptions, review status, geometry, confidence, parent/child links, evidence URLs |
| `groups` | Functional areas such as a desk setup, with member object IDs |
| `relations` | Geometry-derived part/content and nearby relationships |
| `hero_object_ids` | Suggested primary subjects for a scene tour |
| `default_selection`, `suggested_views` | Default tour subjects and up to three saved camera views |
| `limitations` | Relevant model/geometry limitations |

`objects[].status` can be `supported`, `uncertain`, `rejected`, or `unreviewed`
(enrichment omitted). Rejected records are retained for inspection; clients can
filter them. Confidence is a detector score, not a calibrated probability.

## Prompt queries

```bash
curl --fail-with-body "$API_URL/v1/analyses/$ANALYSIS_ID/queries" \
  -H "Authorization: Bearer $SPLAT_API_KEY" \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"Show off my desk setup"}'
```

Poll the returned job's `status_url`, then fetch its `result_url`. The result
contains the original prompt, `selection.object_ids`, `selection.group_ids`, a
message, suggested views with image URLs/cameras, and `unshown_object_ids`.
Resolve those IDs against the analysis result. A no-match response succeeds with
empty subject and view lists. Prompt matching reuses the saved scene index;
it does not rerun detection or object descriptions.

All image and source URLs also require the bearer key. An HTML `<img>` cannot
attach that header automatically: proxy the image through your own application
backend, or fetch it with authorization and display a Blob URL.

## Coordinates

Object `position`, `scale`, `bounding_box`, and `oriented_box` remain in the
original **source** space. Units are source-scene units; metric scale is not
verified. Top-level `position`/`scale` describe the axis-aligned estimate. An
`oriented_box` has its own center, full local-axis extents, and XYZW quaternion.
Use those fields together when drawing a rotated box.

Suggested cameras are in **viewer** space, with vertical field of view in degrees.
The supplied 4×4 `source_to_viewer` matrix maps `[x,y,z]` to `[x,-y,-z]`. To use
these cameras, apply that transform to the splat and its annotations, composing
the box quaternion with the same rotation. Do not transform the cameras a second
time. Alternatively transform the cameras back into source space for your renderer.

Suggested views are existing analysis renders with depth visibility checks.
They are not camera animations or collision-checked paths.

## Operations and errors

Jobs/results and SQLite queue state survive process/container restarts when the
data volume persists. Interrupted jobs are queued again. Completed geometry is
reused when only semantic enrichment was interrupted; saved caption batches are
also reused. An interrupted earlier analysis stage is rerun. Completed results
are retained until an operator removes them; this version has no retention or
delete endpoint. Monitor disk space on the VM.

Errors use `{"error":{"code":"...","message":"...","details":[]}}`.
401 means missing/invalid key; 400/415/422 mean invalid input; 413 means file too
large; 429 means the bounded queue is full; 404 means missing resource; 409 means
the result is not ready or semantic enrichment was omitted. Retry queued work by
polling, rather than uploading the same file repeatedly.

The API key identifies one trusted deployment, not separate end users. Keep it in
your application backend; add your own user authentication/authorization there.
`SPLAT_CORS_ORIGINS` can allow explicit browser origins for development. It is
empty by default. Model inference stays on the host; the first job downloads
the public pretrained weights into the persistent cache.

Run API lifecycle checks without downloading models:

```bash
uv pip install --python .venv/bin/python -r requirements-api-test.txt
.venv/bin/python check-api.py
```
