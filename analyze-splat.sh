#!/bin/sh
set -eu
if [ "$#" -lt 1 ]; then
    echo 'Usage: ./analyze-splat.sh /path/to/scene.spz [--quality low|medium|high] [--job_dir /path/to/output]' >&2
    exit 2
fi
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SCENE=$1
shift
exec "$ROOT/run.sh" --ply "$SCENE" "$@"

