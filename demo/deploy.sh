#!/usr/bin/env bash
# Uploads the demo to a Hugging Face Space.
#
# Usage (from anywhere):
#     hf auth login                                  # once
#     demo/deploy.sh <user>/<space-name> --static    # browser-only page (free on every account)
#     demo/deploy.sh <user>/<space-name>             # Gradio server (needs a paid plan)
#
# The Space is created if it does not exist, and its contents are REPLACED by
# the build below. Both builds contain pipeline.py, presets.yaml, the sample
# thumbnails, core/ and the two scripts it imports from, plus either
# index.html (--static, `sdk: static`) or app.py and requirements.txt
# (`sdk: gradio`), and README.md from demo/ with the matching header.
set -euo pipefail

SPACE_ID=${1:?usage: demo/deploy.sh <user>/<space-name> [--static]}
MODE=${2:-gradio}
case "$MODE" in
    --static) SDK=static ;;
    gradio)   SDK=gradio ;;
    *) echo "unknown option: $MODE (expected --static)" >&2; exit 1 ;;
esac

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT

# Use an installed hf CLI, or run it from a temporary pixi environment.
if command -v hf >/dev/null; then
    hf() { command hf "$@"; }
else
    hf() { pixi exec --spec huggingface_hub hf "$@"; }
fi

cp "$ROOT"/demo/{pipeline.py,presets.yaml} "$ROOT"/{remove_refraction.py,visualise_refraction.py} "$BUILD"/
cp -r "$ROOT"/demo/samples "$BUILD"/
mkdir -p "$BUILD"/core
cp "$ROOT"/core/*.py "$BUILD"/core/

if [ "$SDK" = static ]; then
    cp "$ROOT"/demo/index.html "$BUILD"/
    # Static Spaces serve index.html; drop the Gradio-only header fields.
    sed -e 's/^sdk: gradio$/sdk: static/' -e '/^sdk_version:/d' -e '/^app_file:/d' \
        "$ROOT"/demo/README.md > "$BUILD"/README.md
else
    cp "$ROOT"/demo/{app.py,requirements.txt,README.md} "$BUILD"/
fi

hf repos create "$SPACE_ID" --repo-type space --space-sdk "$SDK" --exist-ok
# --delete "*" makes the Space mirror the build (stale files are removed; .gitattributes is kept).
hf upload "$SPACE_ID" "$BUILD" . --repo-type space --delete "*" --commit-message "Update demo ($SDK)"
