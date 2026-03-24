#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p export

pandoc technical_note.md \
  --citeproc \
  --pdf-engine=xelatex \
  --output export/shape-budget-technical-note.pdf
