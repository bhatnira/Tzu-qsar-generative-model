#!/usr/bin/env bash
# compile.sh – Build the QSAR presentation PDF using XeLaTeX
# Usage: bash compile.sh
set -e
cd "$(dirname "$0")"
echo "=== Pass 1 ==="
xelatex -interaction=batchmode qsar_presentation.tex
echo "=== Pass 2 (cross-references) ==="
xelatex -interaction=batchmode qsar_presentation.tex
echo ""
echo "Done! → qsar_presentation.pdf ($(pdfinfo qsar_presentation.pdf | grep Pages))"
