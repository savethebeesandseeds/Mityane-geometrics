#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build
export TEXINPUTS="$PWD/vendor/IEEEtran//:${TEXINPUTS:-}"
export BSTINPUTS="$PWD/vendor/IEEEtran/bibtex//:${BSTINPUTS:-}"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -latexoption=-no-shell-escape -outdir=build representation-encoder.tex
qpdf --check build/representation-encoder.pdf
pdfinfo build/representation-encoder.pdf
