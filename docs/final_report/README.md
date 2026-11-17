# Final report

This directory contains the LaTeX source for the capstone final report. The
document structure follows the agreed final-report outline. Substantive prose,
figures, tables, and verified results will be added in later drafting stages.

## Build

From the repository root:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error final_report/main.tex
```

The rendered document will be written to `final_report/main.pdf`.

Clean generated build files with:

```bash
latexmk -cd -C final_report/main.tex
```

## Organization

- `sections/`: front matter and report chapters
- `appendices/`: supporting material
- `equations/`: reusable LaTeX equations included by report sections
- `algorithms/`: reusable LaTeX pseudocode included by methodology sections
- `figures/`: report-ready figures
- `build_figures/`: LaTeX-compatible derivatives generated from the SVG sources
- `tables/`: generated or hand-authored LaTeX tables
- `references.bib`: consolidated bibliography

The title, authors (alphabetical by surname), course, advisor, and submission
date are set by the title-page commands near the top of `main.tex`; the PDF
metadata is set in the `\hypersetup` block below them.

The authoritative figure sources in `figures/` are SVG files only. Because
pdfLaTeX cannot include SVG directly, `build_figures/` contains generated PDF
derivatives used during compilation; these are not the editable source figures.
