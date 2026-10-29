# LaTeX literature review / report

## Render the report

From the repository root, run:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error research_paper/Latex/main.tex
```

Alternatively, build from this directory:

```bash
cd research_paper/Latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The rendered document is written to `research_paper/Latex/main.pdf`.
