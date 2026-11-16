"""Build the end-to-end pipeline diagram of the final report (Figure 6.1).

    python docs/diagrams/pipeline_overview.py

Writes, from the one layout below:
  docs/diagrams/pipeline_overview.drawio       editable source (open in draw.io)
  docs/diagrams/pipeline_overview.svg          vector render
  docs/final_report/figures/pipeline_overview.pdf  vector render used by the report
Coordinates are draw.io pixels on a 1000 x 520 canvas.
"""
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
W, H = 1000, 520

# Role colors (fill, stroke): the same blue, orange and purple as the intervention notebooks.
STYLES = {
    "solver": ("#dbe8f6", "#2a78d6"),
    "critic": ("#fde7d9", "#eb6834"),
    "claude": ("#ebe3f5", "#8f5ccf"),
    "step": ("#ffffff", "#6b6b6b"),
}
BAND_FILL, INK = "#f2f2ef", "#222222"

BANDS = [  # (title, x, y, w, h)
    ("1  Build the interaction dataset", 0, 0, W, 195),
    ("2  Find where the decision is represented (fit on train, chosen on validation)", 0, 215, W, 130),
    ("3  Test causality", 0, 365, W, 155),
]

# id: (style, x, y, w, h, [bold title, line, ...])
BOXES = {
    "questions": ("step", 15, 46, 125, 116, ["Questions", "MuSiQue", "+ synthetic", "(MuSiQue format)"]),
    "solver1": ("solver", 165, 46, 160, 116, ["Solver · Gemma 3 4B", "Attempt 1:", "answers from the", "paragraphs (A1)"]),
    "critic": ("critic", 350, 46, 180, 116, ["Critic · Qwen3 4B", "1. answers without", "seeing A1", "2. compares with A1,", "recommends C1"]),
    "solver2": ("solver", 555, 46, 185, 116, ["Solver · Gemma 3 4B", "Attempt 2: chooses", "between A1 and C1,", "shown as two candidates", "(random order)"]),
    "activations": ("step", 765, 34, 220, 66, ["Activations", "residual stream at the final", "prompt token, nine layers"]),
    "labels": ("claude", 765, 108, 220, 66, ["Claude Sonnet 4.5 judge", "labels each A2: Adoption,", "Retention, Agreement or Out"]),
    "layers": ("step", 715, 245, 270, 90, ["Layer-selection probe", "Adoption vs Retention", "on each layer: layer 25 is best"]),
    "saes": ("step", 445, 245, 240, 90, ["SAEs at layer 25", "custom: L1, TopK", "pretrained: Gemma Scope 2,", "decoder-adapted"]),
    "features": ("step", 15, 245, 400, 90, ["Feature-selection probe", "10 Adoption + 10 Retention features per SAE,", "plus 100 random feature sets", "matched on firing frequency"]),
    "edit": ("solver", 15, 393, 400, 116, ["Replay attempt 2 with an edit", "held-out intervention episodes;", "layer 25, final prompt token;", "targeted vs random features, same size (|Δ| = 0.2 |x|)", "controls: SAE reconstruction, raw activations"]),
    "relabel": ("claude", 445, 393, 240, 116, ["Claude relabels A2", "same judge and prompt", "as the stored labels"]),
    "result": ("step", 715, 393, 270, 116, ["Flip rate", "targeted − random,", "9 prespecified tests,", "Holm correction"]),
}

# (source, target, (exit x, exit y), (entry x, entry y), label); points are fractions of each box.
EDGES = [
    ("questions", "solver1", (1, 0.5), (0, 0.5), ""),
    ("solver1", "critic", (1, 0.5), (0, 0.5), ""),
    ("critic", "solver2", (1, 0.5), (0, 0.5), ""),
    ("solver2", "activations", (1, 0.181), (0, 0.5), ""),
    ("solver2", "labels", (1, 0.819), (0, 0.5), ""),
    ("labels", "layers", (0.386, 1), (0.5, 0), "activations and labels"),
    ("layers", "saes", (0, 0.5), (1, 0.5), ""),
    ("saes", "features", (0, 0.5), (1, 0.5), ""),
    ("features", "edit", (0.5, 1), (0.5, 0), "feature sets"),
    ("edit", "relabel", (1, 0.5), (0, 0.5), ""),
    ("relabel", "result", (1, 0.5), (0, 0.5), ""),
]


def point(box_id, fx, fy):
    _, x, y, w, h, _ = BOXES[box_id]
    return x + fx * w, y + fy * h


def render(paths):
    inch = 6.5 / W
    fig = plt.figure(figsize=(W * inch, H * inch))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for title, x, y, w, h in BANDS:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                                    facecolor=BAND_FILL, edgecolor="none"))
        ax.text(x + 10, y + 15, title, fontsize=7.5, fontweight="bold", color=INK, va="center")
    for style, x, y, w, h, lines in BOXES.values():
        fill, stroke = STYLES[style]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                                    facecolor=fill, edgecolor=stroke, linewidth=0.9))
        step = 18
        top = y + h / 2 - step * (len(lines) - 1) / 2
        for i, line in enumerate(lines):
            ax.text(x + w / 2, top + i * step, line, ha="center", va="center", color=INK,
                    fontsize=6.2, fontweight="bold" if i == 0 else "normal")
    for source, target, exit_point, entry_point, label in EDGES:
        start, end = point(source, *exit_point), point(target, *entry_point)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                     color="#444444", linewidth=0.9, shrinkA=0, shrinkB=0))
        if label:
            ax.text(start[0] - 6, (start[1] + end[1]) / 2, label, fontsize=6, color="#444444",
                    style="italic", va="center", ha="right")
    for path in paths:
        fig.savefig(path, metadata={"Date": None} if path.suffix == ".pdf" else None)
    plt.close(fig)


def drawio(path):
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    for i, (title, x, y, w, h) in enumerate(BANDS):
        style = ("rounded=1;arcSize=3;absoluteArcSize=0;fillColor=%s;strokeColor=none;align=left;"
                 "verticalAlign=top;spacingLeft=10;spacingTop=2;fontStyle=1;fontSize=12;html=1;" % BAND_FILL)
        cells.append(f'<mxCell id="band{i + 1}" value="{escape(escape(title))}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
    for box_id, (style_name, x, y, w, h, lines) in BOXES.items():
        fill, stroke = STYLES[style_name]
        html = "<b>" + escape(lines[0]) + "</b><br>" + "<br>".join(escape(line) for line in lines[1:])
        style = f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=10;"
        cells.append(f'<mxCell id="{box_id}" value="{escape(html)}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
    for i, (source, target, (ex, ey), (nx, ny), label) in enumerate(EDGES):
        style = (f"endArrow=block;endFill=1;html=1;rounded=0;strokeColor=#444444;exitX={ex};exitY={ey};"
                 f"exitDx=0;exitDy=0;entryX={nx};entryY={ny};entryDx=0;entryDy=0;fontStyle=2;fontSize=10;"
                 "labelBackgroundColor=none;align=right;spacingRight=6;")
        cells.append(f'<mxCell id="edge{i + 1}" value="{escape(label)}" style="{style}" edge="1" parent="1" '
                     f'source="{source}" target="{target}"><mxGeometry relative="1" as="geometry" /></mxCell>')
    body = "\n        ".join(cells)
    path.write_text(
        '<mxfile host="app.diagrams.net">\n'
        '  <diagram name="Pipeline overview" id="pipelineOverview">\n'
        f'    <mxGraphModel dx="1200" dy="700" grid="1" gridSize="5" guides="1" tooltips="1" connect="1" '
        f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">\n'
        f'      <root>\n        {body}\n      </root>\n'
        '    </mxGraphModel>\n  </diagram>\n</mxfile>\n')


if __name__ == "__main__":
    drawio(ROOT / "docs/diagrams/pipeline_overview.drawio")
    render([ROOT / "docs/diagrams/pipeline_overview.svg", ROOT / "docs/final_report/figures/pipeline_overview.pdf"])
    print("wrote pipeline_overview.drawio, .svg and figures/pipeline_overview.pdf")
