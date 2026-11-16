"""Build the Solver-Critic-Validator diagram of one episode (final report, Section 6.4).

    python docs/diagrams/agent_architecture.py

Writes, from the one layout below:
  docs/diagrams/agent_architecture.drawio          editable source (open in draw.io)
  docs/diagrams/agent_architecture.svg             vector render
  docs/final_report/figures/agent_architecture.pdf vector render used by the report
Coordinates are draw.io pixels on a 1000 x 370 canvas.
"""
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
W, H = 1000, 370

# Role colors (fill, stroke): the same blue and orange as the intervention notebooks.
STYLES = {
    "solver": ("#dbe8f6", "#2a78d6", "solid"),
    "critic": ("#fde7d9", "#eb6834", "solid"),
    "validator": ("#e3f1ea", "#1b8a5a", "solid"),
    "data": ("#ffffff", "#6b6b6b", "solid"),
    "record": ("#ffffff", "#6b6b6b", "dashed"),
}
INK, ARROW = "#222222", "#444444"

# id: (style, x, y, w, h, [bold title, line, ...])
BOXES = {
    "inputs": ("data", 15, 120, 145, 120, ["Inputs", "question and", "20 paragraphs,", "in every Solver", "and Critic prompt"]),
    "solver1": ("solver", 195, 50, 200, 80, ["Solver · Gemma 3 4B", "Attempt 1: answers", "from the paragraphs"]),
    "blind": ("critic", 195, 230, 200, 80, ["Critic · Qwen3 4B", "Step 1: answers blind,", "without seeing A1"]),
    "compare": ("critic", 435, 135, 200, 90, ["Critic · Qwen3 4B", "Step 2: compares its", "answer with A1; returns", "verdict, C1, explanation"]),
    "solver2": ("solver", 675, 135, 190, 90, ["Solver · Gemma 3 4B", "Attempt 2: chooses", "between A1 and C1,", "shown as two candidates", "in random order"]),
    "activations": ("record", 880, 135, 110, 90, ["Activations", "9 layers, at the", "final prompt", "token of each", "Solver attempt"]),
    "validator": ("validator", 675, 270, 190, 80, ["Validator · Gemma 3 4B", "Does A2 match the gold", "answer? YES or NO"]),
    "gold": ("data", 880, 270, 110, 80, ["Gold answer", "and aliases", "(seen only by", "the Validator)"]),
}

# (source, target, exit (fx, fy), entry (fx, fy), waypoints, label, label position, dashed)
EDGES = [
    ("inputs", "solver1", (1, 0.25), (0, 0.5), [(178, 150), (178, 90)], "", None, False),
    ("inputs", "blind", (1, 0.75), (0, 0.5), [(178, 210), (178, 270)], "", None, False),
    ("solver1", "compare", (1, 0.75), (0.5, 0), [(535, 110)], "A1", (545, 100), False),
    ("solver1", "solver2", (0.5, 0), (0.5, 0), [(295, 25), (770, 25)], "A1", (780, 40), False),
    ("blind", "compare", (1, 0.5), (0.5, 1), [(535, 270)], "its own answer", (545, 255), False),
    ("compare", "solver2", (1, 0.5), (0, 0.5), [], "C1", (645, 168), False),
    ("solver2", "validator", (0.5, 1), (0.5, 0), [], "A2", (780, 248), False),
    ("solver2", "activations", (1, 0.5), (0, 0.5), [], "", None, True),
    ("gold", "validator", (0, 0.5), (1, 0.5), [], "", None, False),
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
    for style, x, y, w, h, lines in BOXES.values():
        fill, stroke, dash = STYLES[style]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                                    facecolor=fill, edgecolor=stroke, linewidth=0.9, linestyle=dash))
        step = 18
        top = y + h / 2 - step * (len(lines) - 1) / 2
        for i, line in enumerate(lines):
            ax.text(x + w / 2, top + i * step, line, ha="center", va="center", color=INK,
                    fontsize=6.2, fontweight="bold" if i == 0 else "normal")
    for source, target, exit_point, entry_point, waypoints, label, label_at, dashed in EDGES:
        route = [point(source, *exit_point), *waypoints, point(target, *entry_point)]
        style = (0, (3, 2)) if dashed else "solid"
        for start, end in zip(route[:-2], route[1:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=ARROW, linewidth=0.9, linestyle=style)
        ax.add_patch(FancyArrowPatch(route[-2], route[-1], arrowstyle="-|>", mutation_scale=7,
                                     color=ARROW, linewidth=0.9, linestyle=style, shrinkA=0, shrinkB=0))
        if label:
            ax.text(*label_at, label, fontsize=6, color=ARROW, style="italic", va="center", ha="left")
    for path in paths:
        fig.savefig(path, metadata={"Date": None} if path.suffix == ".pdf" else None)
    plt.close(fig)


def drawio(path):
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    for box_id, (style_name, x, y, w, h, lines) in BOXES.items():
        fill, stroke, dash = STYLES[style_name]
        html = "<b>" + escape(lines[0]) + "</b><br>" + "<br>".join(escape(line) for line in lines[1:])
        style = (f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=10;"
                 + ("dashed=1;" if dash == "dashed" else ""))
        cells.append(f'<mxCell id="{box_id}" value="{escape(html)}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
    for i, (source, target, (ex, ey), (nx, ny), waypoints, label, _, dashed) in enumerate(EDGES):
        style = (f"edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;html=1;rounded=0;strokeColor={ARROW};"
                 f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;entryX={nx};entryY={ny};entryDx=0;entryDy=0;"
                 "fontStyle=2;fontSize=10;labelBackgroundColor=#ffffff;" + ("dashed=1;" if dashed else ""))
        points = "".join(f'<mxPoint x="{x}" y="{y}" />' for x, y in waypoints)
        geometry = (f'<mxGeometry relative="1" as="geometry"><Array as="points">{points}</Array></mxGeometry>'
                    if waypoints else '<mxGeometry relative="1" as="geometry" />')
        cells.append(f'<mxCell id="edge{i + 1}" value="{escape(label)}" style="{style}" edge="1" parent="1" '
                     f'source="{source}" target="{target}">{geometry}</mxCell>')
    body = "\n        ".join(cells)
    path.write_text(
        '<mxfile host="app.diagrams.net">\n'
        '  <diagram name="Solver-Critic-Validator loop" id="agentArchitecture">\n'
        f'    <mxGraphModel dx="1200" dy="600" grid="1" gridSize="5" guides="1" tooltips="1" connect="1" '
        f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">\n'
        f'      <root>\n        {body}\n      </root>\n'
        '    </mxGraphModel>\n  </diagram>\n</mxfile>\n')


if __name__ == "__main__":
    drawio(ROOT / "docs/diagrams/agent_architecture.drawio")
    render([ROOT / "docs/diagrams/agent_architecture.svg", ROOT / "docs/final_report/figures/agent_architecture.pdf"])
    print("wrote agent_architecture.drawio, .svg and figures/agent_architecture.pdf")
