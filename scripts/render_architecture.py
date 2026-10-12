#!/usr/bin/env python3
"""Render the documented recurrent manifold training architecture.

Run from any directory with matplotlib installed.  The output is a repository-
owned scientific diagram, not a model result.  SVG text remains selectable.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
W, H = 2400, 2110
C = {
    "navy": "#20324F", "muted": "#7184A4", "line": "#D7E0EF",
    "blue": "#3C73C7", "blue_bg": "#ECF3FE",
    "purple": "#8966C2", "purple_bg": "#F1ECF9",
    "teal": "#209687", "teal_bg": "#EBF6F3",
    "orange": "#CD7838", "orange_bg": "#FFF3E9",
    "gray": "#8193AF", "gray_bg": "#F6F8FB",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(16, H / 150), dpi=150)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")


def txt(x, y, s, size=12, color="navy", weight="normal", ha="left", va="center"):
    return ax.text(x, y, s, fontsize=size, color=C.get(color, color),
                   weight=weight, ha=ha, va=va, linespacing=1.32)


def box(x, y, w, h, title, lines=(), tone="gray", title_size=13, body_size=11.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=14",
                 edgecolor=C[tone], facecolor=C[f"{tone}_bg"], linewidth=1.35))
    if lines:
        ty = y + (32 if h < 130 else 37)
        txt(x + w/2, ty, title, title_size, weight="bold", ha="center")
        start = ty + 37
        for i, line in enumerate(lines):
            txt(x + w/2, start + i*32, line, body_size, "muted", ha="center")
    else:
        txt(x + w/2, y+h/2, title, title_size, weight="bold", ha="center")


def arrow(points, tone="blue", lw=1.6, dashed=False, head=True):
    # Draw elbow segments separately, with an arrowhead on the final segment.
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=C[tone], linewidth=lw,
                linestyle=(0, (4, 4)) if dashed else "-", solid_capstyle="round")
    p, q = points[-2:]
    ax.add_patch(FancyArrowPatch(p, q,
                 arrowstyle="-|>" if head else "-", mutation_scale=13,
                 linewidth=lw, color=C[tone],
                 linestyle=(0, (4, 4)) if dashed else "-",
                 shrinkA=0, shrinkB=0))


def section(y, number, name, tone, note=""):
    txt(80, y, f"{number}   {name}", 15, tone, "bold")
    if note:
        txt(W-80, y, note, 11.3, "muted", ha="right")


def rule(y):
    ax.plot([80, W-80], [y, y], color=C["line"], linewidth=0.9)


# Header: the visual vocabulary follows the supplied Climate Manifold figure.
txt(80, 66, "NeuralGCM + Recurrent Climate Manifold", 27, weight="bold")
txt(80, 120, "Autoregressive latent dynamics + separate spatial memory + stochastic temperature closure", 13.6, "muted")
txt(W-80, 165, "Conditional sequence model · not reinforcement learning", 11.3, "muted", ha="right")
rule(186)

# 01 — A strictly observed initialization, preserving both memory streams.
section(224, "01", "OBSERVED INITIALIZATION", "teal", "No future observations are used to initialize the forecast")
box(80, 264, 400, 103, "Observed source fields", ["past t < 0 + exact origin t = 0"], "gray")
box(550, 264, 415, 103, "Frozen spatial encoder E_s", ["same normalization and weights"], "blue")
box(1035, 264, 485, 103, "Past latents + origin latent", ["z_-H → ... → z_-1  ;  z_0"], "purple")
box(1590, 264, 730, 103, "Spatial initialization", ["past sequence → M_s,0    |    origin → z_0"], "teal")
for a, b in [(480, 550), (965, 1035), (1520, 1590)]:
    arrow([(a+8, 315), (b-12, 315)], "teal")
box(80, 409, 400, 103, "Observed physical history", ["past fields + origin fields"], "gray", title_size=12)
box(550, 409, 970, 103, "Frozen NeuralGCM encode + closure warmup", ["initialize X_0, closure memory m_c,0 and OU state Z_0"], "blue")
box(1590, 409, 730, 103, "Expanded forecast state S_0", ["X_0, z_0, M_s,0, m_c,0, Z_0"], "teal")
arrow([(488, 460), (538, 460)], "teal")
arrow([(1528, 460), (1578, 460)], "teal")
arrow([(1955, 375), (1955, 399)], "teal")
txt(1955, 537, "Spatial initialization preserves the closure warmup", 10.5, "teal", ha="center")

# 02 — Show the real carried state and all four synchronous update branches.
section(590, "02", "RECURRENT FORECAST", "blue", "Repeat every internal time step h · frozen-step Gaussian transition")
ax.add_patch(FancyBboxPatch((60, 626), 2280, 702,
             boxstyle="round,pad=0,rounding_size=19", facecolor="white",
             edgecolor=C["line"], linewidth=1.0))
box(80, 673, 370, 520, "", [], "gray", title_size=14)
# Custom state-card hierarchy, with explicit carried components.
ax.add_patch(FancyBboxPatch((82, 675), 366, 516,
             boxstyle="round,pad=0,rounding_size=12", facecolor=C["gray_bg"],
             edgecolor="none"))
txt(265, 717, "Current state S_t", 14, weight="bold", ha="center")
for y, title, detail in [
    (793, "X_t", "physical model state"),
    (879, "z_t", "current spatial latent"),
    (965, "M_s,t", "separate spatial memory"),
    (1051, "m_c,t", "closure memory"),
    (1137, "Z_t", "OU stochastic state"),
]:
    txt(265, y, title, 14, weight="bold", ha="center")
    txt(265, y+31, detail, 10.6, "muted", ha="center")

box(540, 661, 770, 104, "Frozen NeuralGCM advance", ["X_t → base physical state X_base,t+h"], "blue")
box(540, 804, 770, 137, "Trainable conditional latent transition", ["p_θ(z_t+h | z_t, T_t, M_s,t)", "Sample new z_t+h and carry it forward"], "purple")
box(540, 980, 770, 103, "Spatial memory update", ["M_s,t+h = a_s M_s,t + (1 − a_s) z_t"], "purple")
box(540, 1122, 770, 138, "Trainable memory + OU closure", ["features [T_t, z_t] + closure memory m_c,t + OU Z_t", "integrate ΔT ; update m_c,t+h and Z_t+h"], "blue", body_size=10.7)
for sy, ty, tone in [(793, 713, "blue"), (879, 872, "purple"), (965, 1031, "purple"), (1137, 1191, "blue")]:
    elbow = 487
    arrow([(458, sy), (elbow, sy), (elbow, ty), (528, ty)], tone)

box(1850, 673, 470, 520, "", [], "blue", title_size=13)
ax.add_patch(FancyBboxPatch((1852, 675), 466, 516,
             boxstyle="round,pad=0,rounding_size=12", facecolor=C["blue_bg"], edgecolor="none"))
txt(2085, 717, "Corrected next state S_t+h", 13, weight="bold", ha="center")
txt(2085, 798, "X_t+h = X_base,t+h + ΔT", 12, weight="bold", ha="center")
txt(2085, 835, "ΔT updates temperature only", 10.6, "muted", ha="center")
for y, line in [(908, "z_t+h  ·  M_s,t+h"), (961, "m_c,t+h  ·  Z_t+h")]:
    txt(2085, y, line, 13.5, weight="bold", ha="center")
ax.plot([1900, 2270], [1010, 1010], color=C["line"], linewidth=0.8)
txt(2085, 1051, "Latent trajectory", 11.5, "purple", "bold", ha="center")
txt(2085, 1104, "z_0 → z_1 → z_2 → ...", 15, "purple", ha="center")
txt(2085, 1153, "Each predicted z becomes the next input", 9.7, "muted", ha="center")

arrow([(1318, 713), (1838, 713)], "blue")
txt(1570, 690, "base physical advance", 10.5, "muted", ha="center")
arrow([(1318, 872), (1838, 872)], "purple")
txt(1570, 846, "new sampled latent z_t+h", 10.5, "purple", ha="center")
arrow([(1318, 1031), (1838, 1031)], "purple")
txt(1570, 1004, "accumulated source memory", 10.5, "purple", ha="center")
arrow([(1318, 1191), (1570, 1191), (1570, 1138), (1838, 1138)], "blue")
txt(1563, 1226, "ΔT, closure memory and OU state", 10.2, "muted", ha="center")
# A single explicit full-state feedback line is distinct from the branch arrows.
arrow([(2085, 1201), (2085, 1290), (265, 1290), (265, 1203)], "blue", lw=1.8)
txt(955, 1310, "Carry the entire corrected state into the next step", 10.7, "blue", ha="center")

# 03 — Training labels are only compared at observation times; no arrow feeds them to S_t.
section(1380, "03", "JOINT SEQUENCE TRAINING", "orange", "Read forecast outputs at observation times · no teacher forcing after origin")
box(80, 1433, 640, 123, "Frozen NeuralGCM decoder", ["corrected forecast states → predicted future fields"], "blue", body_size=10.7)
box(880, 1433, 650, 123, "Physical forecast loss  L_fields", ["fair CRPS + predictive-mean MSE"], "orange")
box(1770, 1433, 550, 123, "Future observed fields", ["training targets only"], "gray")
arrow([(728, 1494), (868, 1494)], "blue")
arrow([(1758, 1494), (1542, 1494)], "gray", dashed=True)
txt(1650, 1464, "labels only", 10.5, "muted", ha="center")

box(80, 1610, 640, 141, "Predicted latent sequence", ["carried z at each observation time", "z_1 → z_2 → ... → z_K"], "purple")
box(880, 1610, 650, 141, "Latent forecast loss  L_latent", ["fair CRPS + predictive-mean MSE", "scored against the observed latent sequence"], "orange", body_size=10.8)
box(1770, 1610, 550, 141, "Future source observations", ["frozen E_s → future latent labels", "same E_s as initialization"], "teal", body_size=10.7)
arrow([(728, 1680), (868, 1680)], "purple")
arrow([(1758, 1680), (1542, 1680)], "gray", dashed=True)
txt(1650, 1650, "labels only", 10.5, "muted", ha="center")

box(880, 1840, 650, 105, "Joint objective", ["L = L_fields + λ_z L_latent"], "orange")
box(80, 1840, 640, 105, "One joint optimizer step", ["update closure + latent-transition parameters"], "gray", body_size=10.7)
arrow([(1205, 1759), (1205, 1828)], "orange")
arrow([(1538, 1530), (1630, 1530), (1630, 1667)], "orange", head=False)
arrow([(1630, 1693), (1630, 1892), (1542, 1892)], "orange")
arrow([(868, 1892), (732, 1892)], "orange")
txt(1770, 1825, "Trainable", 11.8, "orange", "bold")
txt(1770, 1870, "closure + latent transition", 11.5, "muted")
txt(1770, 1915, "NeuralGCM and E_s stay frozen", 11.1, "muted")

rule(1990)
txt(80, 2028, "Solid arrows: model / loss flow     Dashed arrows: training labels only     T_t: thermal features of X_t", 10.6, "muted")
txt(80, 2070, "Configurable defaults: closure τ = 6 / 24 / 120 h · spatial τ = 24 / 120 / 720 h · latent relaxation = 24 h", 10.5, "muted")
txt(W-80, 2070, "Architecture, not a forecast-skill claim", 10.5, "muted", ha="right")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("svg", "png"):
    path = OUT / f"recurrent_manifold_training.{ext}"
    fig.savefig(path, dpi=150, metadata={"Creator": "Neural-GCM-feat-stochastic"} if ext == "svg" else None)
    print(path)
plt.close(fig)
