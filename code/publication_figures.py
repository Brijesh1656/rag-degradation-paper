import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Load all JSON files ───────────────────────────────────────────
files = glob.glob("*.json")
data = []
for f in files:
    with open(f) as fp:
        data.extend(json.load(fp))

seen = set()
unique_data = []
for row in data:
    key = (row["sessionId"], row["turnNumber"])
    if key not in seen:
        seen.add(key)
        unique_data.append(row)
data = unique_data

# ── Sessions ──────────────────────────────────────────────────────
sessions = defaultdict(list)
for row in data:
    sessions[row["sessionId"]].append(row)

turn_data = defaultdict(list)
for row in data:
    turn_data[row["turnNumber"]].append(row)

turn_counts = [len(v) for v in sessions.values()]
turns_list = sorted(turn_data.keys())

# ── Compute stats ─────────────────────────────────────────────────
avg_tokens = [np.mean([r["contextTokenCount"] for r in turn_data[t]]) for t in turns_list]
std_tokens = [np.std([r["contextTokenCount"] for r in turn_data[t]]) for t in turns_list]

avg_sims = [np.mean([r["similarityScore"] for r in turn_data[t]]) for t in turns_list]
std_sims = [np.std([r["similarityScore"] for r in turn_data[t]]) for t in turns_list]

# Keyword drift
vectorizer = TfidfVectorizer()
pair_sims = defaultdict(list)
for sid, turns in sessions.items():
    turns_sorted = sorted(turns, key=lambda x: x["turnNumber"])
    queries = [t["subQuery"] for t in turns_sorted if t.get("subQuery")]
    if len(queries) < 2:
        continue
    try:
        vecs = vectorizer.fit_transform(queries).toarray()
        for i in range(len(vecs) - 1):
            pair = f"{i+1}-{i+2}"
            cs = cosine_similarity([vecs[i]], [vecs[i+1]])[0][0]
            pair_sims[pair].append(cs)
    except:
        pass

drift_pairs = []
drift_vals = []
for pair in ["1-2", "2-3", "3-4", "4-5"]:
    if pair_sims[pair]:
        drift_pairs.append(f"Turns {pair}")
        drift_vals.append(np.mean(pair_sims[pair]))

# Scatter data
tier1 = [r for r in data if r["retrievalTier"] == 1]
tier1_ctx = [r["contextTokenCount"] for r in tier1]
tier1_sim = [r["similarityScore"] for r in tier1]
r_val, _ = stats.pearsonr(tier1_ctx, tier1_sim) if len(tier1_ctx) > 1 else (0, 1)

# Regression line
if len(tier1_ctx) > 1:
    z = np.polyfit(tier1_ctx, tier1_sim, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(tier1_ctx), max(tier1_ctx), 100)

# Degradation
t1_mean = avg_sims[0] if avg_sims else 0
t4_mean = avg_sims[3] if len(avg_sims) > 3 else avg_sims[-1]
degradation = (t1_mean - t4_mean) / t1_mean * 100 if t1_mean > 0 else 0

# ── Confidence Estimator AUROC ────────────────────────────────────
labeled = [r for r in data if r.get("wasCorrect") is not None]
C_baseline = avg_tokens[0] if avg_tokens else 1

fpr, tpr, roc_auc = None, None, None
if len(labeled) >= 2: # Kept threshold low for testing
    from sklearn.metrics import roc_auc_score, roc_curve
    X, y = [], []
    for r in labeled:
        t, C, g = r["turnNumber"], r["contextTokenCount"], r["similarityScore"]
        alpha, beta = 0.18, 0.09
        conf = g * np.exp(-alpha * t) * (1 - beta * np.log(max(C / C_baseline, 1e-9)))
        X.append([conf])
        y.append(1 if r["wasCorrect"] else 0)

    X, y = np.array(X), np.array(y)
    if len(set(y)) == 2:
        roc_auc = roc_auc_score(y, X[:, 0])
        fpr, tpr, _ = roc_curve(y, X[:, 0])

# ── Style settings ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

BLUE   = "#378ADD"
RED    = "#E24B4A"
AMBER  = "#EF9F27"
GREEN  = "#1D9E75"
GRAY   = "#888780"

# Background box for annotations to make numbers readable over lines
label_bg = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)

# ── Summary metric cards (printed, not plotted) ───────────────────
print(f"\n  Sessions: {len(sessions)}  |  "
      f"Turn 1 sim: {t1_mean:.3f}  |  "
      f"Turn 4 sim: {t4_mean:.3f}  |  "
      f"Degradation: {degradation:.1f}%")
if roc_auc is not None:
    print(f"  Confidence Estimator AUROC: {roc_auc:.3f}\n")
else:
    print("  AUROC: Pending (Needs more 'wasCorrect' True/False labels in JSON)\n")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Context Length Growth
# ══════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(turns_list, avg_tokens, color=BLUE, linewidth=2, marker='o',
        markersize=6, markerfacecolor='white', markeredgewidth=2, zorder=3)
ax.fill_between(turns_list, [a - s for a, s in zip(avg_tokens, std_tokens)],
                [a + s for a, s in zip(avg_tokens, std_tokens)], alpha=0.15, color=BLUE)
for t, v in zip(turns_list, avg_tokens):
    ax.annotate(f'{v:.0f}', (t, v), textcoords="offset points", xytext=(0, 12),
                ha='center', fontsize=8.5, color=BLUE, fontweight='500', bbox=label_bg)
ax.set_xlabel("Reasoning turn number", fontsize=10, color=GRAY)
ax.set_ylabel("Mean context token count", fontsize=10, color=GRAY)
ax.set_title("Figure 1: Context length growth across turns", fontsize=11, fontweight='500', pad=12)
ax.set_xticks(turns_list)
ax.tick_params(colors=GRAY)
plt.tight_layout()
plt.savefig("figure1_context_growth.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Retrieval Quality Across Turns
# ══════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(5.5, 3.8))
ax.errorbar(turns_list, avg_sims, yerr=std_sims, color=RED, linewidth=2, marker='o',
            markersize=6, markerfacecolor='white', markeredgewidth=2, capsize=4, capthick=1.5, zorder=3)
ax.fill_between(turns_list, [a - s for a, s in zip(avg_sims, std_sims)],
                [a + s for a, s in zip(avg_sims, std_sims)], alpha=0.10, color=RED)
for t, v in zip(turns_list, avg_sims):
    # Added bbox background and slightly higher offset to avoid line/errorbar overlap
    ax.annotate(f'{v:.3f}', (t, v), textcoords="offset points", xytext=(0, 14),
                ha='center', fontsize=8.5, color=RED, fontweight='500', bbox=label_bg)
ax.set_xlabel("Reasoning turn number", fontsize=10, color=GRAY)
ax.set_ylabel("Mean retrieval relevance score", fontsize=10, color=GRAY)
ax.set_title("Figure 2: Retrieval quality across turns", fontsize=11, fontweight='500', pad=12)
ax.set_xticks(turns_list)
ax.tick_params(colors=GRAY)
plt.tight_layout()
plt.savefig("figure2_retrieval_quality.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Keyword Drift
# ══════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(5.5, 3.8))
if drift_pairs:
    bar_colors = [BLUE, AMBER, RED, GREEN][:len(drift_pairs)]
    bars = ax.bar(drift_pairs, drift_vals, color=bar_colors, width=0.5, zorder=3)
    for bar, val in zip(bars, drift_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='500', color=GRAY)
    ax.set_ylim(min(drift_vals) * 0.90, max(drift_vals) * 1.10)
ax.set_xlabel("Consecutive turn pair", fontsize=10, color=GRAY)
ax.set_ylabel("Mean inter-turn cosine similarity", fontsize=10, color=GRAY)
ax.set_title("Figure 3: Keyword drift across turns", fontsize=11, fontweight='500', pad=12)
ax.tick_params(colors=GRAY)
plt.tight_layout()
plt.savefig("figure3_keyword_drift.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Context vs Similarity Scatter
# ══════════════════════════════════════════════════════════════════
fig4, ax = plt.subplots(figsize=(5.5, 3.8))
ax.scatter(tier1_ctx, tier1_sim, color=GREEN, alpha=0.55, s=30, edgecolors='none', zorder=3)
if len(tier1_ctx) > 1:
    ax.plot(x_line, p_line(x_line), color=RED, linewidth=1.5, linestyle='--', label=f'r = {r_val:.3f}', zorder=4)
    ax.legend(fontsize=9, frameon=False, loc='upper right', labelcolor=RED)
ax.set_xlabel("Context token count", fontsize=10, color=GRAY)
ax.set_ylabel("Cosine similarity score", fontsize=10, color=GRAY)
ax.set_title("Figure 4: Context length vs retrieval quality", fontsize=11, fontweight='500', pad=12)
ax.tick_params(colors=GRAY)
plt.tight_layout()
plt.savefig("figure4_scatter.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — ROC Curve (New Individual File)
# ══════════════════════════════════════════════════════════════════
fig5, ax = plt.subplots(figsize=(5.5, 3.8))
if fpr is not None and tpr is not None:
    ax.plot(fpr, tpr, color=AMBER, linewidth=2, label=f'AUC = {roc_auc:.2f}', zorder=3)
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.5, linestyle='--', zorder=2)
    ax.legend(fontsize=9, frameon=False, loc='lower right', labelcolor=GRAY)
else:
    ax.text(0.5, 0.5, "Pending Data\nAdd 'wasCorrect': true/false to logs", 
            ha='center', va='center', color=GRAY, fontsize=10)
ax.set_xlabel("False positive rate", fontsize=10, color=GRAY)
ax.set_ylabel("True positive rate", fontsize=10, color=GRAY)
ax.set_title("Figure 5: Confidence estimator ROC curve", fontsize=11, fontweight='500', pad=12)
ax.tick_params(colors=GRAY)
plt.tight_layout()
plt.savefig("figure5_roc_curve.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════════════
# COMBINED — All 5 figures in one file (using GridSpec)
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 11))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)
fig.suptitle(f"Multi-Turn RAG Degradation Analysis — Math Professor AI\n(n={len(sessions)} sessions)", 
             fontsize=13, fontweight='500', y=0.96, color=GRAY)

# Panel 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(turns_list, avg_tokens, color=BLUE, linewidth=2, marker='o', markersize=5, markerfacecolor='white', markeredgewidth=2, zorder=3)
ax1.fill_between(turns_list, [a - s for a, s in zip(avg_tokens, std_tokens)], [a + s for a, s in zip(avg_tokens, std_tokens)], alpha=0.15, color=BLUE)
for t, v in zip(turns_list, avg_tokens):
    ax1.annotate(f'{v:.0f}', (t, v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color=BLUE, bbox=label_bg)
ax1.set_xlabel("Reasoning turn", fontsize=9, color=GRAY)
ax1.set_ylabel("Mean context tokens", fontsize=9, color=GRAY)
ax1.set_title("Figure 1: Context length growth", fontsize=10, fontweight='500')
ax1.set_xticks(turns_list)
ax1.tick_params(colors=GRAY, labelsize=8)

# Panel 2
ax2 = fig.add_subplot(gs[0, 1])
ax2.errorbar(turns_list, avg_sims, yerr=std_sims, color=RED, linewidth=2, marker='o', markersize=5, markerfacecolor='white', markeredgewidth=2, capsize=3, capthick=1.5, zorder=3)
ax2.fill_between(turns_list, [a - s for a, s in zip(avg_sims, std_sims)], [a + s for a, s in zip(avg_sims, std_sims)], alpha=0.10, color=RED)
for t, v in zip(turns_list, avg_sims):
    ax2.annotate(f'{v:.3f}', (t, v), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=8, color=RED, bbox=label_bg)
ax2.set_xlabel("Reasoning turn", fontsize=9, color=GRAY)
ax2.set_ylabel("Mean retrieval similarity", fontsize=9, color=GRAY)
ax2.set_title("Figure 2: Retrieval quality across turns", fontsize=10, fontweight='500')
ax2.set_xticks(turns_list)
ax2.tick_params(colors=GRAY, labelsize=8)

# Panel 3
ax3 = fig.add_subplot(gs[1, 0])
if drift_pairs:
    bar_colors = [BLUE, AMBER, RED, GREEN][:len(drift_pairs)]
    bars = ax3.bar(drift_pairs, drift_vals, color=bar_colors, width=0.5, zorder=3)
    for bar, val in zip(bars, drift_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002, f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='500', color=GRAY)
    ax3.set_ylim(min(drift_vals) * 0.90, max(drift_vals) * 1.10)
ax3.set_xlabel("Consecutive turn pair", fontsize=9, color=GRAY)
ax3.set_ylabel("Inter-turn cosine similarity", fontsize=9, color=GRAY)
ax3.set_title("Figure 3: Keyword drift across turns", fontsize=10, fontweight='500')
ax3.tick_params(colors=GRAY, labelsize=8)

# Panel 4
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(tier1_ctx, tier1_sim, color=GREEN, alpha=0.5, s=20, edgecolors='none', zorder=3)
if len(tier1_ctx) > 1:
    ax4.plot(x_line, p_line(x_line), color=RED, linewidth=1.5, linestyle='--', label=f'r = {r_val:.3f}', zorder=4)
    ax4.legend(fontsize=8, frameon=False, labelcolor=RED)
ax4.set_xlabel("Context token count", fontsize=9, color=GRAY)
ax4.set_ylabel("Cosine similarity", fontsize=9, color=GRAY)
ax4.set_title("Figure 4: Context length vs retrieval quality", fontsize=10, fontweight='500')
ax4.tick_params(colors=GRAY, labelsize=8)

# Panel 5 (Spanning the bottom row)
ax5 = fig.add_subplot(gs[2, :])
if fpr is not None and tpr is not None:
    ax5.plot(fpr, tpr, color=AMBER, linewidth=2, label=f'AUC = {roc_auc:.2f}', zorder=3)
    ax5.plot([0, 1], [0, 1], color=GRAY, linewidth=1.5, linestyle='--', zorder=2)
    ax5.legend(fontsize=9, frameon=False, loc='lower right', labelcolor=GRAY)
else:
    ax5.text(0.5, 0.5, "Pending Data: Add 'wasCorrect': true/false to JSON logs to plot ROC Curve", 
             ha='center', va='center', color=GRAY, fontsize=10)
ax5.set_xlabel("False positive rate", fontsize=9, color=GRAY)
ax5.set_ylabel("True positive rate", fontsize=9, color=GRAY)
ax5.set_title("Figure 5: Confidence estimator ROC curve", fontsize=10, fontweight='500')
ax5.tick_params(colors=GRAY, labelsize=8)

plt.savefig("rag_analysis_publication.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Saved: rag_analysis_publication.png")
print("\nDone. 6 files saved:")
print("  figure1_context_growth.png")
print("  figure2_retrieval_quality.png")
print("  figure3_keyword_drift.png")
print("  figure4_scatter.png")
print("  figure5_roc_curve.png")
print("  rag_analysis_publication.png  ← combined overview (all 5 panels)")