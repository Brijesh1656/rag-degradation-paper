import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ── 1. Load Data ──────────────────────────────────────────────────
files = glob.glob("*.json")
data = []
for f in files:
    with open(f) as fp:
        data.extend(json.load(fp))

# Remove duplicates by sessionId + turnNumber
seen = set()
unique_data = []
for row in data:
    key = (row["sessionId"], row["turnNumber"])
    if key not in seen:
        seen.add(key)
        unique_data.append(row)
data = unique_data

print(f"\n{'='*55}")
print(f"  FULL ANALYSIS — {len(data)} turns")
print(f"{'='*55}")

# ── 2. Sessions & Basic Stats ─────────────────────────────────────
sessions = defaultdict(list)
for row in data:
    sessions[row["sessionId"]].append(row)

print(f"\nTotal sessions     : {len(sessions)}")
if sessions:
    turn_counts = [len(v) for v in sessions.values()]
    print(f"Avg turns/session  : {np.mean(turn_counts):.2f}")
    print(f"Min turns/session  : {min(turn_counts)}")
    print(f"Max turns/session  : {max(turn_counts)}")

# ── 3. Per-Turn Stats ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print("TABLE 2: Per-Turn Retrieval Statistics")
print(f"{'─'*60}")
print(f"{'Turn':<6} {'N':>4} {'Avg Sim':>10} {'Std Sim':>10} {'Correct%':>10} {'Avg Tokens':>12}")
print(f"{'─'*60}")

turn_data = defaultdict(list)
for row in data:
    turn_data[row["turnNumber"]].append(row)

for t in sorted(turn_data.keys()):
    rows = turn_data[t]
    sims = [r["similarityScore"] for r in rows]
    toks = [r["contextTokenCount"] for r in rows]
    correct = [r["wasCorrect"] for r in rows if r.get("wasCorrect") is not None]
    correct_pct = (sum(correct) / len(correct) * 100) if correct else float("nan")
    print(f"  {t:<4} {len(rows):>4} {np.mean(sims):>10.4f} {np.std(sims):>10.4f} "
          f"{correct_pct:>9.1f}% {np.mean(toks):>12.0f}")

# ── 4. Degradation & Correlation ──────────────────────────────────
t1_sims = [r["similarityScore"] for r in turn_data.get(1, [])]
t4_sims = [r["similarityScore"] for r in turn_data.get(4, [])]
if t1_sims and t4_sims:
    deg = (np.mean(t1_sims) - np.mean(t4_sims)) / np.mean(t1_sims) * 100
    print(f"\nRetrieval degradation turn 1→4 : {deg:.1f}%")
    if len(t1_sims) > 1 and len(t4_sims) > 1:
        t_stat, p_val = stats.ttest_ind(t1_sims, t4_sims)
        print(f"Statistical significance       : p = {p_val:.4f}")

print(f"\n{'─'*60}")
print("CORRELATION: Context Length vs Retrieval Quality")
print(f"{'─'*60}")
tier1 = [r for r in data if r["retrievalTier"] == 1]
r_val, p_val = 0, 1
if tier1 and len(tier1) > 1:
    ctx = [r["contextTokenCount"] for r in tier1]
    sim = [r["similarityScore"] for r in tier1]
    r_val, p_val = stats.pearsonr(ctx, sim)
    print(f"Tier 1 only (n={len(tier1)}):\n  Pearson r = {r_val:.4f}  (p = {p_val:.4f})")

all_ctx = [r["contextTokenCount"] for r in data]
all_sim = [r["similarityScore"] for r in data]
r_all = 0
if len(data) > 1:
    r_all, p_all = stats.pearsonr(all_ctx, all_sim)
    print(f"All turns (n={len(data)}):\n  Pearson r = {r_all:.4f}  (p = {p_all:.4f})")

# ── 5. Keyword Drift ──────────────────────────────────────────────
print(f"\n{'─'*60}")
print("KEYWORD DRIFT: Inter-turn Sub-query Similarity")
print(f"{'─'*60}")
vectorizer = TfidfVectorizer()
pair_sims = defaultdict(list)
for sid, turns in sessions.items():
    turns_sorted = sorted(turns, key=lambda x: x["turnNumber"])
    queries = [t["subQuery"] for t in turns_sorted if t.get("subQuery")]
    if len(queries) < 2: continue
    try:
        vecs = vectorizer.fit_transform(queries).toarray()
        for i in range(len(vecs) - 1):
            pair = f"{i+1}-{i+2}"
            cs = cosine_similarity([vecs[i]], [vecs[i + 1]])[0][0]
            pair_sims[pair].append(cs)
    except:
        pass

for pair in ["1-2", "2-3", "3-4", "4-5", "5-6"]:
    if pair_sims[pair]:
        print(f"  Turns {pair}: {np.mean(pair_sims[pair]):.4f} (n={len(pair_sims[pair])})")

# ── 6. Confidence Estimator AUROC ─────────────────────────────────
print(f"\n{'─'*60}")
print("CONFIDENCE ESTIMATOR AUROC (TABLE 3)")
print(f"{'─'*60}")
labeled = [r for r in data if r.get("wasCorrect") is not None]
C_baseline = np.mean([r["contextTokenCount"] for r in data if r["turnNumber"] == 1]) if data else 1

auroc_results = {}
fpr, tpr, roc_auc = None, None, None

if len(labeled) >= 2: # Lowered threshold so it runs even with minimal labeled data
    from sklearn.metrics import roc_auc_score, roc_curve
    X, y = [], []
    for r in labeled:
        t, C, g = r["turnNumber"], r["contextTokenCount"], r["similarityScore"]
        alpha, beta = 0.18, 0.09
        conf = g * np.exp(-alpha * t) * (1 - beta * np.log(max(C / C_baseline, 1e-9)))
        X.append([conf, t, C, g])
        y.append(1 if r["wasCorrect"] else 0)

    X, y = np.array(X), np.array(y)
    
    if len(set(y)) == 2: # Ensure we have both correct and incorrect labels
        auroc_results["Turn-aware"] = roc_auc_score(y, X[:, 0])
        auroc_results["Turn only"] = roc_auc_score(y, -X[:, 1])
        auroc_results["Context only"] = roc_auc_score(y, -X[:, 2])
        auroc_results["Similarity only"] = roc_auc_score(y, X[:, 3])
        
        # Data for Figure 5
        fpr, tpr, _ = roc_curve(y, X[:, 0])
        roc_auc = auroc_results["Turn-aware"]
        
        for k, v in auroc_results.items():
            print(f"  {k:<20} : {v:.4f}")
    else:
        print("  Need BOTH correct AND incorrect labels in data to calculate AUROC.")
else:
    print(f"  Not enough labeled data (have {len(labeled)}). Need at least 2 labeled turns.")

# ── 7. Distributions ──────────────────────────────────────────────
print(f"\n{'─'*60}")
print("PROBLEM TYPE & TIER DISTRIBUTION")
print(f"{'─'*60}")
types = defaultdict(int)
for sid, turns in sessions.items():
    if "problemType" in turns[0]:
        types[turns[0]["problemType"]] += 1
for k, v in sorted(types.items(), key=lambda x: -x[1]):
    print(f"  {k:<14}: {v:>4} sessions ({v / len(sessions) * 100:.0f}%)")

for t in [1, 2, 3]:
    n = sum(1 for r in data if r.get("retrievalTier") == t)
    if data: print(f"  Tier {t}: {n:>4} turns ({n / len(data) * 100:.0f}%)")

# ── 8. Text File Export ───────────────────────────────────────────
t1_mean = np.mean([r["similarityScore"] for r in turn_data.get(1, [])]) if 1 in turn_data else 0
t4_mean = np.mean([r["similarityScore"] for r in turn_data.get(4, [])]) if 4 in turn_data else 0

with open("analysis_results.txt", "w") as f:
    f.write(f"RAG Analysis Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"{'='*55}\n\n")
    f.write(f"Total sessions     : {len(sessions)}\n")
    f.write(f"Total turns        : {len(data)}\n")
    if sessions: f.write(f"Avg turns/session  : {np.mean(turn_counts):.2f}\n")
    f.write(f"\n{'─'*55}\nNUMBERS TO PUT IN PAPER\n{'─'*55}\n")
    f.write(f"Turn 1 mean sim    : {t1_mean:.3f}\n")
    f.write(f"Turn 4 mean sim    : {t4_mean:.3f}\n")
    if t1_mean > 0: f.write(f"Degradation        : {(t1_mean - t4_mean) / t1_mean * 100:.1f}%\n")
    f.write(f"Pearson r          : {r_val:.3f}\n")
    f.write(f"C_baseline tokens  : {C_baseline:.0f}\n")
    
    if auroc_results:
        f.write(f"\n{'─'*55}\nTABLE 3: Confidence Estimator AUROC\n{'─'*55}\n")
        for k, v in auroc_results.items():
            f.write(f"{k:<20} : {v:.4f}\n")

    f.write(f"\n{'─'*55}\nKEYWORD DRIFT\n{'─'*55}\n")
    for pair in ["1-2", "2-3", "3-4", "4-5"]:
        if pair_sims[pair]: f.write(f"Turns {pair}: {np.mean(pair_sims[pair]):.4f}\n")
        
    f.write(f"\n{'─'*55}\nPER-TURN TABLE\n{'─'*55}\n")
    f.write(f"{'Turn':<6} {'N':>4} {'Avg Sim':>10} {'Avg Tokens':>12}\n")
    for t in sorted(turn_data.keys()):
        rows = turn_data[t]
        f.write(f"  {t:<4} {len(rows):>4} {np.mean([r['similarityScore'] for r in rows]):>10.4f} {np.mean([r['contextTokenCount'] for r in rows]):>12.0f}\n")

print("\nResults saved to analysis_results.txt")

# ══════════════════════════════════════════════════════════════════
# GRAPHS (Now with 5 Subplots)
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3)
fig.suptitle('Multi-Turn RAG Degradation Analysis', fontsize=16, fontweight='bold', y=0.96)

turns_list = sorted(turn_data.keys())

# Graph 1
ax1 = fig.add_subplot(gs[0, 0])
avg_tokens = [np.mean([r["contextTokenCount"] for r in turn_data[t]]) for t in turns_list]
std_tokens = [np.std([r["contextTokenCount"] for r in turn_data[t]]) for t in turns_list]
ax1.plot(turns_list, avg_tokens, 'b-o', linewidth=2, markersize=8)
ax1.fill_between(turns_list, [a - s for a, s in zip(avg_tokens, std_tokens)], [a + s for a, s in zip(avg_tokens, std_tokens)], alpha=0.2, color='blue')
ax1.set_title('Figure 1: Context Length Growth', fontweight='bold')
ax1.set_xlabel('Turn Number')
ax1.set_ylabel('Mean Tokens')
ax1.grid(True, alpha=0.3)

# Graph 2
ax2 = fig.add_subplot(gs[0, 1])
avg_sims = [np.mean([r["similarityScore"] for r in turn_data[t]]) for t in turns_list]
std_sims = [np.std([r["similarityScore"] for r in turn_data[t]]) for t in turns_list]
ax2.errorbar(turns_list, avg_sims, yerr=std_sims, fmt='r-o', linewidth=2, capsize=5)
ax2.set_title('Figure 2: Retrieval Quality', fontweight='bold')
ax2.set_xlabel('Turn Number')
ax2.set_ylabel('Relevance Score')
ax2.grid(True, alpha=0.3)

# Graph 3
ax3 = fig.add_subplot(gs[1, 0])
drift_pairs, drift_vals = [], []
for pair in ["1-2", "2-3", "3-4", "4-5"]:
    if pair_sims[pair]:
        drift_pairs.append(pair)
        drift_vals.append(np.mean(pair_sims[pair]))
if drift_pairs:
    ax3.bar(drift_pairs, drift_vals, color=['#2196F3', '#FF9800', '#F44336', '#9C27B0'], edgecolor='black', alpha=0.8)
ax3.set_title('Figure 3: Keyword Drift', fontweight='bold')
ax3.set_ylabel('Cosine Similarity')
ax3.grid(True, alpha=0.3, axis='y')

# Graph 4
ax4 = fig.add_subplot(gs[1, 1])
if tier1:
    tier1_ctx = [r["contextTokenCount"] for r in tier1]
    tier1_sim = [r["similarityScore"] for r in tier1]
    ax4.scatter(tier1_ctx, tier1_sim, alpha=0.6, color='green', edgecolor='black')
    if len(tier1_ctx) > 1:
        z = np.polyfit(tier1_ctx, tier1_sim, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(tier1_ctx), max(tier1_ctx), 100)
        ax4.plot(x_line, p_line(x_line), "r--", linewidth=2, label=f'r = {r_val:.3f}')
        ax4.legend()
ax4.set_title('Figure 4: Context vs Similarity', fontweight='bold')
ax4.set_xlabel('Context Tokens')
ax4.set_ylabel('Similarity')
ax4.grid(True, alpha=0.3)

# Graph 5: ROC Curve (Spans bottom row)
ax5 = fig.add_subplot(gs[2, :])
if fpr is not None and tpr is not None:
    ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'Proposed Estimator (AUC = {roc_auc:.2f})')
    ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax5.set_title('Figure 5: Confidence Estimator ROC Curve', fontweight='bold')
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.legend(loc="lower right")
else:
    ax5.text(0.5, 0.5, "Not enough labeled data (True/False 'wasCorrect')\nAdd labels to JSON to generate ROC Curve.", 
             ha='center', va='center', fontsize=12, color='gray')
    ax5.set_title('Figure 5: Confidence Estimator ROC Curve (PENDING DATA)', fontweight='bold')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rag_analysis_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graphs saved as rag_analysis_final.png")