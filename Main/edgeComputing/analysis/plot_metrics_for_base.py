# Main/edgeComputing/analysis/plot_metrics.py
import os, json, glob, math
from collections import defaultdict, OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # Import for custom legend

# --- PATH CONFIGURATION ---
# CHANGE 1: Define where to READ the log files from
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../results/baseline_run"))

# CHANGE 2: Define where to SAVE the generated plots
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../results/analysis_base"))
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create the output folder if it doesn't exist

# 1) Read all event logs from the INPUT_DIR
logs = glob.glob(os.path.join(INPUT_DIR, "*_events.log")) # <-- Uses INPUT_DIR
if not logs:
    raise SystemExit(f"No event logs found in {INPUT_DIR}")

events = []
for f in logs:
    with open(f, "r") as fh:
        for line in fh:
            try:
                rec = json.loads(line.strip())
                if "ts" in rec and "node" in rec and "event" in rec:
                    events.append(rec)
            except Exception:
                continue

# (The rest of the data processing code remains the same)

# 2) Organize events per client and per round
events = sorted(events, key=lambda x: (x["node"], x["ts"]))
by_node = defaultdict(list)
for e in events:
    by_node[e["node"]].append(e)

# 3) Aggregate metrics per client
clients = sorted(by_node.keys())
summary = {}
global_first_ts = min(e["ts"] for e in events) if events else 0
global_last_ts = max(e["ts"] for e in events) if events else 0

for c in clients:
    evs = by_node[c]
    rounds = []
    current = []
    last_ts = None
    for e in evs:
        if last_ts is None or e["ts"] - last_ts > 30:
            if current:
                rounds.append(current)
            current = [e]
        else:
            current.append(e)
        last_ts = e["ts"]
    if current:
        rounds.append(current)

    round_metrics = []
    total_bytes_sent = 0
    count_skip = count_full = count_compress = 0
    for r in rounds:
        m = {"ts": min(x["ts"] for x in r), "decision": "skip", "model_size_after_bytes": 0} # Default to skip
        for x in r:
            ev = x["event"]
            val = x["value"]
            if ev == "decision":
                m["decision"] = val
            elif ev == "model_size_after_bytes":
                m["model_size_after_bytes"] = int(val)
        
        if m["decision"] == "skip":
            count_skip += 1
        elif m["decision"] == "compress":
            count_compress += 1
            total_bytes_sent += m["model_size_after_bytes"]
        elif m["decision"] == "full":
            count_full += 1
            total_bytes_sent += m["model_size_after_bytes"]

        round_metrics.append(m)

    total_time = global_last_ts - global_first_ts
    summary[c] = {
        "rounds": round_metrics,
        "total_bytes_sent": total_bytes_sent,
        "count_skip": count_skip,
        "count_full": count_full,
        "count_compress": count_compress,
        "total_time_s": total_time
    }

# 4) Print a short summary
print("=== SUMMARY OF BASELINE RUN ===") # <-- Updated title
# ... (rest of the summary printing is fine) ...
total_bytes = sum(summary[c]["total_bytes_sent"] for c in summary)
total_time = max(s["total_time_s"] for s in summary.values()) if summary else 0.0
bw_eff = total_bytes / total_time if total_time > 0 else float("nan")
print(f"Clients found: {clients}")
print(f"Total bytes sent (all clients): {total_bytes} bytes")
print(f"Total elapsed time: {total_time:.1f} s")
print(f"Bandwidth efficiency (bytes/sec): {bw_eff:.1f}")

for c in clients:
    s = summary[c]
    print(f"- {c}: bytes_sent={s['total_bytes_sent']}, skips={s['count_skip']}, full={s['count_full']}, compress={s['count_compress']}")


# 5) Plots
# All plots will now save to the OUTPUT_DIR

# 5a. Model size and decision per round (SCATTER PLOT)
for c in clients:
    rm = summary[c]["rounds"]
    if not rm: continue
    
    rounds_idx = list(range(1, len(rm) + 1))
    sizes = [r["model_size_after_bytes"] or 0 for r in rm]
    decisions = [r["decision"] or 'skip' for r in rm]
    
    color_map = {'full': 'orange', 'compress': 'blue', 'skip': '#cccccc'}
    colors = [color_map.get(d, '#cccccc') for d in decisions]

    plt.figure(figsize=(12, 6))
    plt.title(f"{c} Update Size and Decision per Round (Baseline)") # <-- Updated title
    plt.xlabel("Round")
    plt.ylabel("Bytes Sent")
    plt.scatter(rounds_idx, sizes, c=colors, alpha=0.8, s=50)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Full Update', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Compressed Update', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Skip', markerfacecolor='#cccccc', markersize=10)
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{c}_update_sizes_scatter.png") # <-- Uses OUTPUT_DIR
    plt.savefig(out)
    plt.close()

# 5b. Decisions per client (stacked bar)
df_rows = []
for c in clients:
    s = summary[c]
    df_rows.append({"client": c, "decision": "skip", "count": s["count_skip"]})
    df_rows.append({"client": c, "decision": "compress", "count": s["count_compress"]})
    df_rows.append({"client": c, "decision": "full", "count": s["count_full"]})
df = pd.DataFrame(df_rows)
if not df.empty:
    pivot = df.pivot(index="client", columns="decision", values="count")
    pivot.plot(kind="bar", stacked=True, color=['blue', 'orange', 'green'])
    plt.title("Decisions per client (counts) - Baseline") # <-- Updated title
    plt.ylabel("count")
    plt.savefig(os.path.join(OUTPUT_DIR, "decisions_per_client.png")) # <-- Uses OUTPUT_DIR
    plt.close()

# 5c. Bytes sent vs skips (scatter)
clients_df = [{"client": c, "bytes_sent": summary[c]["total_bytes_sent"], "skips": summary[c]["count_skip"]} for c in clients]
cdf = pd.DataFrame(clients_df)
if not cdf.empty:
    plt.figure()
    plt.scatter(cdf["bytes_sent"], cdf["skips"])
    plt.xlabel("bytes_sent")
    plt.ylabel("skips")
    plt.title("Bytes sent vs skips (per client) - Baseline") # <-- Updated title
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "bytes_vs_skips.png")) # <-- Uses OUTPUT_DIR
    plt.close()

print(f"\nPlots for baseline run saved to: {OUTPUT_DIR}") # <-- Final confirmation message