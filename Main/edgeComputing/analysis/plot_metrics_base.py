import os, json, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Function to process a single results directory ---
def analyze_run(results_dir):
    logs = glob.glob(os.path.join(results_dir, "*_events.log"))
    if not logs:
        print(f"Warning: No logs found in {results_dir}")
        return {}

    events = []
    for f in logs:
        with open(f, "r") as fh:
            for line in fh:
                try:
                    events.append(json.loads(line.strip()))
                except Exception:
                    continue

    client_data = {}
    client_names = sorted(list(set(e['node'] for e in events)))
    
    for client in client_names:
        client_events = [e for e in events if e['node'] == client and e['event'] == 'model_size_after_bytes']
        if not client_events:
            continue
        
        # Calculate the average size of the model updates that were NOT skipped
        update_sizes = [e['value'] for e in client_events]
        avg_size = np.mean(update_sizes) if update_sizes else 0
        client_data[client] = avg_size
        
    return client_data

# --- Main analysis ---
# Define paths to the two results directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../results"))
baseline_results_dir = os.path.join(BASE_DIR, "baseline_run")
smart_results_dir = os.path.join(BASE_DIR, "smart_run")
plot_output_dir = os.path.join(BASE_DIR, "comparison_plots")
os.makedirs(plot_output_dir, exist_ok=True)

# Analyze both runs
baseline_data = analyze_run(baseline_results_dir)
smart_data = analyze_run(smart_results_dir)

if not baseline_data or not smart_data:
    raise SystemExit("Missing data from one or both simulation runs. Please ensure both ran correctly.")

# --- Create the Comparison Plot ---
clients = sorted(baseline_data.keys())
baseline_sizes = [baseline_data.get(c, 0) for c in clients]
smart_sizes = [smart_data.get(c, 0) for c in clients]

x = np.arange(len(clients))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_sizes, width, label='Baseline (Uncompressed)')
rects2 = ax.bar(x + width/2, smart_sizes, width, label='SMART (Adaptive Compression)')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Average Model Update Size (Bytes)')
ax.set_title('Comparison of Average Update Size per Client')
ax.set_xticks(x)
ax.set_xticklabels(clients)
ax.legend()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation for y-axis

fig.tight_layout()

# Save the plot
output_path = os.path.join(plot_output_dir, "size_comparison_per_client.png")
plt.savefig(output_path)
print(f"Comparison plot saved to: {output_path}")
plt.show()