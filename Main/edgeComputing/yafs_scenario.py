#NOTE !!!! IF YOU WANT TO RUN THE COMPRESSED VERSION(THE SMART ONES)
'''
THIS HAS TO BE CHANGED IN THIS CODE:
results_dir = os.path.abspath("results/smart_run")
# ...
subprocess.Popen(["python3", "Main/edgeComputing/fl_client.py"], env=env)
'''

#NOTE FOR THE SAME ABOVE IF YOU WANT TO RUN THE UNCOMPRESSED VERSION (THE BASELINE ONE)
'''
THIS HAS TO BE CHANGED IN THIS CODE:
results_dir = os.path.abspath("results/baseline_run")
# ...
subprocess.Popen(["python3", "Main/edgeComputing/fl_client_base.py"], env=env)
'''


from yafs.topology import Topology
import networkx as nx
import subprocess, os, time

# --- CONFIG ---
results_dir = os.path.abspath("results/smart_run")
server_addr = "127.0.0.1:8080"


# --- DEFINE TOPOLOGY ---
topo = Topology()
topo.G = nx.DiGraph()

# Add nodes manually
topo.G.add_node("Server", model={"IPT": 10000, "RAM": 8})
for i in range(3):
    topo.G.add_node(f"Client{i}", model={"IPT": 1000, "RAM": 2})
    topo.G.add_edge(f"Client{i}", "Server", PR=1)

# --- CALLBACK TO LAUNCH CLIENTS ---
def launch_client(client_id):
    env = os.environ.copy()
    env["CLIENT_ID"] = str(client_id)
    env["RESULTS_DIR"] = results_dir
    #env["BW_MBPS"] = str(bw)
    subprocess.Popen(["python3", "Main/edgeComputing/fl_client.py"], env=env)
    print(f"[YAFS] Started client{client_id} (bandwidth will be randomized per round)")

# --- SIMULATION EXECUTION ---
print("[YAFS] Starting Flower server...")
subprocess.Popen(["python3", "Main/edgeComputing/fl_server.py"])
time.sleep(3)  # wait for server to be ready

# Launch clients with staggered timing (simulate network startup)
for i in range(3):
    launch_client(i)
    time.sleep(2)  # delay between client starts

print("[YAFS] Simulation running — press Ctrl+C to stop.")
try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    print("\n[YAFS] Simulation stopped by user.")
