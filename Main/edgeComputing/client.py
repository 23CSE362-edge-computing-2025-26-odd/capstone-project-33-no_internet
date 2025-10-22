# edgeComputing/client.py

from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from compression import should_skip
from model import CNNModel
from dataset import get_mnist_loaders
from compression import should_skip
import json
import os
client_id = int(os.getenv("CLIENT_ID", 0))

def train_one_round(model_state_dict, epochs=1, client_id=0, results_dir="./results", bw_mbps=None,compression_method="topk"):
    """
    Trains locally and uses a multi-tiered adaptive strategy for sending updates.
    """
    import time, torch
    from model import CNNModel
    from dataset import get_mnist_loaders
    from compression import compress_state_dict
    from instrumentation import write_event, model_bytes_from_state


    if bw_mbps is not None:
        write_event(results_dir, f"client{client_id}", "bandwidth_mbps", bw_mbps, "Mbps")

    # === NEW: Define the bandwidth tiers and compression ratio ranges ===
    # Bandwidth thresholds (in Mbps)
    skip_bw = 0.1
    high_compression_bw_max = 5.0
    med_compression_bw_max = 12.5
    
    # Top-K ratio ranges for compression
    # For Low Bandwidth zone (high compression)
    low_bw_min_ratio = 0.05  # 5%
    low_bw_max_ratio = 0.15  # 15%
    
    # For Medium Bandwidth zone (low compression)
    med_bw_min_ratio = 0.15  # 15%
    med_bw_max_ratio = 0.40  # 40%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Standard Training (no changes here) ---
    model = CNNModel().to(device)
    model.load_state_dict(model_state_dict)
    train_loader, _ = get_mnist_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    train_time_s = time.time() - t0
    
    state_to_send = model.state_dict()
    size_full = model_bytes_from_state(state_to_send)
    write_event(results_dir, f"client{client_id}", "model_size_before_bytes", size_full, "bytes")

    # === NEW MULTI-TIERED DECISION LOGIC ===
    decision = "full"  # Default if bandwidth is high
    payload = state_to_send
    size_after = size_full
    dynamic_topk_ratio = 1.0 # Default to 1.0 (full)


        # Create the dictionary of metrics to return
    final_metrics = {
        "train_time_s": train_time_s,
        "decision": decision,
        "model_size_before_bytes": size_full,
        "model_size_after_bytes": size_after,
    }
    
    # Calculate accuracy after training and add it to the metrics dictionary
    # This only runs if an update is being sent (not skipped)

    accuracy = 0.0

    if decision != "skip":
        eval_results = evaluate_model(model.state_dict())
        accuracy = eval_results.get("accuracy", 0.0)
        
    final_metrics = {
        "train_time_s": train_time_s,
        "decision": decision,
        "model_size_before_bytes": size_full,
        "model_size_after_bytes": size_after,
        "accuracy": accuracy, # Now 'accuracy' always has a value
        "client_id": client_id
    }

    # The single, final return statement

    if bw_mbps is None or bw_mbps < skip_bw:
        # --- Skip Zone ---
        decision = "skip"
        write_event(results_dir, f"client{client_id}", "decision", decision, "")
        return None, len(train_loader.dataset), {"train_time_s": train_time_s, "decision": decision}

    elif bw_mbps < high_compression_bw_max:
        # --- High Compression Zone ---
        decision = "compress"
        zone_width = high_compression_bw_max - skip_bw
        client_pos = bw_mbps - skip_bw
        bw_score = client_pos / zone_width
        
        ratio_range = low_bw_max_ratio - low_bw_min_ratio
        dynamic_topk_ratio = low_bw_min_ratio + (bw_score * ratio_range)

    elif bw_mbps < med_compression_bw_max:
        # --- Medium Compression Zone ---
        decision = "compress"
        zone_width = med_compression_bw_max - high_compression_bw_max
        client_pos = bw_mbps - high_compression_bw_max
        bw_score = client_pos / zone_width
        
        ratio_range = med_bw_max_ratio - med_bw_min_ratio
        dynamic_topk_ratio = med_bw_min_ratio + (bw_score * ratio_range)

    # --- Apply compression if decision was 'compress' ---
    if decision == "compress":
        print(f"[Client {client_id}] Adaptive compression: BW={bw_mbps:.2f} Mbps -> topk_ratio={dynamic_topk_ratio:.2f}")
        compressed_payload = compress_state_dict(state_to_send, method=compression_method, topk_ratio=dynamic_topk_ratio)
        size_after = len(compressed_payload)
        payload = compressed_payload
    
    write_event(results_dir, f"client{client_id}", "decision", decision, "")
    write_event(results_dir, f"client{client_id}", "model_size_after_bytes", size_after, "bytes")

    return model.state_dict(), len(train_loader.dataset), {"train_time_s": train_time_s,
                                                           "decision": decision,
                                                           "model_size_before_bytes": size_full,
                                                           "model_size_after_bytes": size_after,
                                                           "accuracy": accuracy,
                                                           "client_id": client_id}
    # The single, final return statement at the end of the function
    return model.state_dict(), len(train_loader.dataset), final_metrics
    

    # Add evaluation step to get accuracy
    # ... (code before the return statements) ...
    

def evaluate_model(model_state_dict):
    """Evaluate model accuracy on test set."""
    import torch
    from model import CNNModel
    from dataset import get_mnist_loaders

    # Check for Apple Metal (MPS) first, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[Client {client_id}] Using device: {device}") 
    model = CNNModel().to(device)
    model.load_state_dict(model_state_dict)
    _, test_loader = get_mnist_loaders()

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total}


#uncompressed function for the client model without compression
# Add this entire new function to the end of your client.py file


##ONLY FOR EDGE METRICS THIS IS DONE!
def train_one_round_baseline(model_state_dict, client_id=0, results_dir="./results",bw_mbps=None):
    """
    A simplified training round that ALWAYS sends the full, uncompressed model.
    """
    import time, torch
    from model import CNNModel
    from dataset import get_mnist_loaders
    from instrumentation import write_event, model_bytes_from_state

    if bw_mbps is not None:
        write_event(results_dir, f"client{client_id}", "bandwidth_mbps", bw_mbps, "Mbps")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    model.load_state_dict(model_state_dict)
    train_loader, _ = get_mnist_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    model.train()
    for _ in range(1): #
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    train_time_s = time.time() - t0
    
    state_to_send = model.state_dict()
    size_full = model_bytes_from_state(state_to_send)
    
    # Log everything as a "full" decision
    write_event(results_dir, f"client{client_id}", "decision", "full", "")
    write_event(results_dir, f"client{client_id}", "model_size_before_bytes", size_full, "bytes")
    write_event(results_dir, f"client{client_id}", "model_size_after_bytes", size_full, "bytes")

    return model.state_dict(), len(train_loader.dataset), {"decision": "full","client_id": client_id}