# edgeComputing/compression.py
import io, pickle, gzip
import torch


def topk_sparsify(update, k=0.1):
    """
    Keep top-k fraction of weights (largest absolute values)

    Parameters:
    update: dict of layer deltas
    k: fraction (0 < k <= 1) or dict per layer {layer_name: fraction}

    Returns:
    sparse_update: dict with same keys, top-k values retained
    """
    sparse_update = {}
    for key, tensor in update.items():
        flat = tensor.view(-1)

        # Determine fraction for this layer
        layer_k = k[key] if isinstance(k, dict) else k
        num_topk = max(1, int(layer_k * flat.numel()))

        # Get top-k indices by absolute value
        values, idx = torch.topk(flat.abs(), num_topk)
        mask = torch.zeros_like(flat)
        mask[idx] = 1
        sparse_update[key] = (flat * mask).view(tensor.shape)

        print(f"[Compression] Layer {key}: kept {num_topk}/{flat.numel()} weights ({layer_k * 100:.2f}%)")
    return sparse_update


def should_skip(bandwidth, update_size, threshold=0.05):
    """
    Decide whether client should skip sending update based on bandwidth

    Parameters:
    bandwidth: available bandwidth in MB
    update_size: size of full update in MB
    threshold: minimum ratio required to send

    Returns:
    bool: True if update should be skipped
    """
    if bandwidth <= 0 or update_size <= 0:
        print("[Compression] Warning: bandwidth or update_size <= 0. Will skip update.")
        return True

    r_t = bandwidth / update_size
    print(f"[Compression] Bandwidth ratio: {r_t:.4f} (threshold {threshold})")
    return r_t < threshold


# ---------------- Example usage ---------------- #
if __name__ == "__main__":
    # Dummy update tensor
    dummy_update = {
        "conv1.weight": torch.randn(32, 1, 3, 3),
        "fc1.weight": torch.randn(128, 64 * 7 * 7)
    }

    # Apply top-10% sparsification
    sparse = topk_sparsify(dummy_update, k=0.1)
    print("Sparse update keys:", list(sparse.keys()))

    # Decide skipping
    bandwidth = 0.02  # MB
    update_size = 1.0  # MB
    skip = should_skip(bandwidth, update_size)
    print("Should skip:", skip)

#append some code
# ---------- helpers to integrate with train_one_round ----------

import io, pickle

def compress_state_dict(state_dict, method="topk", topk_ratio=0.5):
    """
    Compress a PyTorch state_dict and return bytes (pickle).
    Uses topk_sparsify when method=="topk".
    """
    # Convert state_dict tensors to CPU tensors
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}

    if method == "topk":
        sparse = topk_sparsify(cpu_state, k=topk_ratio)
        # Save sparse dict as bytes for size measurement / transport
        buf = io.BytesIO()
        # Use torch.save so types are preserved
        torch.save(sparse, buf)
        #return buf.getvalue()
        return gzip.compress(buf.getvalue())
    else:
        # fallback: full state bytes
        buf = io.BytesIO()
        torch.save(cpu_state, buf)
        return buf.getvalue()


def decompress_state_bytes(bts):
    """
    Decompress bytes produced by compress_state_dict back to a state-dict-like object.
    Returns the object loaded by torch.load (sparse representation).
    """
    buf = io.BytesIO(bts)
    obj = torch.load(buf)
    return obj


def bytes_size(xbytes):
    """helper to get length in bytes"""
    return len(xbytes)


# compatibility wrapper
def decompress_state_dict(bts):
    return decompress_state_bytes(bts)
