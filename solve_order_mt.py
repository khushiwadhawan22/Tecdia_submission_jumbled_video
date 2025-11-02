import os, json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PAIRS_CSV = "shortlist/pair_scores.csv"
N_FRAMES  = 300           # adjust if needed
BEAM      = 16            # beam width (try 8..12)
TOP_DEG   = 12             # use top-k outgoing edges per node during expansion to prune
NUM_WORKERS = max(2, min(os.cpu_count() or 4, 12))

OUT_INIT  = "output/order_initial.txt"
OUT_REFIN = "output/order_refined.txt"
LOG_JSON  = "logs/stage4_solver.json"

def load_graph():
    """Load pair_scores.csv and build adjacency lists sorted by score desc."""
    df = pd.read_csv(PAIRS_CSV)
    adj = {i: [] for i in range(1, N_FRAMES+1)}
    # aggregate by i, keeping best j edges
    for _, r in df.iterrows():
        i = int(r.i); j = int(r.j); s = float(r.blend_score)
        adj[i].append((j, s))
    for i in adj:
        adj[i].sort(key=lambda x: x[1], reverse=True)
    return adj

def pick_start(adj, top_in=5):
    """Pick a start frame with weakest incoming support (sum of top_in incoming edges)."""
    incoming = np.zeros(N_FRAMES+1, dtype=np.float32)
    for i in adj:
        for j, s in adj[i][:top_in]:
            incoming[j] += s
    # choose minimum incoming score
    start = int(np.argmin(incoming[1:])) + 1
    return start, incoming

def beam_search(adj, start, beam=BEAM, top_deg=TOP_DEG):
    """
    Beam search over sparse k-NN graph.
    State = (total_score, path_list, used_set).
    We prune expansions to the top 'top_deg' outgoing neighbors to keep it fast.
    """
    init = (0.0, [start], {start})
    frontier = [init]

    steps = 0
    while True:
        steps += 1
        # finished?
        if len(frontier[0][1]) == N_FRAMES:
            break

        new_frontier = []
        for total, path, used in frontier:
            last = path[-1]
            # consider only a few best outgoing edges to prune branching
            for j, s in adj[last][:top_deg]:
                if j in used:
                    continue
                new_total = total + s
                new_path = path + [j]
                # NOTE: set() copy is cheap here because beam and branching are small
                new_used = set(used)
                new_used.add(j)
                new_frontier.append((new_total, new_path, new_used))

        if not new_frontier:
            # fallback: append any unused node with best average incoming (rare)
            unused = [x for x in range(1, N_FRAMES+1) if x not in frontier[0][2]]
            if not unused: break
            new_frontier = [(frontier[0][0], frontier[0][1] + [unused[0]], frontier[0][2] | {unused[0]} )]

        # keep best 'beam' partial paths
        new_frontier.sort(key=lambda x: x[0], reverse=True)
        frontier = new_frontier[:beam]

    # return best path
    frontier.sort(key=lambda x: x[0], reverse=True)
    best_total, best_path, _ = frontier[0]
    return best_path, best_total, steps

def build_edge_lookup(adj):
    """Map (i,j) -> score for O(1) edge lookups in 2-opt and AAS."""
    m = {}
    for i in adj:
        for j, s in adj[i]:
            m[(i, j)] = s
    return m

def two_opt(order, edge_score, passes=2, window=20):
    """
    Simple 2-opt on directed scores:
    try reversing middle segments if it increases total adjacent score.
    """
    arr = order[:]
    n = len(arr)
    for _ in range(passes):
        improved = False
        for i in range(n - 3):
            for k in range(i + 2, min(i + window, n - 1)):
                a, b = arr[i],   arr[i+1]
                c, d = arr[k],   arr[k+1]
                old = edge_score.get((a,b), 0.0) + edge_score.get((c,d), 0.0)
                new = edge_score.get((a,c), 0.0) + edge_score.get((b,d), 0.0)
                if new > old:
                    arr[i+1:k+1] = reversed(arr[i+1:k+1])
                    improved = True
        if not improved:
            break
    return arr

def avg_adj_score(seq, edge_score):
    if len(seq) < 2:
        return 0.0
    s = 0.0
    for i in range(len(seq)-1):
        s += edge_score.get((seq[i], seq[i+1]), 0.0)
    return s / (len(seq)-1)

def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    adj = load_graph()
    start, incoming = pick_start(adj, top_in=5)

    order_init, total_init, steps = beam_search(adj, start, BEAM, TOP_DEG)

    edge_score = build_edge_lookup(adj)
    aas_init = avg_adj_score(order_init, edge_score)

    order_ref = two_opt(order_init, edge_score, passes=3, window=30)
    aas_ref  = avg_adj_score(order_ref, edge_score)

    # Save orders
    with open(OUT_INIT, "w") as f:
        f.write("\n".join(map(str, order_init)))
    with open(OUT_REFIN, "w") as f:
        f.write("\n".join(map(str, order_ref)))

    # Log
    with open(LOG_JSON, "w") as f:
        json.dump({
            "start_frame": start,
            "beam_width": BEAM,
            "top_out_degree": TOP_DEG,
            "steps": steps,
            "aas_initial": aas_init,
            "aas_refined": aas_ref
        }, f, indent=2)

    print(f"Start frame: {start}")
    print(f"AAS initial: {aas_init:.4f}")
    print(f"AAS refined: {aas_ref:.4f}")
    print(f"Saved:\n  - {OUT_INIT}\n  - {OUT_REFIN}\n  - {LOG_JSON}")

if __name__ == "__main__":
    main()
