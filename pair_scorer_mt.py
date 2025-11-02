import os, csv, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, cv2, pandas as pd
from skimage.metrics import structural_similarity as ssim

# ====== CONFIG ======
SMALL_DIR      = "frames_small"
SHORTLIST_TOP  = "shortlist/K20_top.csv"   # or K30_top.csv if you widened K
OUT_CSV        = "shortlist/pair_scores.csv"
LOG_JSON       = "logs/stage3_pair_scorer.json"
RESIZE_WIDTH   = 640
NUM_WORKERS    = max(2, min(os.cpu_count() or 4, 12))  # sensible default

# Blend weights (try 0.7/0.3 if you want stronger SSIM)
W_SSIM, W_INL = 0.6, 0.4

# ORB + matcher
ORB = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def read_gray_resize(path, w=RESIZE_WIDTH):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    if im.shape[1] != w:
        h = int(round(im.shape[0] * (w / im.shape[1])))
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    return im

def orb_inliers(a_gray, b_gray):
    ka, da = ORB.detectAndCompute(a_gray, None)
    kb, db = ORB.detectAndCompute(b_gray, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    matches = BF.knnMatch(da, db, k=2)
    good = [m for m, n in matches if n is not None and m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    ptsA = np.float32([ka[m.queryIdx].pt for m in good])
    ptsB = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
    return int(mask.sum()) if mask is not None else 0

def ssim_01(a_gray, b_gray):
    v = ssim(a_gray, b_gray, data_range=255)
    return float(max(0.0, min(1.0, v)))


def main():
    os.makedirs("logs", exist_ok=True)

    # ---- Load shortlist ----
    df = pd.read_csv(SHORTLIST_TOP).sort_values("frame_id")
    N = len(df)
    neighbors = {int(r.frame_id): [int(v) for k, v in r.items() if k.startswith("n")] for _, r in df.iterrows()}

    # ---- Preload all 640w grayscale frames once ----
    print(f"Preloading {N} frames from {SMALL_DIR} ...")
    frames = [None] * (N + 1)  # 1-based
    for i in range(1, N + 1):
        path = os.path.join(SMALL_DIR, f"{i:05d}.jpg")
        frames[i] = read_gray_resize(path)

    # ---- Build full pair list from shortlist ----
    pairs = [(i, j) for i in range(1, N + 1) for j in neighbors[i]]

    # ---- Make pairs symmetric (ensure both i->j and j->i exist) ----
    pair_set = set(pairs)
    for i, j in list(pairs):
        if (j, i) not in pair_set:
            pair_set.add((j, i))
    pairs = list(pair_set)

    print(f"Scoring {len(pairs)} pairs using {NUM_WORKERS} threads")

    # ---- PASS 1: ORB inliers (parallel) to get normalization cap ----
    def worker_inliers(p):
        i, j = p
        return (i, j, orb_inliers(frames[i], frames[j]))

    inlier_map = {}
    raw_inliers = []
    print("Pass 1: ORB-RANSAC inliers ...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(worker_inliers, p): p for p in pairs}
        step = max(1, len(futures) // 100)
        for k, fut in enumerate(as_completed(futures), 1):
            i, j, inl = fut.result()
            inlier_map[(i, j)] = inl
            raw_inliers.append(inl)
            if k % (step * 10) == 0:
                print(f"  ... {k}/{len(pairs)} pairs")

    cap = max(50, int(np.percentile(raw_inliers, 95))) if raw_inliers else 200
    print(f"Inlier cap (95th pct): {cap}")

    # ---- PASS 2: SSIM + blend (parallel), reuse inliers from pass 1 ----
    def worker_blend(p):
        i, j = p
        s = ssim_01(frames[i], frames[j])
        inl = inlier_map[(i, j)]
        inl_norm = min(inl / cap, 1.0)
        score = W_SSIM * s + W_INL * inl_norm
        return (i, j, s, inl, inl_norm, score)

    results = []
    print("Pass 2: SSIM + blend ...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(worker_blend, p): p for p in pairs}
        step = max(1, len(futures) // 100)
        for k, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if k % (step * 10) == 0:
                print(f"  ... {k}/{len(pairs)} pairs")

    # ---- Make scores symmetric by averaging i->j and j->i ----
    tmp = {}
    for (i, j, s, inl, inn, sc) in results:
        tmp[(i, j)] = (s, inl, inn, sc)

    sym_rows = []
    seen = set()
    for (i, j), (s1, in1, inn1, sc1) in tmp.items():
        if (i, j) in seen:
            continue
        if (j, i) in tmp:
            s2, in2, inn2, sc2 = tmp[(j, i)]
            s   = 0.5 * (s1 + s2)
            inl = int(round(0.5 * (in1 + in2)))
            inn = 0.5 * (inn1 + inn2)
            sc  = 0.5 * (sc1 + sc2)
            seen.add((j, i))
        else:
            s, inl, inn, sc = s1, in1, inn1, sc1
        sym_rows.append((i, j, s, inl, inn, sc))
        seen.add((i, j))

    # ---- Save outputs ----
    os.makedirs("shortlist", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i","j","ssim","inliers","inliers_norm","blend_score"])
        w.writerows(sym_rows)

    os.makedirs("logs", exist_ok=True)
    with open(LOG_JSON, "w") as f:
        json.dump({
            "frames": N,
            "pairs_evaluated": len(sym_rows),
            "inliers_cap": cap,
            "threads": NUM_WORKERS,
            "weights": {"ssim": W_SSIM, "inliers_norm": W_INL},
            "notes": "Parallel; pairs made symmetric; scores averaged for i<->j."
        }, f, indent=2)

    print("\n Done!")
    print(f"Saved: {OUT_CSV}  (rows={len(sym_rows)})")
    print(f"Log:   {LOG_JSON}")
    print(f"Threads used: {NUM_WORKERS}, cap={cap}")

if __name__ == "__main__":
    main()
