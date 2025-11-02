import os, shutil

FULL = "frames_full"
OUT  = "frames_out"
ORDER_FILE = "output/order_phase_ls.txt"


def main():
    # Load refined order from file
    with open(ORDER_FILE) as f:
        idxs = [int(x.strip()) for x in f if x.strip()]
    print(f"Loaded {len(idxs)} frame indices from {ORDER_FILE}")

    # Create output folder
    os.makedirs(OUT, exist_ok=True)

    # Copy full-resolution frames in correct order
    for n, idx in enumerate(idxs, 1):
        src = os.path.join(FULL, f"{idx:05d}.jpg")
        dst = os.path.join(OUT,  f"{n:05d}.jpg")
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        if n % 50 == 0:
            print(f"Copied {n}/{len(idxs)} frames...")

    print(f"\n Wrote {len(idxs)} frames to {OUT}")

if __name__ == "__main__":
    main()
