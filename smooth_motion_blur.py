import os, cv2, glob, numpy as np
from tqdm import tqdm

SRC = "frames_out"
DST = "frames_finalsmooth"
os.makedirs(DST, exist_ok=True)

paths = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
if not paths:
    raise RuntimeError("No frames found in frames_out/. Generate them first.")

print(f"[info] loaded {len(paths)} frames from {SRC}")

# Parameters
alpha = 0.6          # blend ratio for motion blur
flow_scale = 0.4     # resize factor for optical flow
blur_strength = 7     # bilateral blur strength
stable_frames = []   # output frames

def resize_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    return cv2.resize(g, (int(w*flow_scale), int(h*flow_scale)))

def warp_blend(prev, curr, flow):
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.repeat(np.arange(w), h), np.tile(np.arange(h), w))).astype(np.float32)
    flow_map = flow_map.reshape(w, h, 2)
    flow_map[...,0] += flow[...,0]
    flow_map[...,1] += flow[...,1]
    return cv2.remap(prev, flow_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

prev = cv2.imread(paths[0])
stable_frames.append(prev)

for i in tqdm(range(1, len(paths))):
    nxt = cv2.imread(paths[i])
    g1 = resize_gray(prev)
    g2 = resize_gray(nxt)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    warped_prev = warp_blend(prev, nxt, flow)
    blended = cv2.addWeighted(warped_prev, alpha, nxt, 1 - alpha, 0)
    smoothed = cv2.bilateralFilter(blended, blur_strength, 50, 50)
    stable_frames.append(smoothed)
    prev = nxt

for i, frame in enumerate(stable_frames):
    cv2.imwrite(os.path.join(DST, f"{i+1:05d}.jpg"), frame)

print(f"[done] Motion-blur smoothed frames saved to {DST}")
