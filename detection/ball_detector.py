import os
import time
import cv2
import numpy as np
from ultralytics import YOLO


def run(frames, fps, size, cfg):
    weights = cfg.get("weights", "weights/ball_best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Ball weights not found: {weights}\n"
            "Download best.pt from Colab and put it there, "
            "or set modules.ball.enabled: false in config.yaml"
        )
    imgsz = cfg.get("imgsz", 1312)
    conf = cfg.get("conf", 0.15)
    max_jump_px = cfg.get("max_jump_px", 75.0)

    t = time.perf_counter()
    model = YOLO(weights)
    model.predict(frames[0], imgsz=imgsz, verbose=False)
    load_s = time.perf_counter() - t

    ids = [k for k, v in model.names.items() if str(v).lower() == "ball"]
    if not ids:
        raise ValueError(f"No class called 'ball' in this model: {model.names}")
    ball_id = ids[0]

    raw_candidates = []
    t = time.perf_counter()

    # Step 1: Detect candidate ball boxes across all frames
    for f in frames:
        r = model.predict(f, imgsz=imgsz, conf=conf, verbose=False)[0]
        cands = []
        for box in r.boxes:
            if int(box.cls[0]) != ball_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            if bw > 50 or bh > 50 or bw < 2 or bh < 2:
                continue
            c = float(box.conf[0])
            cands.append({"x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0, "conf": c})
        raw_candidates.append(cands)

    # Step 2: Track continuous trajectory using motion prediction & velocity gating
    tracked_points = [None] * len(frames)
    last_pos = None
    velocity = [0.0, 0.0]

    for i, cands in enumerate(raw_candidates):
        if not cands:
            last_pos = None
            velocity = [0.0, 0.0]
            continue

        if last_pos is None:
            best = max(cands, key=lambda c: c["conf"])
            tracked_points[i] = best
            last_pos = best
            velocity = [0.0, 0.0]
        else:
            pred_x = last_pos["x"] + velocity[0]
            pred_y = last_pos["y"] + velocity[1]

            best_cand = None
            best_cost = float("inf")

            for c in cands:
                dist = np.hypot(c["x"] - pred_x, c["y"] - pred_y)
                cost = dist - 25.0 * c["conf"]
                if dist < max_jump_px and cost < best_cost:
                    best_cost = cost
                    best_cand = c

            if best_cand is not None:
                new_vx = best_cand["x"] - last_pos["x"]
                new_vy = best_cand["y"] - last_pos["y"]
                velocity = [0.7 * new_vx + 0.3 * velocity[0], 0.7 * new_vy + 0.3 * velocity[1]]
                tracked_points[i] = best_cand
                last_pos = best_cand
            else:
                last_pos = None
                velocity = [0.0, 0.0]

    # Step 3: Interpolate small gaps (1 to 5 missed frames) along trajectory
    interpolated = list(tracked_points)
    i = 0
    n = len(interpolated)
    while i < n:
        if interpolated[i] is None:
            gap_start = i
            while i < n and interpolated[i] is None:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_start > 0 and gap_end < n and gap_len <= 5:
                p1 = interpolated[gap_start - 1]
                p2 = interpolated[gap_end]
                total_dist = np.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])
                if total_dist <= max_jump_px * (gap_len + 1):
                    for k in range(gap_start, gap_end):
                        alpha = (k - gap_start + 1) / (gap_len + 1)
                        interpolated[k] = {
                            "x": p1["x"] + alpha * (p2["x"] - p1["x"]),
                            "y": p1["y"] + alpha * (p2["y"] - p1["y"]),
                            "conf": 0.5,
                            "interpolated": True,
                        }
        else:
            i += 1

    # Step 4: Trajectory smoothing (Gaussian/moving window)
    smoothed = list(interpolated)
    window = 3
    for k in range(len(interpolated)):
        if interpolated[k] is not None:
            nbrs = [
                interpolated[j]
                for j in range(max(0, k - window), min(len(interpolated), k + window + 1))
                if interpolated[j] is not None
            ]
            if len(nbrs) >= 3:
                smoothed[k] = {
                    "x": float(np.mean([p["x"] for p in nbrs])),
                    "y": float(np.mean([p["y"] for p in nbrs])),
                    "conf": interpolated[k]["conf"],
                }

    process_s = time.perf_counter() - t
    found = sum(1 for d in smoothed if d is not None)

    return {
        "data": {
            "points": smoothed,
            "trail": cfg.get("trail", 18),
            "max_jump_px": max_jump_px,
        },
        "timing": {"load_s": load_s, "process_s": process_s},
        "stats": {
            "frames_with_ball": found,
            "detection_rate": round(found / len(frames), 3),
        },
    }


def draw(frame, i, data):
    pts = data["points"]
    trail_len = data.get("trail", 18)
    max_jump = data.get("max_jump_px", 75.0)

    start_idx = max(0, i - trail_len)
    trail_segment = pts[start_idx : i + 1]

    # Draw gradient glowing trail
    glow = frame.copy()
    has_segments = False

    for k in range(len(trail_segment) - 1):
        p1 = trail_segment[k]
        p2 = trail_segment[k + 1]
        if p1 is None or p2 is None:
            continue

        dist = np.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])
        if dist > max_jump * 1.5:
            continue

        progress = (k + 1) / len(trail_segment)
        pt1 = (int(p1["x"]), int(p1["y"]))
        pt2 = (int(p2["x"]), int(p2["y"]))

        # Outer soft glow on overlay
        glow_color = (0, int(180 + 75 * progress), int(220 + 35 * progress))
        cv2.line(glow, pt1, pt2, glow_color, int(progress * 6) + 2, lineType=cv2.LINE_AA)

        # Crisp inner core line on frame
        core_color = (200, 255, 255)
        cv2.line(frame, pt1, pt2, core_color, max(1, int(progress * 2)), lineType=cv2.LINE_AA)
        has_segments = True

    if has_segments:
        cv2.addWeighted(glow, 0.40, frame, 0.60, 0, frame)

    # Draw high-visibility reticle around current ball position
    curr = pts[i]
    if curr is not None:
        cx, cy = int(curr["x"]), int(curr["y"])
        # High contrast outer circle
        cv2.circle(frame, (cx, cy), 11, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 11, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        # Bright center dot
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
