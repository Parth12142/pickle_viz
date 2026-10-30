import collections
import time
import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_ROI = [(0.12, 0.10), (0.88, 0.10), (0.98, 0.99), (0.02, 0.99)]
PLAYER_COLORS = [
    (0, 165, 255),   # P1: Orange
    (80, 80, 255),   # P2: Coral Red
    (0, 220, 0),     # P3: Lime Green
    (255, 200, 0),   # P4: Cyan/Sky Blue
    (200, 80, 200),  # P5: Violet
    (0, 255, 255),   # P6: Yellow
]


def _center(box):
    return (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))


def _foot_in_roi(box, roi):
    if roi is None:
        return True
    foot = (float((box[0] + box[2]) / 2), float(box[3]))
    return cv2.pointPolygonTest(roi, foot, False) >= 0


class StablePlayerTracker:
    """Maintains consistent canonical player IDs (P1, P2, P3, P4) across occlusions,

    using spatial proximity matching, trajectory smoothing, and court constraints.
    """

    def __init__(self, max_players=4, max_lost_frames=60, smooth_alpha=0.75):
        self.max_players = max_players
        self.max_lost_frames = max_lost_frames
        self.smooth_alpha = smooth_alpha
        self.slots = {}  # slot_id: dict(box, smooth_box, lost, hits, history)
        self.next_slot_id = 1

    def update(self, detections, h, w, roi=None):
        # 1. Filter out candidate boxes that are out-of-bounds or tiny background artifacts
        candidates = []
        for d in detections:
            box = d["box"]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bh < 0.07 * h or bw < 0.02 * w:
                continue
            if not _foot_in_roi(box, roi):
                continue
            candidates.append(d)

        # 2. Match candidate detections to currently tracked slots
        unmatched_dets = list(range(len(candidates)))
        assigned = {}

        max_dist = 0.25 * np.hypot(w, h)

        for slot_id, slot in list(self.slots.items()):
            best_idx = None
            best_dist = float("inf")
            c_slot = _center(slot["smooth_box"])

            for idx in unmatched_dets:
                det_box = candidates[idx]["box"]
                c_det = _center(det_box)
                dist = np.hypot(c_slot[0] - c_det[0], c_slot[1] - c_det[1])
                if dist < best_dist and dist < max_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                assigned[slot_id] = candidates[best_idx]
                unmatched_dets.remove(best_idx)
                slot["lost"] = 0
                slot["hits"] += 1
            else:
                slot["lost"] += 1

        # 3. Update existing matched slots with smoothed boxes
        for slot_id, det in assigned.items():
            slot = self.slots[slot_id]
            curr_box = det["box"]
            slot["box"] = curr_box
            s_box = slot["smooth_box"]
            a = self.smooth_alpha
            new_smooth = [a * c + (1.0 - a) * s for c, s in zip(curr_box, s_box)]
            slot["smooth_box"] = new_smooth

            # Store foot position in history for footstep trail
            cx = int((new_smooth[0] + new_smooth[2]) / 2)
            cy = int(new_smooth[3])
            slot.setdefault("history", []).append((cx, cy))
            if len(slot["history"]) > 20:
                slot["history"].pop(0)

        # 4. Remove stale slots that have been lost for too long
        for slot_id in list(self.slots.keys()):
            if self.slots[slot_id]["lost"] > self.max_lost_frames:
                del self.slots[slot_id]

        # 5. Initialize new slots for remaining unmatched candidates (up to max_players)
        for idx in unmatched_dets:
            if len(self.slots) < self.max_players:
                box = candidates[idx]["box"]
                used_ids = set(self.slots.keys())
                avail_id = next((i for i in range(1, self.max_players + 1) if i not in used_ids), None)
                if avail_id is None:
                    avail_id = self.next_slot_id
                    self.next_slot_id += 1
                cx = int((box[0] + box[2]) / 2)
                cy = int(box[3])
                self.slots[avail_id] = {
                    "box": box,
                    "smooth_box": list(box),
                    "lost": 0,
                    "hits": 1,
                    "history": [(cx, cy)],
                }

        # Return active players in current frame
        active = []
        for slot_id, slot in sorted(self.slots.items()):
            if slot["lost"] <= 2:
                active.append({
                    "id": slot_id,
                    "box": slot["smooth_box"],
                    "history": list(slot.get("history", [])),
                })
        return active


def run(frames, fps, size, cfg):
    w, h = size
    t_start = time.perf_counter()

    model = YOLO(cfg.get("model", "yolo11n.pt"))
    model.predict(frames[0], classes=[0], verbose=False)
    load_s = time.perf_counter() - t_start

    roi = None
    if cfg.get("use_roi", True):
        pts = cfg.get("roi", DEFAULT_ROI)
        roi = np.array([(int(x * w), int(y * h)) for x, y in pts], dtype=np.int32)

    max_players = cfg.get("max_players", 4)
    tracker = StablePlayerTracker(
        max_players=max_players,
        max_lost_frames=int(fps * 2.0),
        smooth_alpha=0.75,
    )

    t_proc = time.perf_counter()
    data = []
    seen = collections.Counter()
    model_ms = []

    for f in frames:
        r = model(f, classes=[0], conf=cfg.get("conf", 0.28), verbose=False)[0]
        model_ms.append(sum(r.speed.values()))

        dets = []
        for box in r.boxes:
            b = box.xyxy[0].tolist()
            dets.append({"box": b, "conf": float(box.conf[0])})

        active = tracker.update(dets, h, w, roi)
        for p in active:
            seen[p["id"]] += 1
        data.append(active)

    process_s = time.perf_counter() - t_proc

    return {
        "data": data,
        "timing": {"load_s": load_s, "process_s": process_s},
        "stats": {
            "players_tracked": len(seen),
            "frames_per_track": {int(i): int(seen[i]) for i in sorted(seen.keys())},
            "mean_players_per_frame": round(float(np.mean([len(d) for d in data])), 2),
            "model_only_ms_per_frame": round(float(np.mean(model_ms)), 1),
        },
    }


def draw(frame, i, data):
    for d in data[i]:
        x1, y1, x2, y2 = map(int, d["box"])
        tid = d["id"]
        color = PLAYER_COLORS[(tid - 1) % len(PLAYER_COLORS)]

        cx = (x1 + x2) // 2
        bw = max(20, x2 - x1)

        # 1. Subtle footstep trail
        history = d.get("history", [])
        for idx in range(len(history) - 1):
            pt1 = history[idx]
            pt2 = history[idx + 1]
            alpha_step = (idx + 1) / len(history)
            cv2.line(frame, pt1, pt2, color, max(1, int(alpha_step * 2)), cv2.LINE_AA)

        # 2. Semi-transparent foot ellipse
        axes = (int(bw * 0.70), max(6, int(bw * 0.22)))
        overlay = frame.copy()
        cv2.ellipse(overlay, (cx, y2), axes, 0, 0, 360, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.ellipse(frame, (cx, y2), axes, 0, 0, 360, color, 2, lineType=cv2.LINE_AA)

        # 3. Modern Player Pill Badge
        label = f"P{tid}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        badge_w = tw + 14
        badge_h = th + 10
        bx1 = cx - badge_w // 2
        by1 = max(5, y1 - badge_h - 6)
        bx2 = bx1 + badge_w
        by2 = by1 + badge_h

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (25, 25, 25), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, label, (bx1 + 7, by2 - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
