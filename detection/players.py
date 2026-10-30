import json, os
import cv2
import numpy as np
from ultralytics import YOLO
from video_io import read_video, write_video
import collections
from render.overlay import draw_player

CLIP = "data/clips/match.mp4"
CACHE = "cache/players.json"

# Rough playing area (pixels, for 1302x740). Players' FEET must be inside it.
# This removes referees and spectators. Adjust after checking outputs/roi_check.png.
ROI = np.array([(300, 120), (960, 120), (1250, 735), (60, 735)], dtype=np.int32)

def detect_players(frames):
    model = YOLO("yolo11n.pt")            # downloads automatically the first time
    cache = []
    for frame in frames:
        r = model.track(frame, persist=True, classes=[0],      # class 0 = person
                        conf=0.3, tracker="bytetrack.yaml", verbose=False)[0]
        dets = []
        if r.boxes.id is not None:
            for box, tid, conf in zip(r.boxes.xyxy.tolist(),
                                      r.boxes.id.tolist(),
                                      r.boxes.conf.tolist()):
                dets.append({"id": int(tid), "box": box, "conf": conf})
        cache.append(dets)
    return cache

def in_roi(box):
    x1, y1, x2, y2 = box
    foot = ((x1 + x2) / 2, y2)             # bottom-centre of the box = feet
    return cv2.pointPolygonTest(ROI, foot, False) >= 0

def keep_real_tracks(cache, min_frames=30):
    counts = collections.Counter(d["id"] for dets in cache for d in dets if in_roi(d["box"]))
    keep = {i for i, n in counts.items() if n >= min_frames}
    return [[d for d in dets if d["id"] in keep] for dets in cache]

def draw(frames, cache):
    out = []
    for frame, dets in zip(frames, cache):
        frame = frame.copy()
        for d in dets:
            if in_roi(d["box"]):
                draw_player(frame, d["box"], d["id"])
        out.append(frame)
    return out

if __name__ == "__main__":
    frames, fps, size = read_video(CLIP)
    if os.path.exists(CACHE):              # reuse cache, don't re-run YOLO
        cache = json.load(open(CACHE))
    else:
        cache = detect_players(frames)
        json.dump(cache, open(CACHE, "w"))
    cv2.imwrite("outputs/roi_check.png", draw(frames[:1], cache[:1])[0])
    write_video(draw(frames, keep_real_tracks(cache)), "outputs/players.mp4", fps, size)
    print("saved outputs/players.mp4 and outputs/roi_check.png")