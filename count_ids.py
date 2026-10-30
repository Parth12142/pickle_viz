import json, collections
from detection.players import in_roi

cache = json.load(open("cache/players.json"))
frames_seen = collections.Counter()
for dets in cache:
    for d in dets:
        if in_roi(d["box"]):
            frames_seen[d["id"]] += 1

print("total frames:", len(cache))
for pid, n in sorted(frames_seen.items()):
    print(f"ID {pid}: seen in {n} frames")