import os
import shutil
import subprocess
import cv2


def _find_ffmpeg():
    # 1. Check system PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Check Windows WinGet install locations
    winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_base):
        for root, _, files in os.walk(winget_base):
            if "ffmpeg.exe" in files:
                return os.path.join(root, "ffmpeg.exe")

    return "ffmpeg"


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, (w, h)


def write_video(frames, path, fps, size):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path.replace(".mp4", "_raw.mp4")
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in frames:
        out.write(f)
    out.release()

    ffmpeg_bin = _find_ffmpeg()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        tmp,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        path,
    ]

    try:
        subprocess.run(cmd, check=True)
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        # Fallback to direct mp4 file if ffmpeg is unavailable
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp, path)


if __name__ == "__main__":
    frames, fps, size = read_video("data/clips/match.mp4")
    print(f"{len(frames)} frames, {fps} FPS, {size}")
