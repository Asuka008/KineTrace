import math


def sample_indices(total, count):
    if total < 1 or count < 1:
        raise ValueError("Frame count must be positive")
    count = min(total, count)
    if count == 1:
        return [total // 2]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def read_clip(path, count, max_side=448):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_indices(total, count)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS for {path}")
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index} of {path}")
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1.0:
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames, {"total_num_frames": total, "fps": fps, "frames_indices": indices}
    finally:
        capture.release()
