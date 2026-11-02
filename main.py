import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------- settings ----------------
# best.pt was trained with broken class names, so use a pretrained COCO model
# (knows "person" and "sports ball"; downloads automatically on first run)
MODEL_PATH = "yolo11m.pt"
VIDEO_IN = "gym_down_corner_elevated2.mp4"
VIDEO_OUT = "testcv.mp4"
SHOW = True             # show live preview window

IMGSZ = 1280            # bigger input = better for small objects (slower)
PERSON_CONF = 0.4       # min confidence for players / people
BALL_CONF = 0.1         # ball is tiny, accept lower confidence
USE_TTA = False         # test-time augmentation: a bit more accurate, ~2-3x slower

USE_TILES = True        # if the ball is missed, re-scan the frame as zoomed-in tiles
TILE_GRID = 2           # 2x2 tiles
TILE_OVERLAP = 0.2      # tiles overlap so the ball isn't cut in half at the edges

MAX_MISSED = 8          # frames to keep guessing the ball position after losing it
MAX_JUMP = 0.15         # max ball movement per frame (fraction of frame width), rejects false balls

# ---------------- setup ----------------
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH)
BALL_IDS = {i for i, n in model.names.items() if "ball" in n.lower()}
PERSON_IDS = {i for i, n in model.names.items()
              if n.lower() in ("person", "player", "goalkeeper", "referee")}
print(f"device={device} classes={model.names}")
if not BALL_IDS:
    print("WARNING: no class named 'ball' in the model, ball logic will be skipped")


def run_model(img):
    """Run the model on an image, return list of (x1, y1, x2, y2, conf, cls)."""
    r = model.predict(img, imgsz=IMGSZ, conf=BALL_CONF, iou=0.5,
                      augment=USE_TTA, device=device, verbose=False)[0]
    return [(*b.xyxy[0].tolist(), float(b.conf), int(b.cls)) for b in r.boxes]


def ball_from_tiles(frame):
    """Zoom into overlapping tiles so the small ball covers more pixels."""
    H, W = frame.shape[:2]
    th = min(H, int(H / TILE_GRID * (1 + TILE_OVERLAP)))
    tw = min(W, int(W / TILE_GRID * (1 + TILE_OVERLAP)))
    found = []
    for gy in range(TILE_GRID):
        for gx in range(TILE_GRID):
            y0 = min(int(gy * H / TILE_GRID), H - th)
            x0 = min(int(gx * W / TILE_GRID), W - tw)
            for x1, y1, x2, y2, c, k in run_model(frame[y0:y0 + th, x0:x0 + tw]):
                if k in BALL_IDS:
                    found.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0, c))
    return found


def center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


class BallTracker:
    """Keeps one ball: rejects far-away false positives and fills short gaps."""

    def __init__(self):
        self.pos, self.vel, self.size, self.missed = None, np.zeros(2), np.array([10.0, 10.0]), 0

    def update(self, cands, W):
        # only accept candidates near where we expect the ball (unless very confident)
        if self.pos is not None:
            pred = self.pos + self.vel
            limit = MAX_JUMP * W * (self.missed + 1)
            cands = [c for c in cands if np.linalg.norm(center(c) - pred) < limit or c[4] > 0.5]

        if cands:
            best = max(cands, key=lambda c: c[4])
            new = center(best)
            if self.pos is not None:
                self.vel = 0.7 * self.vel + 0.3 * (new - self.pos) / (self.missed + 1)
            self.pos, self.missed = new, 0
            self.size = np.array([best[2] - best[0], best[3] - best[1]])
            return best[:4], best[4], False

        # no detection: predict from last velocity for a few frames
        if self.pos is None:
            return None
        self.missed += 1
        if self.missed > MAX_MISSED:
            self.pos, self.vel = None, np.zeros(2)
            return None
        self.pos = self.pos + self.vel
        (cx, cy), (w, h) = self.pos, self.size
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 0.0, True


def draw(frame, box, text, color, thick=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    cv2.putText(frame, text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# ---------------- main loop ----------------
cap = cv2.VideoCapture(VIDEO_IN)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30, (W, H))
ball = BallTracker()
n = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    n += 1

    dets = run_model(frame)

    # people / players
    for x1, y1, x2, y2, c, k in dets:
        if k in PERSON_IDS and c >= PERSON_CONF:
            draw(frame, (x1, y1, x2, y2), f"{model.names[k]} {c:.2f}", (0, 255, 0))

    # ball
    if BALL_IDS:
        cands = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c, k in dets if k in BALL_IDS]
        if not cands and USE_TILES:
            cands = ball_from_tiles(frame)
        res = ball.update(cands, W)
        if res:
            box, c, guessed = res
            if guessed:
                draw(frame, box, "ball?", (0, 165, 255), 1)    # orange = predicted
            else:
                draw(frame, box, f"ball {c:.2f}", (0, 255, 255))  # yellow = detected

    out.write(frame)
    if n % 50 == 0:
        print(f"{n}/{total} frames")
    if SHOW:
        cv2.imshow("Soccer Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"saved {VIDEO_OUT}")
