import cv2
from ultralytics import YOLO
import time

model = YOLO(r"C:\Users\hp\runs\detect\train5\weights\best.pt")

url = "http://192.168.1.3:8080/video"

cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ─────────────────────────────────────────────────────────────
#  TUNABLE PARAMETERS  (adjust these to match your track/car)
# ─────────────────────────────────────────────────────────────
FRAME_W, FRAME_H      = 480, 360

# Zone thresholds (as fraction of frame width)
LEFT_ZONE             = 0.30   # border center < 30%  → FAR LEFT
RIGHT_ZONE            = 0.70   # border center > 70%  → FAR RIGHT
# 30–70% = centre zone (border is ahead/diagonal)

# Height thresholds (pixels) for border closeness
BORDER_VISIBLE_MIN    = 100    # below this → border is far away, ignore for steering
BORDER_CLOSE          = 200    # border is getting close → graduated correction
BORDER_DANGER         = 290    # border is very close    → sharp emergency turn
BORDER_CRITICAL       = 330    # border fills frame      → REVERSE (safety override)

# Area thresholds for shapes / arrows
SHAPE_AREA_MIN        = 6000   # circle / rectangle
ARROW_AREA_MIN        = 6000

# Cooldowns (seconds)
SHAPE_COOLDOWN        = 3.0
ARROW_COOLDOWN        = 2.0
STEER_COOLDOWN        = 0.4    # min time between printing a steer command

# How many frames of border history to keep for road-centre tracking
HISTORY_LEN           = 8
# ─────────────────────────────────────────────────────────────

frame_count         = 0
last_command        = ""
last_shape_time     = 0.0
last_arrow_time     = 0.0
last_steer_time     = 0.0

# Rolling history of detected border positions (centre_x values)
border_history      = []   # list of floats

frame_center        = FRAME_W / 2
left_zone_px        = FRAME_W * LEFT_ZONE
right_zone_px       = FRAME_W * RIGHT_ZONE


def emit(command: str):
    """Print a command only if it changed or enough time has passed."""
    global last_command, last_steer_time
    now = time.time()
    if command != last_command or (now - last_steer_time) > 2.0:
        print(command)
        last_command    = command
        last_steer_time = now


def road_centre_command(border_cx_list: list) -> str | None:
    """
    PRIMARY LOGIC — keep car in the middle of the road.

    Uses a rolling average of recent border positions to estimate
    where the road centre is, then gently corrects drift.

    The border marks one EDGE of the road. If the border is on the
    LEFT wall, the car should be somewhere to the RIGHT of that border.
    If on the RIGHT wall, the car should be to the LEFT.

    Returns a command string or None if no action needed.
    """
    if not border_cx_list:
        return None

    avg_border_cx = sum(border_cx_list) / len(border_cx_list)

    # How far is the border from the frame centre?
    offset = avg_border_cx - frame_center   # negative → border is left of centre

    # Dead-band: if the border is roughly centred, the road is straight — no steer needed
    if abs(offset) < FRAME_W * 0.08:
        return "ROAD CENTRE OK — straight ahead"

    # Border is to the LEFT → car is drifting left → steer RIGHT (gentle)
    if offset < 0:
        return "ROAD CENTRE CORRECTION → STEER RIGHT ~15°"

    # Border is to the RIGHT → car is drifting right → steer LEFT (gentle)
    return "ROAD CENTRE CORRECTION → STEER LEFT ~15°"


def security_border_command(largest_cx: float, largest_h: float) -> str | None:
    """
    SECONDARY LOGIC — emergency corrections when border is dangerously close.

    Runs ONLY when the primary road-centre correction is not enough.
    Returns a command string or None.
    """
    if largest_h < BORDER_VISIBLE_MIN:
        return None

    # ── CRITICAL: border fills frame → REVERSE ────────────────────────────
    if largest_h >= BORDER_CRITICAL:
        return "⚠️  CRITICAL — BORDER FILLING FRAME → REVERSE NOW"

    # ── DANGER level ───────────────────────────────────────────────────────
    if largest_h >= BORDER_DANGER:
        if largest_cx < left_zone_px:
            return "🚨 DANGER — BORDER TOO CLOSE ON LEFT → STEER HARD RIGHT ~60°"
        elif largest_cx > right_zone_px:
            return "🚨 DANGER — BORDER TOO CLOSE ON RIGHT → STEER HARD LEFT ~60°"
        else:
            # Border is large and roughly centred → approaching a TURN
            return "🚨 DANGER — BORDER AHEAD (TURN APPROACHING) → STEER RIGHT ~80°"

    # ── CLOSE level ───────────────────────────────────────────────────────
    if largest_h >= BORDER_CLOSE:
        if largest_cx < left_zone_px:
            return "⚠️  CLOSE — BORDER LEFT → STEER RIGHT ~30°"
        elif largest_cx > right_zone_px:
            return "⚠️  CLOSE — BORDER RIGHT → STEER LEFT ~30°"
        else:
            return "⚠️  CLOSE — BORDER AHEAD → STEER RIGHT ~45°"

    return None


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % 3 != 0:       # process every 3rd frame
        continue

    frame        = cv2.resize(frame, (FRAME_W, FRAME_H))
    results      = model(frame, imgsz=416, device=0, verbose=False)
    boxes        = results[0].boxes

    # Per-frame accumulators
    largest_border_cx = None
    largest_border_h  = 0
    arrow_detected    = False
    current_time      = time.time()

    for box in boxes:
        cls      = int(box.cls[0])
        label    = model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        height   = y2 - y1
        width    = x2 - x1
        area     = width * height
        center_x = (x1 + x2) / 2

        # ── Shape detection ───────────────────────────────────────────────
        if label == "circle" and area > SHAPE_AREA_MIN:
            if current_time - last_shape_time > SHAPE_COOLDOWN:
                print("🔵 CIRCLE DETECTED NEARBY")
                last_shape_time = current_time

        if label == "rectangle" and area > SHAPE_AREA_MIN:
            if current_time - last_shape_time > SHAPE_COOLDOWN:
                print("🟥 RECTANGLE DETECTED NEARBY")
                last_shape_time = current_time

        # ── Border tracking ───────────────────────────────────────────────
        if label == "border":
            # Track the TALLEST (closest) border
            if height > largest_border_h:
                largest_border_h  = height
                largest_border_cx = center_x

        # ── Arrow detection ───────────────────────────────────────────────
        if label == "arrow" and area > ARROW_AREA_MIN:
            if current_time - last_arrow_time > ARROW_COOLDOWN:
                arrow_detected    = True
                last_arrow_time   = current_time
                direction         = "LEFT" if center_x < frame_center else "RIGHT"
                emit(f"↪️  ARROW DETECTED → TURN {direction} ~35°")

    # ── Update rolling border history ─────────────────────────────────────
    if largest_border_cx is not None and largest_border_h > BORDER_VISIBLE_MIN:
        border_history.append(largest_border_cx)
        if len(border_history) > HISTORY_LEN:
            border_history.pop(0)
    else:
        # Gradually forget old history when border disappears
        if border_history:
            border_history.pop(0)

    # ── Decision tree (priority order) ────────────────────────────────────
    # Arrow instructions override everything else when freshly triggered
    if not arrow_detected:

        # 1. SECURITY CHECK first (critical / danger / close border)
        sec_cmd = security_border_command(
            largest_border_cx if largest_border_cx is not None else frame_center,
            largest_border_h
        )

        if sec_cmd:
            emit(sec_cmd)

        else:
            # 2. PRIMARY ROAD-CENTRE TRACKING (gentle continuous correction)
            rc_cmd = road_centre_command(border_history)
            if rc_cmd:
                emit(rc_cmd)

    # ── Annotate and display ───────────────────────────────────────────────
    annotated = results[0].plot()

    # Draw zone lines on frame for debugging
    cv2.line(annotated, (int(left_zone_px), 0),  (int(left_zone_px), FRAME_H),  (0, 255, 255), 1)
    cv2.line(annotated, (int(right_zone_px), 0), (int(right_zone_px), FRAME_H), (0, 255, 255), 1)
    cv2.line(annotated, (int(frame_center), 0),  (int(frame_center), FRAME_H),  (255, 255, 0), 1)

    cv2.imshow("YOLO Detection", annotated)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
