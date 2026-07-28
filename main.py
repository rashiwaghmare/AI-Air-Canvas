import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

finger_ids = {
    "thumb": (4, 3),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18)
}

def fingers_up(landmarks):
    fingers = {}
    thumb_tip = landmarks[4]
    thumb_joint = landmarks[3]
    fingers["thumb"] = thumb_tip.x < thumb_joint.x
    for name in ["index", "middle", "ring", "pinky"]:
        tip_id, joint_id = finger_ids[name]
        tip = landmarks[tip_id]
        joint = landmarks[joint_id]
        fingers[name] = tip.y < joint.y
    return fingers

ret, test_frame = cap.read()
h, w, _ = test_frame.shape

canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
draw_color = (255, 255, 255)
brush_thickness = 4

# ---- Undo/Redo setup ----
undo_stack = []
redo_stack = []
drawing_active = False
MAX_HISTORY = 20

def save_state():
    undo_stack.append(canvas.copy())
    if len(undo_stack) > MAX_HISTORY:
        undo_stack.pop(0)
    redo_stack.clear()

# ---- Shape recognition setup ----
stroke_points = []

def recognize_and_draw_shape(points, color):
    if len(points) < 5:
        return False

    pts_array = np.array(points, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts_array)
    if bw < 30 or bh < 30:
        return False

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts_array)
    cv2.fillPoly(mask, [hull], 255)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    corners = len(approx)

    area = cv2.contourArea(contour)
    if area < 500:
        return False

    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    cx, cy = x + bw // 2, y + bh // 2

    if circularity > 0.75:
        radius = int((bw + bh) / 4)
        cv2.circle(canvas, (cx, cy), radius, color, brush_thickness)
        return True
    elif corners == 3:
        cv2.drawContours(canvas, [approx], 0, color, brush_thickness)
        return True
    elif corners == 4:
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, brush_thickness)
        return True

    return False

# ---- Fullscreen window setup ----
cv2.namedWindow("AI Air Canvas", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("AI Air Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    now_drawing = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            fingers = fingers_up(landmarks)

            index_tip = landmarks[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            if fingers["index"] and not fingers["middle"]:
                now_drawing = True

                # Stroke just started -> save canvas BEFORE this stroke, start fresh point list
                if not drawing_active:
                    save_state()
                    stroke_points = []

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)
                prev_x, prev_y = x, y
                stroke_points.append((x, y))

                cv2.circle(frame, (x, y), 5, (0, 0, 255), cv2.FILLED)

            elif fingers["index"] and fingers["middle"]:
                prev_x, prev_y = 0, 0
                cv2.circle(frame, (x, y), 5, (0, 255, 255), 2)

            else:
                prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Stroke just ended -> try shape recognition
    if drawing_active and not now_drawing:
        recognize_and_draw_shape(stroke_points, draw_color)
        stroke_points = []

    drawing_active = now_drawing

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("AI Air Canvas", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        save_state()
        canvas[:] = 0
    elif key == ord('u'):  # UNDO
        if undo_stack:
            redo_stack.append(canvas.copy())
            canvas = undo_stack.pop()
    elif key == ord('r'):  # REDO
        if redo_stack:
            undo_stack.append(canvas.copy())
            canvas = redo_stack.pop()

cap.release()
cv2.destroyAllWindows()