import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import time

from config import (
    WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES,
    GESTURE
)
from models.transformer_only import LandmarkTransformer

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 30
NUM_JOINTS = 21
NUM_ANGLES = 15
actions = list(GESTURE.values())

def load_model(model_path='./checkpoint/best_model.pth'):
    model = LandmarkTransformer().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_drawing, hands

def calculate_finger_angles(joint, finger_indices):
    angles = []
    points = [WRIST] + finger_indices
    for i in range(len(points)-2):
        p1, p2, p3 = points[i:i+3]
        v1 = joint[p2,:3] - joint[p1,:3]
        v2 = joint[p3,:3] - joint[p2,:3]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        angles.append(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return angles

def process_frame(frame, hands, mp_drawing, mp_hands):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    seq = []
    if result.multi_hand_landmarks:
        res = result.multi_hand_landmarks[0]
        joint = np.zeros((NUM_JOINTS,4), dtype=np.float32)
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        all_angles = []
        for finger in [THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES]:
            all_angles.extend(calculate_finger_angles(joint, finger))
        all_angles = np.degrees(all_angles)

        feature = np.concatenate([joint.flatten(), all_angles])
        seq.append(feature)
        mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    return frame, result.multi_hand_landmarks, seq

def predict_gesture(model, seq, seq_length=WINDOW_SIZE):
    if len(seq) < seq_length:
        return None, None, None

    data = np.array(seq[-seq_length:], dtype=np.float32)
    x = torch.from_numpy(data).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    return conf.item(), idx.item(), actions[idx.item()]

def display_prediction(frame, landmarks, action, conf):
    if landmarks and action and conf >= 0.8:
        x = int(landmarks[0].landmark[0].x * frame.shape[1])
        y = int(landmarks[0].landmark[0].y * frame.shape[0]) + 20
        cv2.putText(
            frame,
            f'{action.upper()} ({conf:.2f})',
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2
        )
    return frame

def main():
    model = load_model()
    mp_hands, mp_drawing, hands = init_mediapipe()
    cap = cv2.VideoCapture(0)

    seq = []
    pred_queue = deque(maxlen=5)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame, landmarks, cur_seq = process_frame(frame, hands, mp_drawing, mp_hands)
            if cur_seq:
                seq.extend(cur_seq)
                conf, idx, action = predict_gesture(model, seq)

                if conf and conf >= 0.8:
                    pred_queue.append(action)
                    if len(pred_queue) == pred_queue.maxlen:
                        most = max(set(pred_queue), key=pred_queue.count)
                        count = pred_queue.count(most)
                        if count >= 5:
                            frame = display_prediction(frame, landmarks, most, conf)
                            pred_queue.clear()

            cv2.imshow('Gesture Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
