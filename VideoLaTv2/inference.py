import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import serial
import time

from config import (
    WRIST,                             # 그대로 두지만 현재 코드에서는 사용 안 함
    GESTURE, COMMAND
)

# ---------------------------------------------------------------------
# 1. 모델: Transformer Encoder + LSTM (model.py) ----------------------
# ---------------------------------------------------------------------
from models.transformer_only import LandmarkTransformer        # ★ 핵심: 새 모델 임포트

def load_model(model_path='./checkpoint/best_model.pth'):
    """state_dict만 저장돼 있는 pt 파일을 읽어와 GestureModel에 로드한다."""
    model = GestureModel()            # model.py 기본 하이퍼파라미터 사용
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

# ---------------------------------------------------------------------
# 2. MediaPipe 초기화 ---------------------------------------------------
# ---------------------------------------------------------------------
def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_drawing, hands

# ---------------------------------------------------------------------
# 3. 프레임 → (21,3) 랜드마크 시퀀스 변환 -------------------------------
# ---------------------------------------------------------------------
def process_frame(frame, hands, mp_drawing, mp_hands):
    """한 프레임에서 21개 손 랜드마크 (x,y,z)만 추출해 seq 리스트로 반환한다."""
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    seq = []
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3), dtype=np.float32)
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]      # visibility·각도 등 불필요
            seq.append(joint)
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    return frame, result.multi_hand_landmarks, seq

# ---------------------------------------------------------------------
# 4. 모델 예측 ---------------------------------------------------------
# ---------------------------------------------------------------------
def predict_gesture(model, seq, seq_length=30):
    """최근 seq_length개 프레임으로 제스처를 예측한다."""
    if len(seq) < seq_length:
        return None, None, None

    # (1, 30, 21, 3)  float32
    input_data = np.expand_dims(np.stack(seq[-seq_length:]), axis=0)
    input_tensor = torch.from_numpy(input_data)

    logits = model(input_tensor)                # (1, n_cls)
    conf, idx = torch.max(logits.data, dim=1)
    conf = conf.item()
    idx = idx.item()
    return conf, idx, actions[idx]

# ---------------------------------------------------------------------
# 5. 예측 결과 오버레이 -------------------------------------------------
# ---------------------------------------------------------------------
def display_prediction(frame, landmarks, action, conf):
    if landmarks and action and conf >= 0.8:
        x_pos = int(landmarks[0].landmark[0].x * frame.shape[1])
        y_pos = int(landmarks[0].landmark[0].y * frame.shape[0]) + 20
        cv2.putText(frame, f'{action.upper()} ({conf:.2f})',
                    org=(x_pos, y_pos),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2)
    return frame

# ---------------------------------------------------------------------
# 6. 메인 루프 ---------------------------------------------------------
# ---------------------------------------------------------------------
def main():
    model = load_model()
    mp_hands, mp_drawing, hands = init_mediapipe()
    cap = cv2.VideoCapture(0)

    # 시리얼 포트: UNO(모터), LEO(LED 등) 구분
    ser_uno = serial.Serial('/dev/ttyUSB1', 9600)
    ser_leo = serial.Serial('/dev/ttyACM0', 9600)
    time.sleep(2)

    LEO_IDX = {0, 1, 4, 5}             # config.py에서 LEO 전용 명령 번호
    seq = []
    pred_queue = deque(maxlen=5)
    action_cnt = 0
    pre_cmd = ''

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame, landmarks, current_seq = process_frame(frame, hands, mp_drawing, mp_hands)
            if current_seq:                           # 손이 감지된 경우만
                seq.extend(current_seq)
                conf, idx, action = predict_gesture(model, seq)

                # 신뢰도 0.8 이상일 때만 투표 큐에 삽입
                if conf and conf >= 0.8:
                    pred_queue.append(action)

                    # 큐가 가득 차면 최빈값 확인
                    if len(pred_queue) == pred_queue.maxlen:
                        pred_list = list(pred_queue)
                        most_common = max(set(pred_list), key=pred_list.count)
                        if pred_list.count(most_common) >= 5:
                            idx = actions.index(most_common)
                            cmd = COMMAND[idx]
                            if cmd and pre_cmd != cmd:
                                print(f'Gesture: {most_common}, conf={conf:.2f}')
                                target = ser_leo if idx in LEO_IDX else ser_uno
                                target.write((cmd + '\n').encode('utf-8'))
                                print(f'→ {"LEO" if idx in LEO_IDX else "UNO"}: {cmd}')
                                action_cnt += 1
                            pre_cmd = cmd if cmd else pre_cmd
                            frame = display_prediction(frame, landmarks, most_common, conf)
                            cv2.imshow('Gesture Recognition', frame)
                            cv2.waitKey(1000)
                            if action_cnt > 100:
                                break

            cv2.imshow('Gesture Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser_uno.close()
        ser_leo.close()

# ---------------------------------------------------------------------
# 7. 실행 --------------------------------------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    actions = list(GESTURE.values())
    main()
