import os, cv2, numpy as np, torch, mediapipe as mp
from collections import deque
from models.model import GestureModel          # 이전에 정의한 네트워크
from config import GESTURE                  # 제스처 이름 리스트

# -------------------------------------------------- #
# 고정 경로
VIDEO_PATH = "assets/example_record/gesture1.mp4"
CKPT_PATH  = "checkpoint/best.pth"

# GUI 표시 여부 (WSL/서버는 False 권장)
SHOW_WINDOW = False
if not SHOW_WINDOW:
    # Qt‐backend 오류 예방
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

# -------------------------------------------------- #
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hands, mp_hands

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 모델 준비
    model = GestureModel(return_attn=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # 2) 비디오 열기
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] 파일을 열 수 없습니다: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    hands, mp_hands = init_mediapipe()

    window_size   = 30                       # 30-프레임 슬라이딩
    landmark_buf  = deque(maxlen=window_size)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # BGR → RGB, MediaPipe 손 검출
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm  = result.multi_hand_landmarks[0].landmark
            xyz = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            landmark_buf.append(xyz)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
            )
        else:
            landmark_buf.append(np.zeros((21, 3), np.float32))  # 패딩

        # 30-프레임 모이면 예측
        if len(landmark_buf) == window_size:
            video_xyz = torch.tensor(list(landmark_buf), dtype=torch.float32)
            video_xyz = video_xyz.unsqueeze(0).to(device)       # (1,30,21,3)

            with torch.no_grad():
                pred = model(video_xyz).argmax(dim=-1).item()

            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            msg   = f"[{t_sec:7.2f}s] 예측: {GESTURE[pred]}"
            print(msg)

            if SHOW_WINDOW:
                cv2.putText(frame, f"{GESTURE[pred]} ({t_sec:5.2f}s)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        # 창 표시(옵션)
        if SHOW_WINDOW:
            cv2.imshow("gesture video", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:   # ESC 종료
                break

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
