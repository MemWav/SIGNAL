import torch
import cv2
import numpy as np

def img_to_landmarks(img_path, mp_hands):
    img_bgr = cv2.imread(img_path)
    res = mp_hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        raise ValueError('No hand detected')
    lm = res.multi_hand_landmarks[0]
    xyz = np.array([[pt.x, pt.y, pt.z] for pt in lm.landmark], dtype=np.float32)
    return torch.tensor(xyz).unsqueeze(0)