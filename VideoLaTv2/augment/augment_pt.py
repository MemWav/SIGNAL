import numpy as np
import os
import math
from pathlib import Path

INPUT_CSV = 'data/normal/test_data/gesture_Turn off Light.csv'  # 원본 파일
OUTPUT_DIR = 'data/augmented_gpt/test_data'  # 저장 디렉토리
NUM_AUG = 5  # 몇 개의 증강 데이터를 생성할 것인지
MAX_ROT_ANGLE = 15  # 회전 각도 범위 (-15도 ~ 15도)

def rotate_3d(joints, angle_x=0, angle_y=0, angle_z=0):
    rx, ry, rz = np.radians([angle_x, angle_y, angle_z])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return (R @ joints.T).T

def augment_csv(input_csv, output_dir, num_aug, max_angle):
    data = np.loadtxt(input_csv, delimiter=',')
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    base_name = Path(input_csv).stem  # 'gesture_Turn off Light'
    os.makedirs(output_dir, exist_ok=True)

    for _ in range(num_aug):
        angle_x = np.random.uniform(-max_angle, max_angle)
        angle_y = np.random.uniform(-max_angle, max_angle)
        angle_z = np.random.uniform(-max_angle, max_angle)

        augmented_data = []
        for row in data:
            joints = row[:21*4].reshape(21, 4)
            angles = row[21*4:]

            coords = joints[:, :3]
            rotated = rotate_3d(coords, angle_x, angle_y, angle_z)
            new_joints = np.hstack((rotated, joints[:, 3:4]))
            new_row = np.concatenate([new_joints.flatten(), angles])
            augmented_data.append(new_row)

        augmented_data = np.array(augmented_data)

        filename = f"{base_name}_{angle_x:.2f}_{angle_y:.2f}_{angle_z:.2f}.csv"
        output_path = os.path.join(output_dir, filename)
        np.savetxt(output_path, augmented_data, delimiter=',')
        print(f"Saved: {output_path}")

if __name__ == '__main__':
    augment_csv(INPUT_CSV, OUTPUT_DIR, NUM_AUG, MAX_ROT_ANGLE)
