import numpy as np
import os
import math
import random
import glob
from config import WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES

# --- 하드코딩된 파라미터 ---
INPUT_CSV_FILE = 'data/normal/test_data/gesture_Turn off Light.csv'  # 입력 파일 경로 (실제 경로에 맞게 수정)
# 출력 파일 '기본' 이름 ('.csv' 제외)
OUTPUT_CSV_BASE_NAME = 'data/augmented/test_data/gesture_Turn off Light'
MAX_ROTATION_ANGLE = 15.0  # 최대 회전 각도 (도) - 이 값이 0이 아닌지 확인!
NUM_AUGMENTATION_FILES = 5 # 생성할 증강 파일 개수
# -----------------------------

ALL_FINGER_INDICES = [THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES]

def calculate_finger_angles(joint):
    """주어진 랜드마크에서 손가락 각도를 계산합니다."""
    angles = []
    joint_3d = joint[:, :3]

    for finger_indices in ALL_FINGER_INDICES:
        points = [WRIST] + finger_indices

        for i in range(len(points) - 2):
            p1_idx, p2_idx, p3_idx = points[i:i+3]
            p1, p2, p3 = joint_3d[p1_idx], joint_3d[p2_idx], joint_3d[p3_idx]

            v1 = p2 - p1
            v2 = p3 - p2

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 < 1e-6 or norm_v2 < 1e-6:
                angle = 0.0
            else:
                v1 = v1 / norm_v1
                v2 = v2 / norm_v2
                dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot_product)

            angles.append(angle)

    return np.degrees(angles)

def get_rotation_matrix(axis, angle_deg):
    """주어진 축과 각도(도)에 대한 3D 회전 행렬을 생성합니다."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis == 'y':
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis == 'z':
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

def rotate_landmarks(landmarks_3d, angle_x_deg, angle_y_deg, angle_z_deg):
    """손목(WRIST)을 기준으로 랜드마크를 회전시킵니다."""
    center = landmarks_3d[WRIST].copy()
    translated_landmarks = landmarks_3d - center

    rot_x = get_rotation_matrix('x', angle_x_deg)
    rot_y = get_rotation_matrix('y', angle_y_deg)
    rot_z = get_rotation_matrix('z', angle_z_deg)

    rotated_landmarks = np.dot(translated_landmarks, rot_y.T)
    rotated_landmarks = np.dot(rotated_landmarks, rot_x.T)
    rotated_landmarks = np.dot(rotated_landmarks, rot_z.T)

    final_landmarks = rotated_landmarks + center
    return final_landmarks

def augment_data_and_save_multiple(input_file, output_base, max_angle, num_files):
    """
    CSV 파일을 읽어 3D 회전 증강 후, 각 증강 버전을 별도의 파일로 저장합니다. (디버깅 포함)
    """
    if not os.path.exists(input_file):
        print(f"오류: 입력 파일 '{input_file}'을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    try:
        original_data = np.loadtxt(input_file, delimiter=',')
        if original_data.size == 0:
            print(f"경고: 입력 파일 '{input_file}'이 비어 있습니다. 증강을 건너뜁니다.")
            return
        if original_data.ndim == 1:
            original_data = original_data.reshape(1, -1)
    except Exception as e:
        print(f"오류: '{input_file}' 파일을 읽는 중 문제가 발생했습니다: {e}")
        return

    print(f"'{input_file}'에서 {original_data.shape[0]}개의 원본 데이터를 로드했습니다.")
    print(f"설정된 최대 회전 각도: {max_angle}") # <--- ★★★ 설정된 최대 각도 확인 ★★★

    output_files_data = [[] for _ in range(num_files)]
    first_frame_processed = False # 첫 프레임 디버깅 플래그

    angles = []

    for row_idx, row in enumerate(original_data):
        landmarks_flat = row[:84]
        label = row[-1]
        landmarks = landmarks_flat.reshape(21, 4)
        landmarks_3d = landmarks[:, :3]

        for i in range(num_files):
            angle_x = random.uniform(-max_angle, max_angle)
            angle_y = random.uniform(-max_angle, max_angle)
            angle_z = random.uniform(-max_angle, max_angle)

            angles.append(f"{angle_x:.02f}_{angle_y:.02f}_{angle_z:.02f}")  # 각도 저장 (디버깅용)

            # 안전을 위해 원본의 복사본을 전달
            rotated_3d = rotate_landmarks(landmarks_3d.copy(), angle_x, angle_y, angle_z)

            # --- ★★★ 디버깅 출력 (첫 프레임 & 첫 증강 파일에 대해서만) ★★★ ---
            if not first_frame_processed and i == 0:
                print("\n--- 디버깅 시작 (첫 프레임, 첫 증강) ---")
                print(f"적용된 랜덤 각도: X={angle_x:.3f}, Y={angle_y:.3f}, Z={angle_z:.3f}")
                # 검지 끝(INDEX_FINGER_TIP = 8) 좌표 비교
                print(f"원본 검지 끝 좌표: {landmarks_3d[8]}")
                print(f"회전된 검지 끝 좌표: {rotated_3d[8]}")
                are_same = np.allclose(landmarks_3d[8], rotated_3d[8]) # np.allclose 사용
                print(f"좌표 동일 여부: {are_same}")
                if are_same and (abs(angle_x) > 0.01 or abs(angle_y) > 0.01 or abs(angle_z) > 0.01):
                    print("   >>> 경고: 각도가 0이 아님에도 좌표가 동일합니다! 회전 로직 확인 필요!")
                elif not are_same:
                     print("   >>> 정상: 좌표가 변경되었습니다.")
                print("--------------------------------------\n")
            # --- ★★★ 디버깅 출력 끝 ★★★ ---

            rotated_landmarks = np.hstack((rotated_3d, landmarks[:, 3:]))
            new_angles = calculate_finger_angles(rotated_landmarks)
            new_row = np.concatenate([rotated_landmarks.flatten(), new_angles, [label]])
            output_files_data[i].append(new_row)

        if not first_frame_processed:
            first_frame_processed = True

    print(f"총 {num_files}개의 증강 파일 생성을 준비합니다.")

    for i in range(num_files):
        # 올바른 파일 이름 생성 (Base + _1.csv 등)
        output_file_name = f"{output_base}_{angles[i]}.csv"
        current_data = np.array(output_files_data[i])

        try:
            os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
            np.savetxt(output_file_name, current_data, delimiter=',')
            print(f"증강된 데이터를 '{output_file_name}' 파일에 저장했습니다. ({current_data.shape[0]}개)")
        except Exception as e:
            print(f"오류: '{output_file_name}' 파일 저장 중 문제가 발생했습니다: {e}")

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    print("데이터 증강 시작 (다중 파일 출력, 디버깅 모드)...")
    augment_data_and_save_multiple(
        INPUT_CSV_FILE,
        OUTPUT_CSV_BASE_NAME,
        MAX_ROTATION_ANGLE,
        NUM_AUGMENTATION_FILES
    )
    print("데이터 증강 완료.")