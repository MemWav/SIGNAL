import numpy as np
import os
import math
import random
import glob  # <--- glob 모듈 추가
# 'config.py' 파일이 이 스크립트와 같은 폴더에 있거나,
# Python이 찾을 수 있는 경로에 있어야 합니다.
try:
    from config import WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES
except ImportError:
    print("오류: config.py 파일을 찾을 수 없습니다. 스크립트와 같은 경로에 있는지 확인하세요.")
    # config.py가 없으면 기본값 설정 (실행은 되지만 정확하지 않을 수 있음)
    WRIST = 0
    THUMB_INDICES = [1, 2, 3, 4]
    INDEX_INDICES = [5, 6, 7, 8]
    MIDDLE_INDICES = [9, 10, 11, 12]
    RING_INDICES = [13, 14, 15, 16]
    PINKY_INDICES = [17, 18, 19, 20]
    print("경고: config.py를 찾지 못해 기본 랜드마크 인덱스를 사용합니다.")


# --- ★★★ 설정해야 할 파라미터 ★★★ ---
INPUT_DIRECTORY = 'data/normal/train_data'  # <--- 원본 CSV 파일들이 있는 디렉토리 경로
OUTPUT_DIRECTORY = 'data/augmented/train_data' # <--- 증강된 파일들을 저장할 디렉토리 경로
MAX_ROTATION_ANGLE = 15.0  # 최대 회전 각도 (도)
NUM_AUGMENTATION_FILES = 5 # 각 원본 파일 당 생성할 증강 파일 개수
# ------------------------------------

ALL_FINGER_INDICES = [THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES]

# --- (calculate_finger_angles, get_rotation_matrix, rotate_landmarks 함수는 이전과 동일) ---

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
            if norm_v1 < 1e-6 or norm_v2 < 1e-6: angle = 0.0
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
    if axis == 'x': return np.array([[1, 0, 0], 
                                     [0, cos_a, -sin_a], 
                                     [0, sin_a, cos_a]])
    elif axis == 'y': return np.array([[cos_a, 0, sin_a], 
                                       [0, 1, 0], 
                                       [-sin_a, 0, cos_a]])
    elif axis == 'z': return np.array([[cos_a, -sin_a, 0], 
                                       [sin_a, cos_a, 0], 
                                       [0, 0, 1]])
    else: raise ValueError("Axis must be 'x', 'y', or 'z'")

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

# --- (augment_data_and_save_multiple 함수는 이전과 거의 동일) ---
# --- (단, 디버깅 코드는 필요시 주석 해제하여 사용) ---

def augment_data_and_save_multiple(input_file, output_base, max_angle, num_files):
    """CSV 파일을 읽어 3D 회전 증강 후, 각 증강 버전을 별도의 파일로 저장합니다."""
    print(f"\n>> 처리 중인 파일: {input_file}")
    if not os.path.exists(input_file):
        print(f"   오류: 입력 파일 '{input_file}'을 찾을 수 없습니다.")
        return False # 오류 발생 시 False 반환

    try:
        original_data = np.loadtxt(input_file, delimiter=',')
        if original_data.size == 0:
            print(f"   경고: 입력 파일 '{input_file}'이 비어 있습니다. 건너뜁니다.")
            return True # 비었지만 오류는 아니므로 True 반환
        if original_data.ndim == 1:
            original_data = original_data.reshape(1, -1)
    except Exception as e:
        print(f"   오류: '{input_file}' 파일을 읽는 중 문제가 발생했습니다: {e}")
        return False

    print(f"   로드된 데이터 수: {original_data.shape[0]}개")

    output_files_data = [[] for _ in range(num_files)]
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

            rotated_3d = rotate_landmarks(landmarks_3d.copy(), angle_x, angle_y, angle_z)

            # --- 디버깅 필요시 주석 해제 ---
            # if row_idx == 0 and i == 0:
            #     print(f"   [Debug] Angles: X={angle_x:.2f}, Y={angle_y:.2f}, Z={angle_z:.2f}")
            #     print(f"   [Debug] Original Tip: {landmarks_3d[8]}")
            #     print(f"   [Debug] Rotated Tip:  {rotated_3d[8]}")
            #     print(f"   [Debug] Are they same? {np.allclose(landmarks_3d[8], rotated_3d[8])}")
            # ---------------------------

            rotated_landmarks = np.hstack((rotated_3d, landmarks[:, 3:]))
            new_angles = calculate_finger_angles(rotated_landmarks)
            new_row = np.concatenate([rotated_landmarks.flatten(), new_angles, [label]])
            output_files_data[i].append(new_row)

    print(f"   총 {num_files}개의 증강 파일 생성을 준비합니다.")

    for i in range(num_files):
        output_file_name = f"{output_base}_{angles[i]}.csv"
        current_data = np.array(output_files_data[i])

        try:
            output_dir = os.path.dirname(output_file_name)
            os.makedirs(output_dir, exist_ok=True) # 출력 디렉토리 생성
            np.savetxt(output_file_name, current_data, delimiter=',')
            print(f"   -> '{output_file_name}' 저장 완료.")
        except Exception as e:
            print(f"   오류: '{output_file_name}' 파일 저장 중 문제가 발생했습니다: {e}")
            return False
    return True # 성공 시 True 반환

# --- ★★★ 메인 실행 부분 수정 ★★★ ---
def process_directory(input_dir, output_dir, max_angle, num_files):
    """지정된 디렉토리의 모든 CSV 파일에 대해 증강을 수행합니다."""
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print("-" * 30)

    # 입력 디렉토리 내의 모든 .csv 파일 경로 찾기
    search_path = os.path.join(input_dir, '*.csv')
    csv_files = glob.glob(search_path)

    if not csv_files:
        print(f"경고: 입력 디렉토리 '{input_dir}'에서 CSV 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.")

    success_count = 0
    fail_count = 0

    for input_file_path in csv_files:
        # 출력 파일 기본 이름 생성
        # 예: 'data/input/gest_1.csv' -> 'data/output/gest_1'
        file_name = os.path.basename(input_file_path) # 'gest_1.csv'
        base_name, _ = os.path.splitext(file_name)   # 'gest_1'
        output_base_path = os.path.join(output_dir, base_name) # 'data/output/gest_1'

        # 증강 함수 호출
        if augment_data_and_save_multiple(input_file_path, output_base_path, max_angle, num_files):
            success_count += 1
        else:
            fail_count += 1

    print("-" * 30)
    print(f"증강 작업 완료: 성공 {success_count}개, 실패 {fail_count}개")


if __name__ == "__main__":
    print("데이터 증강 시작 (디렉토리 단위 처리)...")
    process_directory(
        INPUT_DIRECTORY,
        OUTPUT_DIRECTORY,
        MAX_ROTATION_ANGLE,
        NUM_AUGMENTATION_FILES
    )
    print("모든 작업 완료.")