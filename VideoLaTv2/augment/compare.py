import numpy as np

# --- 중요: 이 코드는 사용자의 로컬 환경에서 실행해야 합니다 ---
# --- 아래 경로에 파일이 실제로 존재해야 합니다 ---

original_path = 'data/augmented/test_data/gesture_Turn off Light_-1.14_1.03_-9.96.csv'
augmented_path = 'data/augmented_gpt/test_data/gesture_Turn off Light_-1.14_1.03_-9.96.csv'

try:
    original_data = np.loadtxt(original_path, delimiter=',')
    augumented_data = np.loadtxt(augmented_path, delimiter=',')

    # NumPy 배열 비교를 위한 올바른 방법 사용
    if np.array_equal(original_data, augumented_data):
        print("두 CSV 파일은 완전히 동일합니다.")
    else:
        print("두 CSV 파일은 다릅니다.")
        # 추가 확인: 형태가 같고 레이블이 같은지 확인
        if original_data.shape == augumented_data.shape:
            print(f"  - 두 파일의 형태는 {original_data.shape}로 동일합니다.")
            if np.array_equal(original_data[:, -1], augumented_data[:, -1]):
                print("  - 레이블(마지막 열)은 동일합니다 (예상된 결과).")
            else:
                print("  - 레이블(마지막 열)이 다릅니다.")
        else:
            print("  - 두 파일은 형태가 다릅니다.")


except FileNotFoundError:
    print(f"오류: '{original_path}' 또는 '{augmented_path}' 파일을 찾을 수 없습니다.")
    print("경로가 올바른지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")