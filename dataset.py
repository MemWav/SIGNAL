from torch.utils.data import Dataset
import torch
import os
import numpy as np

class GestureDatasetCSV(Dataset):
    def __init__(self, folder_dir, window_size=30, num_joints=21, num_angles=15):
        self.window_size = window_size
        self.num_joints  = num_joints
        self.num_angles  = num_angles
        # feature dim: coords (J*4) + angles
        self.input_dim   = num_joints * 4 + num_angles
        self.data   = []
        self.labels = []
        self._load_data(folder_dir)

    def _load_data(self, folder_dir):
        for fname in os.listdir(folder_dir):
            if not fname.endswith('.csv'):
                continue
            path = os.path.join(folder_dir, fname)
            # CSV 구조: 첫 N-1 열이 feature, 마지막 열이 label
            arr = np.loadtxt(path, delimiter=',')  # shape (N_frames, input_dim + 1)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            features = arr[:, :-1].astype(np.float32)  # (N_frames, input_dim)
            labels   = arr[:,  -1].astype(np.int64)    # (N_frames,)

            # sliding window 생성
            num_frames = features.shape[0]
            for i in range(num_frames - self.window_size + 1):
                window = features[i : i + self.window_size]  # (window_size, input_dim)
                label  = labels[i + self.window_size - 1]    # 마지막 프레임 라벨
                self.data.append(window)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])  # (T, input_dim)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
    # class GestureDatasetCSV(Dataset):
    # """
    # CSV 파일에서 (x,y,z,visibility)*21 + 15 angles 를 모두 읽어
    # sliding window 형태로 윈도우 단위 샘플을 생성합니다.
    # """
    # def __init__(self, folder_dir, window_size=30):
    #     self.window_size = window_size
    #     self.data = []
    #     self.labels = []
    #     self._load_data(folder_dir)

    # def _load_data(self, folder_dir):
    #     for fn in os.listdir(folder_dir):
    #         if not fn.endswith('.csv'):
    #             continue
    #         path = os.path.join(folder_dir, fn)
    #         df = pd.read_csv(path, header=None)
    #         arr = df.values  # shape (N, 84+15+1) = (N, 100) 예시
    #         features = arr[:, :-1]  # 모든 feature 열
    #         labels   = arr[:,  -1].astype(int)

    #         # sliding window
    #         for i in range(len(arr) - self.window_size + 1):
    #             win = features[i:i+self.window_size]         # (window, feat_dim)
    #             lbl = labels[i+self.window_size-1]           # 라벨
    #             self.data.append(win.astype(np.float32))
    #             self.labels.append(int(lbl))

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     x = torch.from_numpy(self.data[idx])   # (T, feat_dim)
    #     y = torch.tensor(self.labels[idx], dtype=torch.long)
    #     return x, y
