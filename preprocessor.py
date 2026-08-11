import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

def list_all_stems(mask_dir):
    """masks 폴더를 훑어서 전체 704개의 파일 stem(확장자/접미사 뗀 이름)을 뽑아낸다.
    train/val/test로 나누기 전에 딱 한 번만 호출해서 전체 목록을 얻는 용도."""
    stems = []
    for mf in os.listdir(mask_dir):
        if mf.startswith("MCU"):
            stem = mf.replace(".png", "")       # Montgomery: mask 파일명 = CXR 파일명
        else:
            stem = mf.replace("_mask.png", "")  # Shenzhen: mask 파일명에 _mask가 붙음
        stems.append(stem)
    return stems


class LungSegmentationDataset(Dataset):
    def __init__(self, cxr_dir, mask_dir, file_stems, img_size=224):
        # 여기서는 실제 이미지를 하나도 안 읽는다 -- "나중에 idx가 오면 어떻게 찾을지"
        # 필요한 정보(경로들, stem 목록)만 기억해두는 게 __init__의 역할.
        self.cxr_dir = cxr_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.file_stems = file_stems  # train/val/test로 미리 나눠서 넘겨받은 부분집합

    def __len__(self):
        return len(self.file_stems)

    def _mask_path_for(self, stem):
        # Montgomery/Shenzhen 파일명 규칙 분기 -- build_file_pairs에서 가져온 로직
        if stem.startswith("MCU"):
            return os.path.join(self.mask_dir, stem + ".png")
        return os.path.join(self.mask_dir, stem + "_mask.png")

    def __getitem__(self, idx):
        stem = self.file_stems[idx]
        cxr_path = os.path.join(self.cxr_dir, stem + ".png")
        mask_path = self._mask_path_for(stem)

        # 1. 파일 열기 + 그레이스케일로 통일 (CXR이 P/RGB/L로 섞여있었으니까)
        cxr_img = Image.open(cxr_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")

        # 2. 리사이즈 -- 마스크는 반드시 NEAREST (이진값 0/255가 안 뭉개지게)
        cxr_img = cxr_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask_img = mask_img.resize((self.img_size, self.img_size), Image.NEAREST)

        # 3. numpy 배열 변환 + 0~1로 정규화
        cxr_arr = np.array(cxr_img, dtype=np.float32) / 255.0
        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0

        # 4. 텐서 변환 + 채널 차원 추가: (H, W) -> (1, H, W)
        #    Conv2d가 (채널, H, W) 형태를 기대하기 때문
        image_tensor = torch.from_numpy(cxr_arr).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return image_tensor, mask_tensor
