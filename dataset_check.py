import numpy as np
import torch
import matplotlib.pyplot as plt

from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO

from configilm.extra.DataSets import BENv2_DataSet
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.utils import save_image # <-- 이 부분을 추가합니다.

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# 4. 시각화를 위한 정규화 함수
def normalize_for_display(image_tensor):
    """(C, H, W) 텐서를 (H, W, C) numpy 배열로 변환하고 0-1로 정규화합니다."""
    # (C, H, W) -> (H, W, C)
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 각 채널을 개별적으로 0-1 범위로 정규화
    img_normalized = np.zeros_like(img_np, dtype=float)
    for i in range(img_np.shape[2]):
        band = img_np[..., i]
        band_min, band_max = band.min(), band.max()
        if band_max > band_min:
            img_normalized[..., i] = (band - band_min) / (band_max - band_min)
        else:
            img_normalized[..., i] = band
    return np.clip(img_normalized, 0, 1) # 클리핑으로 안정성 확보


def make_transform(is_train: bool = True, data_type: str = "opt", resize_size: int = 256):
    
    if data_type == "opt":
        transforms_list = [
            v2.ToTensor(),
            v2.Resize((resize_size, resize_size)),
            v2.Lambda(lambda x: torch.clamp(x / 10000.0, 0.0, 1.0)),
        ]
    elif data_type == "sar":
        transforms_list = [
            v2.ToTensor(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.Lambda(lambda x: torch.clamp((x + 30.0) / 40.0, 0.0, 1.0)),
        ]
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    if is_train:
        transforms_list.extend([
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=15),
        ])
    
    transforms_list.extend([
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return v2.Compose(transforms_list)

# 1. 데이터셋 경로 설정
datapath = {
    "images_lmdb": "/home/hyunseo/workspace/rico-hdl/Encoded-BigEarthNet",
    "metadata_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

transform = {"opt": make_transform(resize_size=256, data_type="opt", is_train=False), "sar": make_transform(resize_size=256, data_type="sar", is_train=False)}


# 2. 데이터셋 로드 (SAR + Sentinel-2 10/20m 밴드)
# 12채널 구성: [VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
ds = BENv2_DataSet.BENv2DataSet(
    data_dirs=datapath,
    img_size=(12, 120, 120),
    split='train',
    transform=transform,
    merge_patch=True
)


# 분석할 이미지 인덱스 선택
image_index = 0
img, lbl = ds[image_index]

print(lbl)

import os

sar = img["sar"]
opt = img["opt"]

save_dir = "./test_img"

sar_tensor_denorm = sar * torch.tensor(NORM_STD).view(3, 1, 1) + torch.tensor(NORM_MEAN).view(3, 1, 1)
opt_tensor_denorm = opt * torch.tensor(NORM_STD).view(3, 1, 1) + torch.tensor(NORM_MEAN).view(3, 1, 1)

sar_tensor_denorm = torch.clamp(sar_tensor_denorm, 0.0, 1.0)
opt_tensor_denorm = torch.clamp(opt_tensor_denorm, 0.0, 1.0)


# 2. save_image 함수를 사용하여 이미지 저장
sar_save_path = os.path.join(save_dir, f"benv2_sar_idx{image_index}.png")
opt_save_path = os.path.join(save_dir, f"benv2_opt_idx{image_index}.png")

save_image(sar_tensor_denorm, sar_save_path)
save_image(opt_tensor_denorm, opt_save_path)

sar_display = normalize_for_display(sar)

opt_display = normalize_for_display(opt)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(sar_display)
axes[0].set_title(f"SAR Image (Index {image_index})\n(R=VV, G=VH, B=VV)")
axes[0].axis('off')

# 오른쪽: Optical RGB 이미지

axes[1].imshow(opt_display)
axes[1].set_title(f"Optical RGB Image (Index {image_index})")
axes[1].axis('off')

plt.tight_layout()

save_path = f"benv2_sar_opt_pair_idx{image_index}.png"
# save_path = f"./test_img/test_benv2_sar_opt_pair_idx{image_index}.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight') 
print(f"시각화된 이미지를 '{save_path}'에 저장했습니다.")


