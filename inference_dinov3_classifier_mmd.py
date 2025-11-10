import torch
import os
from PIL import Image
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torchvision.transforms import v2
import torch.nn.functional as F
import warnings
from utils.transforms import make_transform
from configilm.extra.DataSets import BENv2_DataSet

from torch.utils.data import DataLoader, Dataset 
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import glob 

# GMM 관련 라이브러리 제거됨

# -----------------------------------------------------------------
# [수정] MMD (Maximum Mean Discrepancy) 계산 함수 (배치 처리)
# -----------------------------------------------------------------
def rbf_kernel(X, Y, gamma):
    """
    RBF (Gaussian) 커널 계산 (내용 동일)
    """
    dist_sq = torch.cdist(X, Y, p=2)**2
    return torch.exp(-gamma * dist_sq)

def calculate_mmd_batched(X, Y, batch_size, gamma=None):
    """
    [NEW] MMD (Maximum Mean Discrepancy)를 배치로 나누어 계산합니다.
    
    X: Source 텐서 (N, D)
    Y: Target 텐서 (M, D)
    batch_size: GPU 메모리에 맞게 조절 (예: 2000 또는 5000)
    """
    X, Y = X.to(torch.float32), Y.to(torch.float32)
    
    if gamma is None:
        gamma = 1.0 / X.shape[1] 
        
    N, M = X.shape[0], Y.shape[0]
    device = X.device
    
    # 1. K_XX.mean() 계산
    print("  Calculating K_XX.mean() (batched)...")
    total_k_xx = torch.tensor(0.0, device=device)
    for i in tqdm(range(0, N, batch_size), desc="K_XX"):
        X_i = X[i : i + batch_size]
        # K_XX[i, :] 계산
        K_i_all = rbf_kernel(X_i, X, gamma)
        total_k_xx += K_i_all.sum()
    mean_k_xx = total_k_xx / (N * N)

    # 2. K_YY.mean() 계산
    print("  Calculating K_YY.mean() (batched)...")
    total_k_yy = torch.tensor(0.0, device=device)
    for i in tqdm(range(0, M, batch_size), desc="K_YY"):
        Y_i = Y[i : i + batch_size]
        # K_YY[i, :] 계산
        K_i_all = rbf_kernel(Y_i, Y, gamma)
        total_k_yy += K_i_all.sum()
    mean_k_yy = total_k_yy / (M * M)

    # 3. K_XY.mean() 계산
    print("  Calculating K_XY.mean() (batched)...")
    total_k_xy = torch.tensor(0.0, device=device)
    for i in tqdm(range(0, N, batch_size), desc="K_XY"):
        X_i = X[i : i + batch_size]
        # K_XY[i, :] 계산
        K_i_all = rbf_kernel(X_i, Y, gamma)
        total_k_xy += K_i_all.sum()
    mean_k_xy = total_k_xy / (N * M)

    mmd_sq = mean_k_xx + mean_k_yy - 2 * mean_k_xy
    
    return torch.clamp(mmd_sq, min=0).sqrt()

# -----------------------------------------------------------------
# 설정값 (기존과 동일)
# -----------------------------------------------------------------
DATA_TYPE = "sar"
INFERENCE_CHECKPOINT_PATH = f"./checkpoints/stage0_{DATA_TYPE}/checkpoint_stage0_epoch101.pth"
SEN12_ROOT_DIR = "/home/hyunseo/workspace/sar2opt/dataset/v_2"
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
LABEL_SIZE = 19
REPO_NAME = "/home/hyunseo/workspace/dinov3"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv"]
MERGE_PATCH = False
RESIZE_SIZE = 256 if MERGE_PATCH else 128
BATCH_SIZE = 112 
SAVE_FIG_FOLDER = "./save_fig"

# [NEW] MMD 계산을 위한 배치 크기 (GPU 메모리 상황에 따라 조절)
# 24GB GPU에서 5000 정도면 (5000 * 121600 * 4 bytes) 약 2.2GB로 안정적입니다.
MMD_CALC_BATCH_SIZE = 5000

# -----------------------------------------------------------------
# (SEN12_Dataset, DinoV3Linear 클래스 - 내용 동일, 생략)
# -----------------------------------------------------------------
class SEN12_Dataset(Dataset):
    """
    SEN1-2 데이터셋을 위한 커스텀 클래스
    (내용 동일)
    """
    def __init__(self, root_dir, transform: dict = None):
        self.root_dir = root_dir
        self.transform_dict = transform 
        
        self.sar_image_paths = []
        self.opt_image_paths = []
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']

        try:
            class_dirs = [d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)]
            if not class_dirs:
                print(f"Warning: No class directories (like 'agri', 'urban') found in {self.root_dir}")
                class_dirs = [self.root_dir] 
        except Exception as e:
            print(f"Error finding class dirs: {e}")
            class_dirs = [self.root_dir] 

        for class_dir in class_dirs:
            s1_dir = os.path.join(class_dir, 's1')
            s2_dir = os.path.join(class_dir, 's2')

            s1_paths_for_class = []
            s2_paths_for_class = []

            for ext in extensions:
                s1_paths_for_class.extend(glob.glob(os.path.join(s1_dir, f'*{ext}')))
                s2_paths_for_class.extend(glob.glob(os.path.join(s2_dir, f'*{ext}')))
            
            s1_paths_for_class.sort()
            s2_paths_for_class.sort()
            
            if len(s1_paths_for_class) != len(s2_paths_for_class):
                print(f"Warning: Mismatch in class '{os.path.basename(class_dir)}'.")
                print(f"  Found {len(s1_paths_for_class)} s1 images but {len(s2_paths_for_class)} s2 images. Skipping this class.")
                continue
                
            self.sar_image_paths.extend(s1_paths_for_class)
            self.opt_image_paths.extend(s2_paths_for_class)
        
        if not self.sar_image_paths:
            print(f"Error: No valid s1/s2 pairs were found in {root_dir}")
        
        self.patches_per_image = 4

    def __len__(self):
        return len(self.sar_image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        image_index = idx // self.patches_per_image
        patch_index = idx % self.patches_per_image

        sar_img_path = self.sar_image_paths[image_index]
        opt_img_path = self.opt_image_paths[image_index]
        
        try:
            sar_image_raw = Image.open(sar_img_path).convert("L")
            sar_image_raw = Image.merge("RGB", (sar_image_raw, sar_image_raw, sar_image_raw))
            opt_image_raw = Image.open(opt_img_path).convert("RGB")
            
        except Exception as e:
            print(f"Error loading image pair at index {image_index}: {e}")
            print(f"  SAR: {sar_img_path}")
            print(f"  OPT: {opt_img_path}")
            return None 

        if patch_index == 0:    # Top-Left (TL)
            box = (0, 0, 128, 128)
        elif patch_index == 1:  # Top-Right (TR)
            box = (128, 0, 256, 128)
        elif patch_index == 2:  # Bottom-Left (BL)
            box = (0, 128, 128, 256)
        else: # patch_index == 3   # Bottom-Right (BR)
            box = (128, 128, 256, 256)
            
        sar_patch = sar_image_raw.crop(box)
        opt_patch = opt_image_raw.crop(box)
        
        sar_tensor = sar_patch
        opt_tensor = opt_patch

        if self.transform_dict is not None:
            if 'sar' in self.transform_dict and self.transform_dict['sar'] is not None:
                sar_tensor = self.transform_dict['sar'](sar_patch)
            if 'opt' in self.transform_dict and self.transform_dict['opt'] is not None:
                opt_tensor = self.transform_dict['opt'](opt_patch)
        
        return {"sar": sar_tensor, "opt": opt_tensor}
    
class DinoV3Linear(nn.Module):
    def __init__(self, backbone, hidden_size: int, num_classes: int, freeze_backbone: bool = True):
        # ... (내용 동일) ...
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.head = nn.Linear(hidden_size, num_classes)

    def forward_features(self, pixel_values):
        return self.backbone(pixel_values)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        logits = self.head(outputs)
        return logits

def evaluate_dataset(model, data_loader, device, dataset_name, data_type, has_labels = False, label_size = None):
    """
    [수정]
    주어진 데이터로더에 대해 평가를 수행하고, 예측 확률(all_probs)을 반환합니다.
    (내용 동일, GMM 코드 없음)
    """
    
    all_labels = []
    all_preds = []
    all_probs = []

    print(f"\n--- 🚀 Starting batch inference on {dataset_name} ({data_type}) data ---")

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            
            if batch is None:
                continue

            if has_labels:
                img, lbl = batch
                inputs = img[data_type].to(device)
                labels = lbl.to(device)
            else:
                img_tensor_dict = batch 
                inputs = img_tensor_dict[data_type].to(device)

            logits = model(inputs)
            probabilities = torch.sigmoid(logits)
            
            all_probs.append(probabilities.cpu())

            if has_labels:
                threshold = 0.5
                preds = (probabilities > threshold).int()
                all_labels.append(labels.cpu())
                
                # [!!! 수정된 부분 !!!]
                # preds 텐서를 .cpu()로 옮긴 후 리스트에 추가합니다.
                all_preds.append(preds.cpu()) 

            # [!!! 중요 !!!] break 문이 제거된 상태여야 합니다. (전체 데이터 처리)

    print(f"Inference complete for {dataset_name}.")
    
    if len(all_probs) == 0:
        print(f"Error: No data processed for {dataset_name}. Skipping GMM.")
        return None
        
    all_probs = torch.cat(all_probs, dim=0).numpy()

    if has_labels:
        # --- Metrics 계산 (BENv2) ---
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # [수정 완료] 이제 all_preds 리스트에 CPU 텐서만 있으므로 .numpy()가 정상 작동합니다.
        all_preds = torch.cat(all_preds, dim=0).numpy()
        
        print(f"\n--- 📊 Multi-Label Confusion Matrix ({dataset_name}) ---")
        mcm = multilabel_confusion_matrix(all_labels, all_preds)
        print(mcm)
        
        print(f"\n--- 📋 Classification Report ({dataset_name}) ---")
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=[f"Class {i:02d}" for i in range(label_size)], 
            zero_division=0
        )
        print(report)
    else:
        print(f"\n--- 📊 Metrics skipped for {dataset_name} (no labels provided) ---")

    # [수정] 예측 확률(probabilities) NumPy 배열을 반환
    return all_probs

def run_inference():

    # --- 공통 변환 정의 ---
    transform_opt_val = make_transform(resize_size=RESIZE_SIZE, data_type="opt", dataset = "benv2", is_train=False)
    transform_sar_val = make_transform(resize_size=RESIZE_SIZE, data_type="sar", dataset = "benv2", is_train=False)
    transform_sar_sen12_val = make_transform(resize_size=RESIZE_SIZE, data_type="sar", dataset = "sen12", is_train=False)
    transform_opt_sen12_val = make_transform(resize_size=RESIZE_SIZE, data_type="opt", dataset = "sen12", is_train=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 모델 로드 (공통) ---
    # (내용 동일, 생략)
    print("--- Loading Model ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls_feature_size = torch.hub.load(REPO_NAME, 'dinov3_vith16plus', source='local').num_features
        backbone = torch.hub.load(REPO_NAME, 'dinov3_vith16plus', source='local')

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    
    model = DinoV3Linear(backbone, hidden_size=cls_feature_size, num_classes=LABEL_SIZE, freeze_backbone=True)
    model = get_peft_model(model, lora_config)
    
    try:
        checkpoint = torch.load(INFERENCE_CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Successfully loaded model weights from epoch {checkpoint.get('epoch', 'N/A')}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        try:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()
            print("Successfully loaded model weights (after stripping 'module.' prefix).")
        except Exception as e_retry:
            print(f"Failed to load checkpoint even after retry: {e_retry}")
            return
            
    print("-" * 40)

    # --- 1. BENv2 평가 (레이블 있음) ---
    # (내용 동일, 생략)
    print("--- 1. Preparing BENv2 (Validation Set) ---")
    datapath = {
        "images_lmdb": "/home/hyunseo/workspace/rico-hdl/Encoded-BigEarthNet",
        "metadata_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata.parquet",
        "metadata_snow_cloud_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    }
    ds_benv2 = BENv2_DataSet.BENv2DataSet(
        data_dirs=datapath,
        img_size=(12, 120, 120),
        split='test',
        transform={"opt": transform_opt_val, "sar": transform_sar_val}, 
        merge_patch=False,
        max_len=5000, 
    )
    loader_benv2 = DataLoader(
        ds_benv2, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Loaded {len(ds_benv2)} BENv2 test images.")
    
    benv2_probs = evaluate_dataset(model, loader_benv2, device, 
                                    dataset_name="BENv2", 
                                    data_type=DATA_TYPE,
                                    has_labels=True, 
                                    label_size=LABEL_SIZE)

    print("-" * 40)
    
    # --- 2. SEN1-2 평가 (레이블 없음) ---
    # (내용 동일, 생략)
    print("--- 2. Preparing SEN1-2 (Unlabeled Set) ---")
    sen12_probs = None 
    if not os.path.exists(SEN12_ROOT_DIR):
        print(f"Warning: SEN12_ROOT_DIR '{SEN12_ROOT_DIR}' not found. Skipping SEN1-2 evaluation.")
    else:
        ds_sen12_full = SEN12_Dataset(
            root_dir=SEN12_ROOT_DIR,
            transform={"opt": transform_opt_sen12_val, "sar": transform_sar_sen12_val} 
        )
        VAL_SPLIT_RATIO = 0.1
        total_size = len(ds_sen12_full)
        val_size = int(total_size * VAL_SPLIT_RATIO)
        train_size = total_size - val_size
        
        print(f"Splitting SEN1-2 full dataset ({total_size} patches) into:")
        print(f"  - Train: {train_size} patches")
        print(f"  - Val:   {val_size} patches")

        generator = torch.Generator().manual_seed(42)
        ds_sen12_train, ds_sen12_val = torch.utils.data.random_split(
            ds_sen12_full, 
            [train_size, val_size],
            generator=generator
        )
        loader_sen12_val = DataLoader(
            ds_sen12_val, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        print(f"Loaded {len(loader_sen12_val)} SEN1-2 val patches.")
        
        sen12_probs = evaluate_dataset(model, loader_sen12_val, device, 
                                        dataset_name="SEN12_Val", 
                                        data_type=DATA_TYPE,
                                        has_labels=False)

    print("-" * 40)
    
    print("-" * 40)
    
    # --- 3. [수정] 그룹별 MMD (Group-wise MMD) 계산 ---
    if benv2_probs is not None and sen12_probs is not None:
        print("\n--- 🔬 Calculating Group-wise MMD (DDA-MLIC philosophy) ---")
        print(f"Using MMD Batch Size: {MMD_CALC_BATCH_SIZE}")
        print("This may take several minutes...")

        # --- 데이터 준비 및 그룹 분할 ---
        # 1. NumPy 배열을 flatten하여 1D로 만듦
        source_flat = benv2_probs.flatten()
        target_flat = sen12_probs.flatten()

        # 2. 0.5를 기준으로 네 그룹으로 분할
        source_neg = source_flat[source_flat < 0.5]
        source_pos = source_flat[source_flat >= 0.5]
        target_neg = target_flat[target_flat < 0.5]
        target_pos = target_flat[target_flat >= 0.5]

        # 3. 각 그룹을 PyTorch 텐서로 변환하고, MMD 계산을 위해 (N, 1) 형태로 만듦
        source_neg_tensor = torch.from_numpy(source_neg).reshape(-1, 1).to(device)
        source_pos_tensor = torch.from_numpy(source_pos).reshape(-1, 1).to(device)
        target_neg_tensor = torch.from_numpy(target_neg).reshape(-1, 1).to(device)
        target_pos_tensor = torch.from_numpy(target_pos).reshape(-1, 1).to(device)

        print("\n--- Data for MMD Calculation ---")
        print(f"Source Negative (<0.5) group shape: {source_neg_tensor.shape}")
        print(f"Source Positive (>=0.5) group shape: {source_pos_tensor.shape}")
        print(f"Target Negative (<0.5) group shape: {target_neg_tensor.shape}")
        print(f"Target Positive (>=0.5) group shape: {target_pos_tensor.shape}")
        
        # --- Negative 그룹 MMD 계산 ---
        mmd_neg = torch.tensor(0.0, device=device)
        if len(source_neg_tensor) > 0 and len(target_neg_tensor) > 0:
            print("\nCalculating MMD for Negative Groups (<0.5)...")
            mmd_neg = calculate_mmd_batched(
                source_neg_tensor,
                target_neg_tensor,
                batch_size=MMD_CALC_BATCH_SIZE,
                gamma=None
            )
        else:
            print("\nSkipping MMD for Negative Groups (one or both are empty).")

        # --- Positive 그룹 MMD 계산 ---
        mmd_pos = torch.tensor(0.0, device=device)
        if len(source_pos_tensor) > 0 and len(target_pos_tensor) > 0:
            print("\nCalculating MMD for Positive Groups (>=0.5)...")
            mmd_pos = calculate_mmd_batched(
                source_pos_tensor,
                target_pos_tensor,
                batch_size=MMD_CALC_BATCH_SIZE,
                gamma=None
            )
        else:
            print("\nSkipping MMD for Positive Groups (one or both are empty).")

        # --- 최종 결과 출력 ---
        # 논문에서는 두 loss를 합산 (여기서는 평가이므로 개별 값과 합을 모두 출력)
        # alpha, beta 가중치는 학습 시 사용. 여기서는 단순 합산.
        total_mmd = mmd_neg + mmd_pos
        
        print("\n" + "="*50)
        print("📊 Group-wise MMD Results")
        print("="*50)
        print(f"  - MMD between Negative Groups (<0.5): {mmd_neg.item():.6f}")
        print(f"  - MMD between Positive Groups (>=0.5): {mmd_pos.item():.6f}")
        print(f"  - Total MMD (Negative + Positive):    {total_mmd.item():.6f}")
        print("-" * 50)
        print("(0에 가까울수록 각 그룹별 분포가 유사함을 의미)")
        
    else:
        print("Skipping MMD calculation because one or both dataset evaluations failed.")

if __name__ == "__main__":
    run_inference()