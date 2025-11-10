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

# [NEW] 새로 추가된 라이브러리
from torch.utils.data import DataLoader, Dataset # Dataset 추가
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import glob # [NEW] SEN1-2 파일 검색을 위해 추가

# [NEW] GMM 피팅 및 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans # [NEW] K-Means 임포트
from scipy.stats import norm

# -----------------------------------------------------------------
# 설정값 (기존과 동일 + SEN1-2 경로 추가)
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
# DATA_TYPE = "opt" # 이 변수는 이제 evaluate_dataset 함수로 이동
BATCH_SIZE = 112 
SAVE_FIG_FOLDER = "./save_fig"

class SEN12_Dataset(Dataset):
    """
    SEN1-2 데이터셋을 위한 커스텀 클래스 (레이블 없음)
    [수정됨] s1(SAR)과 s2(OPT) 이미지를 정렬(sort)하여 쌍으로 로드합니다.
    [수정됨] 256x256 이미지를 4개의 128x128 패치로 분할합니다.
    [수정됨] 'sar', 'opt' 키를 가진 transform 딕셔너리를 사용합니다.
    """
    def __init__(self, root_dir, transform: dict = None):
        """
        :param root_dir: 데이터셋의 최상위 경로 (예: 'v_2/')
        :param transform: {"sar": sar_transform, "opt": opt_transform} 형태의 딕셔너리
        """
        self.root_dir = root_dir
        self.transform_dict = transform 
        
        self.sar_image_paths = []
        self.opt_image_paths = []
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']

        # [수정됨] 'agri', 'urban' 등 하위 클래스 폴더를 찾습니다.
        try:
            class_dirs = [d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)]
            if not class_dirs:
                print(f"Warning: No class directories (like 'agri', 'urban') found in {self.root_dir}")
                # root_dir 자체가 클래스 폴더를 포함하는 경로일 수 있으므로, 재시도
                class_dirs = [self.root_dir] 
        except Exception as e:
            print(f"Error finding class dirs: {e}")
            class_dirs = [self.root_dir] # fallback

        # [수정됨] 각 클래스 폴더 내에서 s1/s2 페어링 수행
        for class_dir in class_dirs:
            s1_dir = os.path.join(class_dir, 's1')
            s2_dir = os.path.join(class_dir, 's2')

            s1_paths_for_class = []
            s2_paths_for_class = []

            for ext in extensions:
                s1_paths_for_class.extend(glob.glob(os.path.join(s1_dir, f'*{ext}')))
                s2_paths_for_class.extend(glob.glob(os.path.join(s2_dir, f'*{ext}')))
            
            # --- [핵심] 정렬(sort) 기반 페어링 ---
            s1_paths_for_class.sort()
            s2_paths_for_class.sort()
            
            # s1과 s2의 이미지 개수가 다르면, 해당 클래스는 경고 후 건너뜁니다.
            if len(s1_paths_for_class) != len(s2_paths_for_class):
                print(f"Warning: Mismatch in class '{os.path.basename(class_dir)}'.")
                print(f"  Found {len(s1_paths_for_class)} s1 images but {len(s2_paths_for_class)} s2 images. Skipping this class.")
                continue
                
            # 정렬된 리스트를 메인 리스트에 추가
            self.sar_image_paths.extend(s1_paths_for_class)
            self.opt_image_paths.extend(s2_paths_for_class)
        
        if not self.sar_image_paths:
            print(f"Error: No valid s1/s2 pairs were found in {root_dir}")
        
        self.patches_per_image = 4

    def __len__(self):
        # 전체 데이터셋 길이는 (찾아낸 쌍(pair)의 수 * 4)
        return len(self.sar_image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        # 1. 어떤 원본 이미지를 로드할지 계산
        image_index = idx // self.patches_per_image
        
        # 2. 해당 이미지에서 몇 번째 패치(0, 1, 2, 3)를 자를지 계산
        patch_index = idx % self.patches_per_image

        # 3. [수정됨] 정렬된 리스트에서 SAR (s1)과 OPT (s2) 경로를 직접 가져옴
        sar_img_path = self.sar_image_paths[image_index]
        opt_img_path = self.opt_image_paths[image_index]
        
        try:
            # 4. 원본 256x256 이미지 쌍(pair) 로드
            sar_image_raw = Image.open(sar_img_path).convert("L")
            sar_image_raw = Image.merge("RGB", (sar_image_raw, sar_image_raw, sar_image_raw))
            
            opt_image_raw = Image.open(opt_img_path).convert("RGB")
            
        except Exception as e:
            print(f"Error loading image pair at index {image_index}: {e}")
            print(f"  SAR: {sar_img_path}")
            print(f"  OPT: {opt_img_path}")
            return None # 오류 발생 시 None 반환 (DataLoader에서 처리 필요)

        # 5. patch_index에 따라 128x128 영역 자르기 (로직 동일)
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
        
        # 6. 128x128 패치에 딕셔너리의 transform 적용 (로직 동일)
        sar_tensor = sar_patch
        opt_tensor = opt_patch

        if self.transform_dict is not None:
            if 'sar' in self.transform_dict and self.transform_dict['sar'] is not None:
                sar_tensor = self.transform_dict['sar'](sar_patch)
            if 'opt' in self.transform_dict and self.transform_dict['opt'] is not None:
                opt_tensor = self.transform_dict['opt'](opt_patch)
        
        # 7. 딕셔너리로 반환 (로직 동일)
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
    주어진 데이터로더에 대해 평가를 수행하고 GMM 플롯을 생성합니다.
    [수정] GMM을 0.5 기준으로 나누어(piecewise) 피팅합니다.
    """
    
    all_labels = []
    all_preds = []
    all_probs = []

    print(f"\n--- 🚀 Starting batch inference on {dataset_name} ({data_type}) data ---")

    with torch.inference_mode():
        # ... (Tqdm 루프 ... 동일) ...
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
                all_preds.append(preds.cpu())

            break

    print(f"Inference complete for {dataset_name}.")
    
    if len(all_probs) == 0:
        print(f"Error: No data processed for {dataset_name}. Skipping GMM.")
        return None
        
    all_probs = torch.cat(all_probs, dim=0).numpy()

    if has_labels:
        # --- Metrics 계산 (BENv2) ---
        # ... (Metrics 코드 ... 동일) ...
        all_labels = torch.cat(all_labels, dim=0).numpy()
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


    # [NEW] --- 0.5 기준 Piecewise GMM 피팅 ---
    print(f"\n--- 📈 Fitting Piecewise GMM (split at z=0.5) ---")
    
    flat_probs = all_probs.flatten()
    
    # 1. 데이터를 0.5 기준으로 두 그룹으로 분할
    data_neg = flat_probs[flat_probs < 0.5].reshape(-1, 1)
    data_pos = flat_probs[flat_probs >= 0.5].reshape(-1, 1)

    # 2. 각 그룹이 비어있는지 확인
    if data_neg.size == 0:
        print("Error: No data found for Negative Component (z < 0.5).")
        return None
    if data_pos.size == 0:
        print("Error: No data found for Positive Component (z >= 0.5).")
        return None
        
    print(f"Fitting GMM 1 to {len(data_neg)} points (< 0.5)")
    print(f"Fitting GMM 2 to {len(data_pos)} points (>= 0.5)")

    # 3. 각 그룹에 n_components=1 GMM을 개별적으로 피팅
    gmm_neg = GaussianMixture(n_components=1, random_state=42, n_init=10)
    gmm_neg.fit(data_neg)
    
    gmm_pos = GaussianMixture(n_components=1, random_state=42, n_init=10)
    gmm_pos.fit(data_pos)

    # 4. 각 GMM에서 파라미터 및 가중치(weight) 추출
    mean_neg = gmm_neg.means_.item()
    std_neg = np.sqrt(gmm_neg.covariances_.item())
    weight_neg = len(data_neg) / len(flat_probs) # 전체 데이터 중 이 그룹의 비율
    
    mean_pos = gmm_pos.means_.item()
    std_pos = np.sqrt(gmm_pos.covariances_.item())
    weight_pos = len(data_pos) / len(flat_probs) # 전체 데이터 중 이 그룹의 비율

    print(f"GMM fitted successfully (piecewise).")
    print(f"  - Negative Component (z < 0.5): Mean={mean_neg:.4f}, Weight={weight_neg:.4f}")
    print(f"  - Positive Component (z >= 0.5): Mean={mean_pos:.4f}, Weight={weight_pos:.4f}")

    # 3. 시각화 (개별 플롯)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    # 히스토그램은 ★전체 원본★ 데이터로 그려야 합니다.
    plt.hist(flat_probs, bins=100, density=True, color='lightgray', alpha=0.8, label='Predicted Probability Frequency')

    x_plot = np.linspace(0, 1, 1000).ravel()
    
    # 각 컴포넌트의 PDF를 가중치(weight)와 함께 계산
    pdf_neg_weighted = weight_neg * norm.pdf(x_plot, loc=mean_neg, scale=std_neg)
    pdf_pos_weighted = weight_pos * norm.pdf(x_plot, loc=mean_pos, scale=std_pos)
    
    # [수정] 두 개의 개별 PDF를 합쳐 전체 Mixture를 만듭니다.
    pdf_total = pdf_neg_weighted + pdf_pos_weighted 
    
    plt.plot(x_plot, pdf_total, color='black', lw=2.5, label='Fitted GMM (Mixture, Piecewise)')
    
    plt.plot(x_plot, pdf_pos_weighted, color='crimson', linestyle='--', lw=3,
             label=f'Positive Component (μ={mean_pos:.2f})')
    
    plt.plot(x_plot, pdf_neg_weighted, color='royalblue', linestyle='--', lw=2, 
             label=f'Negative Component (μ={mean_neg:.2f})')
    
    # [수정] 제목과 파일명 변경
    plt.title(f'Distribution for {dataset_name} (Fitted GMM, Piecewise 0.5 Split)', fontsize=16, weight='bold')
    plt.xlabel('Predicted Probability (z)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    
    save_filename = os.path.join(SAVE_FIG_FOLDER, f"predicted_probability_{dataset_name}_gmm_fit_piecewise.png")
    plt.savefig(save_filename, dpi=300)
    print(f"\nSaved the plot as '{save_filename}'")
    plt.close() # 그래프 창을 닫아 다음 플롯에 영향이 없도록 함

    # [수정] GMM 파라미터 반환
    return {'mean_neg': mean_neg, 'std_neg': std_neg, 'mean_pos': mean_pos, 'std_pos': std_pos}
# -----------------------------------------------------------------
# [NEW] 비교 플롯을 그리는 헬퍼 함수
# -----------------------------------------------------------------
def plot_comparison_pdfs(benv2_params, sen12_params):
    """
    BENv2와 SEN12의 GMM 컴포넌트를 가중치(phi) 없이 비교하는 플롯을 생성합니다.
    """
    print("\n--- 📊 Plotting Comparison of Unweighted GMM Components (phi-없이) ---")
    x_plot = np.linspace(0, 1, 1000).ravel()
    
    # --- 플롯 1: Negative Components (Comp 1) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    # BENv2 Negative PDF (phi-없이)
    pdf_neg_benv2 = norm.pdf(x_plot, loc=benv2_params['mean_neg'], scale=benv2_params['std_neg'])
    plt.plot(x_plot, pdf_neg_benv2, color='blue', linestyle='--', lw=2, 
             label=f"BENv2 Negative (μ={benv2_params['mean_neg']:.2f}, σ={benv2_params['std_neg']:.2f})")

    # SEN12 Negative PDF (phi-없이)
    pdf_neg_sen12 = norm.pdf(x_plot, loc=sen12_params['mean_neg'], scale=sen12_params['std_neg'])
    plt.plot(x_plot, pdf_neg_sen12, color='cyan', linestyle='-', lw=2, 
             label=f"SEN12 Negative (μ={sen12_params['mean_neg']:.2f}, σ={sen12_params['std_neg']:.2f})")
    
    plt.title('Comparison of Negative Components (Unweighted, phi-없이)', fontsize=16, weight='bold')
    plt.xlabel('Predicted Probability (z)', fontsize=12)
    plt.ylabel('Density (Unweighted)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    save_filename = os.path.join(SAVE_FIG_FOLDER, "comparison_negative_components.png")
    plt.savefig(save_filename, dpi=300)
    plt.close()
    print(f"Saved '{save_filename}'")

    # --- 플롯 2: Positive Components (Comp 2) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # BENv2 Positive PDF (phi-없이)
    if benv2_params['std_pos'] > 0: # Check if it was fitted
        pdf_pos_benv2 = norm.pdf(x_plot, loc=benv2_params['mean_pos'], scale=benv2_params['std_pos'])
        plt.plot(x_plot, pdf_pos_benv2, color='red', linestyle='--', lw=2, 
                 label=f"BENv2 Positive (μ={benv2_params['mean_pos']:.2f}, σ={benv2_params['std_pos']:.2f})")

    # SEN12 Positive PDF (phi-없이)
    if sen12_params['std_pos'] > 0: # Check if it was fitted
        pdf_pos_sen12 = norm.pdf(x_plot, loc=sen12_params['mean_pos'], scale=sen12_params['std_pos'])
        plt.plot(x_plot, pdf_pos_sen12, color='orange', linestyle='-', lw=2, 
                 label=f"SEN12 Positive (μ={sen12_params['mean_pos']:.2f}, σ={sen12_params['std_pos']:.2f})")

    plt.title('Comparison of Positive Components (Unweighted, phi-없이)', fontsize=16, weight='bold')
    plt.xlabel('Predicted Probability (z)', fontsize=12)
    plt.ylabel('Density (Unweighted)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    save_filename = os.path.join(SAVE_FIG_FOLDER, "comparison_positive_components.png")
    plt.savefig(save_filename, dpi=300)
    plt.close()
    print(f"Saved '{save_filename}'")


# -----------------------------------------------------------------
# [수정됨] BENv2 주석 해제 및 비교 플롯 호출
# -----------------------------------------------------------------
def run_inference():

    # --- 공통 변환 정의 ---
    transform_opt_val = make_transform(resize_size=RESIZE_SIZE, data_type="opt", dataset = "benv2", is_train=False)
    transform_sar_val = make_transform(resize_size=RESIZE_SIZE, data_type="sar", dataset = "benv2", is_train=False)

    transform_sar_sen12_val = make_transform(resize_size=RESIZE_SIZE, data_type="sar", dataset = "sen12", is_train=False)
    transform_opt_sen12_val = make_transform(resize_size=RESIZE_SIZE, data_type="opt", dataset = "sen12", is_train=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 모델 로드 (공통) ---
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
    print("--- 1. Preparing BENv2 (Validation Set) ---")
    
    # [수정] BENv2 평가 주석 해제
    datapath = {
        "images_lmdb": "/home/hyunseo/workspace/rico-hdl/Encoded-BigEarthNet",
        "metadata_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata.parquet",
        "metadata_snow_cloud_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
    }
    ds_benv2 = BENv2_DataSet.BENv2DataSet(
        data_dirs=datapath,
        img_size=(12, 120, 120),
        split='test',
        transform={"opt": transform_opt_val, "sar": transform_sar_val}, # 개별 변환 전달
        merge_patch=False,
        max_len=5000, # SEN12와 샘플 수를 맞추기 위해 5000개로 제한 (옵션)
    )

    loader_benv2 = DataLoader(
        ds_benv2, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Loaded {len(ds_benv2)} BENv2 test images.")
    
    # [호출 1] BENv2 평가 실행 (opt 모델, 레이블 있음)
    # [수정] GMM 파라미터 반환 받기
    benv2_params = evaluate_dataset(model, loader_benv2, device, 
                                    dataset_name="BENv2", 
                                    data_type=DATA_TYPE,
                                    has_labels=True, 
                                    label_size=LABEL_SIZE)

    print("-" * 40)
    
    # --- 2. SEN1-2 평가 (레이블 없음) ---
    print("--- 2. Preparing SEN1-2 (Unlabeled Set) ---")
    sen12_params = None # [NEW] 파라미터 변수 초기화
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
        

        sen12_params = evaluate_dataset(model, loader_sen12_val, device, 
                                        dataset_name="SEN12_Val", 
                                        data_type=DATA_TYPE,
                                        has_labels=False)

    print("-" * 40)
    
    # --- 3. [NEW] 비교 플롯 생성 ---
    if benv2_params and sen12_params:
        plot_comparison_pdfs(benv2_params, sen12_params)
    else:
        print("Skipping comparison plots because one or both dataset evaluations failed.")


if __name__ == "__main__":
    run_inference()