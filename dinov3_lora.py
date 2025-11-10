import torch
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.transforms import v2
import glob
from types import SimpleNamespace

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from configilm import util

from configilm.extra.DataSets import BENv2_DataSet
import torchvision.transforms as T

import wandb
import uuid

import torchmetrics

from torch.autograd import Function
from typing import Optional, Any, Tuple
from sklearn.mixture import GaussianMixture

from dda_mlic import GMM_Discrepancy, WarmStartGradientReverseLayer, GradientReverseFunction
from utils.loss import mld_loss
from utils.transforms import make_transform


dist.init_process_group("nccl")
local_rank = int(os.environ['LOCAL_RANK'])
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)
    
STAGE = 0
DATA_TYPE = "sar"
LABEL_SIZE = 19

CHECKPOINT_DIR = f"./checkpoints/stage{STAGE}_{DATA_TYPE}"
RESUME_CHECKPOINT_PATH = None
start_epoch = 1
checkpoint = None
wandb_run_id = None

INFERENCE_CHECKPOINT_PATH = "./checkpoints/stage0_sar/checkpoint_stage0_epoch101.pth"
TEACHER_CHECKPOINT_PATH = "./checkpoints/stage0_opt/checkpoint_stage0_epoch101.pth"

if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
    checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location='cpu')
    start_epoch = checkpoint['epoch']
    wandb_run_id = checkpoint['wandb_run_id']
    if local_rank == 0:
        print(f"Found checkpoint. Resuming from {RESUME_CHECKPOINT_PATH}")
        print(f"Resuming epoch {start_epoch} and wandb run ID {wandb_run_id}")
else:
    if local_rank == 0:
        print("No checkpoint found, starting from scratch.")

MODEL_NAME = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
REPO_NAME = "/home/hyunseo/workspace/dinov3"  
DATASET_ROOT_DIR = "/home/hyunseo/workspace/sar2opt/dataset/QXSLAB_SAROPT"
datapath = {
    "images_lmdb": "/home/hyunseo/workspace/rico-hdl/Encoded-BigEarthNet",
    "metadata_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/home/hyunseo/workspace/sar2opt/dataset/BigEarthNetv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv"]

DATASET_NAME = "QXSLAB_SAR2OPT"
BATCH_SIZE = 112  # DDP 사용 시: "GPU당" 배치 크기가 됨
LEARNING_RATE = 1e-4
MERGE_PATCH = False
NUM_EPOCHS = 100
LAMBDA_MMD = 0.5
RESIZE_SIZE = 256 if MERGE_PATCH else 128
VIS_BATCH_SIZE_TOTAL = BATCH_SIZE * 2 # DDP 월드 크기(GPU 개수)만큼 배분됨
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# [NEW] PCA를 위한 모델별 레이어 수 정의 (참고 코드 기반)
# DINOv3 ViT-H/16은 32개의 레이어를 가집니다.
MODEL_TO_NUM_LAYERS = {
    "facebook/dinov3-vits14": 12,
    "facebook/dinov3-vitb14": 12,
    "facebook/dinov3-vitl14": 24,
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": 32, # 사용자 모델
}


LAYERS_TO_EXTRACT = [25, 28, 31]  # 추출할 레이어 인덱스 (0부터 시작)
VALIDATION_INTERVAL_STEPS = 3000

# [Scheduler] Warmup 및 스케줄러 관련 하이퍼파라미터
WARMUP_STEPS = 500      # 훈련 초기 Warmup을 진행할 스텝 수
T_MAX = NUM_EPOCHS      # Cosine Annealing의 주기 (전체 에포크)
ETA_MIN = 1e-6          # 최소 학습률


UDA_ENABLED = True if STAGE == 1 else False # STAGE 1에서만 UDA 활성화
SEN12_ROOT_DIR = "/path/to/your/SEN1-2/SAR/images" # !! SEN1-2 SAR 이미지 경로 설정 필요 !!
LAMBDA_ADV = 0.1  # Adversarial loss 가중치
GMM_REG_0 = 0.5   # GMM C0 클러스터 가중치 (논문의 alpha_1)
GMM_REG_1 = 0.5   # GMM C1 클러스터 가중치 (논문의 alpha_2)


class DinoV3Linear(nn.Module):
    def __init__(self, backbone, hidden_size: int, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.head = nn.Linear(hidden_size, num_classes)

    def forward_features(self, pixel_values):
        # Backbone을 통해 feature만 추출
        return self.backbone(pixel_values)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        logits = self.head(outputs)
        return logits





if local_rank == 0:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if wandb_run_id:
        wandb.init(
            project="sar2opt-dinov3", 
            id=wandb_run_id,
            resume="must",
        )
    else:
        # 새 실행
        random_id = uuid.uuid4().hex[:8]
        wandb.init(
            project="sar2opt-dinov3", 
            name=f"stage_{STAGE}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}_{random_id}", 
            resume="allow",
            tags=[DATA_TYPE]

        )
    print(f"Using device: {device} (DDP local_rank: {local_rank})")


def validate_and_log(model, val_loader, device, num_classes, global_step):
    model.eval()
    
    # 1. 메트릭 객체 생성 (모든 Rank에서 동일하게)
    apm_metric = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average='macro').to(device)
    apmu_metric = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average='micro').to(device)
    fm1_metric = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='macro').to(device)
    fmu1_metric = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='micro').to(device)

    val_progress_bar = tqdm(val_loader, desc="Validating", disable=(dist.get_rank() != 0))
    
    with torch.inference_mode():
        for img, lbl in val_progress_bar:
            if STAGE == 0:
                inputs = img[DATA_TYPE].to(device)
            elif STAGE == 1:
                inputs = img["sar"].to(device)
            else:
                 raise ValueError(f"Invalid STAGE: {STAGE}")
            
            labels = lbl.to(device).int() 
            
            with torch.cuda.amp.autocast():
                logits = model(inputs)
            
            preds = torch.sigmoid(logits)
            
            # 2. 모든 Rank가 .update() 호출
            apm_metric.update(preds, labels)
            apmu_metric.update(preds, labels)
            fm1_metric.update(preds, labels)
            fmu1_metric.update(preds, labels)

    # 3. 모든 Rank가 .update()를 마칠 때까지 대기 (선택 사항이지만 안전함)
    dist.barrier() 
    
    # 4. [중요] .compute()는 "if rank == 0" *밖에서* 모든 Rank가 호출해야 함
    # DDP 동기화가 여기서 발생합니다.
    apm = apm_metric.compute()
    apmu = apmu_metric.compute()
    fm1 = fm1_metric.compute()
    fmu1 = fmu1_metric.compute()

    # 5. [중요] .reset()도 모든 Rank가 호출해야 함
    apm_metric.reset()
    apmu_metric.reset()
    fm1_metric.reset()
    fmu1_metric.reset()
    
    # 6. 로깅 및 출력은 Rank 0에서만 수행
    if dist.get_rank() == 0:
        print("\n--- Calculating Metrics on Rank 0 ---")
        print(f"Validation Step: {global_step}")
        print(f"APM: {apm:.4f}, APμ: {apmu:.4f}, FM1: {fm1:.4f}, Fμ1: {fmu1:.4f}")

        wandb.log({
            "validation/APM": apm,
            "validation/APμ": apmu,
            "validation/FM1": fm1,
            "validation/Fμ1": fmu1,
        }, step=global_step)
        
    # 7. 모든 Rank가 로깅까지 마칠 때까지 대기
    dist.barrier()
    model.train()

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
)
cls_feature_size = torch.hub.load(REPO_NAME, 'dinov3_vith16plus', source='local').num_features

if STAGE == 0:
    backbone = torch.hub.load(REPO_NAME, 'dinov3_vith16plus', source='local', weights='/home/hyunseo/workspace/sar2opt/SAR2OPT/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    model = DinoV3Linear(backbone, hidden_size=cls_feature_size, num_classes=LABEL_SIZE, freeze_backbone=True)
    model = get_peft_model(model, lora_config).to(device)
    model = DDP(model, device_ids=[local_rank])
    if local_rank == 0:
        model.module.print_trainable_parameters()

elif STAGE == 1:
    if local_rank == 0:
        print(f"--- STAGE 1: Multi-Label Logits Distillation ---")
        print(f"Loading Teacher model from {TEACHER_CHECKPOINT_PATH}")

    teacher_backbone = torch.hub.load(REPO_NAME, 'dinov3_vith16plus', source='local')
    teacher_model = DinoV3Linear(teacher_backbone, hidden_size=cls_feature_size, num_classes=LABEL_SIZE, freeze_backbone=True)
    teacher_model = get_peft_model(teacher_model, lora_config).to(device)
    
    teacher_ckpt = torch.load(TEACHER_CHECKPOINT_PATH, map_location=device)
    teacher_model.load_state_dict(teacher_ckpt['model_state_dict'])
    
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    

    student_backbone = torch.hub.load(REPO_NAME, 'dinov3_vith16plus', source='local', weights='/home/hyunseo/workspace/sar2opt/SAR2OPT/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    model = DinoV3Linear(student_backbone, hidden_size=cls_feature_size, num_classes=LABEL_SIZE, freeze_backbone=True)
    model = get_peft_model(model, lora_config).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    gmm_args = SimpleNamespace(reg_0=GMM_REG_0, reg_1=GMM_REG_1)
    adv_loss_fn = GMM_Discrepancy(classifier=model.module.head, args=gmm_args)

    if local_rank == 0:
        model.module.print_trainable_parameters()


transform = {
    "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=True),
    "sar": make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=True)
}

transform_val = {
    "opt": make_transform(resize_size=RESIZE_SIZE, data_type="opt", is_train=False),
    "sar": make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=False)
}

class SEN12_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 지원하는 이미지 확장자 리스트
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        if local_rank == 0 and not self.image_paths:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 1채널 SAR 이미지를 3채널로 복사하여 사용
        image = Image.open(img_path).convert("L")
        image = Image.merge("RGB", (image, image, image))
        
        if self.transform:
            image = self.transform(image)
            
        return image


train_dataset = BENv2_DataSet.BENv2DataSet(
    data_dirs=datapath,
    img_size=(12, 120, 120),
    split='train',
    transform=transform,
    merge_patch=MERGE_PATCH
)

val_dataset = BENv2_DataSet.BENv2DataSet(
    data_dirs=datapath,
    img_size=(12, 120, 120),
    split='test',
    transform=transform_val,
    merge_patch=MERGE_PATCH
)

train_sampler = DistributedSampler(train_dataset, shuffle=True)
val_sampler = DistributedSampler(val_dataset, shuffle=False) # 검증 시에는 셔플 불필요

train_dataloader = DataLoader(train_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True, 
                        sampler=train_sampler)
                        
val_dataloader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE * 2,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        sampler=val_sampler)


if UDA_ENABLED:
    sen12_transform = make_transform(resize_size=RESIZE_SIZE, data_type="sar", is_train=True)
    sen12_dataset = SEN12_Dataset(root_dir=SEN12_ROOT_DIR, transform=sen12_transform)
    sen12_sampler = DistributedSampler(sen12_dataset, shuffle=True)
    # 소스 배치와 타겟 배치의 크기를 동일하게 맞춤
    sen12_dataloader = DataLoader(sen12_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=sen12_sampler)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
task_criterion = mld_loss if STAGE == 1 else nn.BCEWithLogitsLoss()

scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

scaler = torch.cuda.amp.GradScaler()

if checkpoint:
    # 현재 훈련 대상인 'model'(Student)의 state_dict를 로드
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if local_rank == 0:
        print(f"Successfully loaded model, optimizer, and scaler state from checkpoint.")
    del checkpoint
dist.barrier()


if local_rank == 0:
    print("\n--- Starting Training (DDP + AMP) ---")
    n_layers = MODEL_TO_NUM_LAYERS.get(MODEL_NAME)
    if n_layers is None:
        print(f"Warning: MODEL_NAME '{MODEL_NAME}' not in MODEL_TO_NUM_LAYERS. Defaulting to 32 layers.")
        n_layers = 32
    else:
        print(f"Found {n_layers} layers for model {MODEL_NAME}.")

global_step = (start_epoch - 1) * len(train_dataloader)

for epoch in range(start_epoch, start_epoch + NUM_EPOCHS, 1):
    model.train()
    
    train_sampler.set_epoch(epoch)

    if UDA_ENABLED:
        sen12_sampler.set_epoch(epoch)
        sen12_iter = iter(sen12_dataloader)
    
    total_loss_epoch, total_task_loss_epoch, total_adv_loss_epoch = 0, 0, 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{start_epoch + NUM_EPOCHS}", disable=(local_rank != 0))
    
    for i, (img, lbl) in enumerate(progress_bar):
        if global_step < WARMUP_STEPS:
            lr_scale = global_step / WARMUP_STEPS
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * lr_scale

        optimizer.zero_grad()

                
        if STAGE == 1:
            # 1. 데이터 준비
            source_sar_inputs = img["sar"].to(device)
            source_opt_inputs = img["opt"].to(device)
            B = source_sar_inputs.shape[0] # 실제 배치 사이즈

            if UDA_ENABLED:
                try:
                    target_sar_inputs = next(sen12_iter)
                except StopIteration:
                    sen12_iter = iter(sen12_dataloader)
                    target_sar_inputs = next(sen12_iter)
                target_sar_inputs = target_sar_inputs.to(device)

                # 타겟 배치가 소스 배치보다 작을 경우 스킵 (DDP에서 마지막 배치)
                if target_sar_inputs.shape[0] != B:
                    continue
                
                # 소스 + 타겟 SAR 이미지 결합
                combined_sar_inputs = torch.cat([source_sar_inputs, target_sar_inputs], dim=0)

            with torch.cuda.amp.autocast():
                # 2. 순전파
                if UDA_ENABLED:
                    # 소스+타겟 특징 추출
                    combined_features = model.module.forward_features(combined_sar_inputs)
                    source_features = combined_features[:B]
                else: # UDA 비활성화 시
                    source_features = model.module.forward_features(source_sar_inputs)

                # 3. Task Loss (KD) 계산
                student_logits = model.module.head(source_features)
                with torch.no_grad():
                    teacher_logits = teacher_model(source_opt_inputs)
                task_loss = task_criterion(student_logits, teacher_logits)
                
                # 4. Adversarial Loss (UDA) 계산
                if UDA_ENABLED:
                    adv_loss = adv_loss_fn(combined_features, B)
                    total_loss = task_loss + LAMBDA_ADV * adv_loss
                else:
                    adv_loss = torch.tensor(0.0)
                    total_loss = task_loss
        else: 
            inputs = img[DATA_TYPE].to(device)
            labels = lbl.to(device)
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                total_loss = task_criterion(logits, labels.float())
                task_loss, adv_loss = total_loss, torch.tensor(0.0)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss_log = total_loss.detach()
        task_loss_log = task_loss.detach() 
        # STAGE 0에서 adv_loss는 어차피 tensor(0.0)이므로 UDA_ENABLED로 분기합니다.

        # 분리된 텐서로 all_reduce를 수행합니다.
        dist.all_reduce(total_loss_log, op=dist.ReduceOp.AVG)
        dist.all_reduce(task_loss_log, op=dist.ReduceOp.AVG)
        if UDA_ENABLED:
            adv_loss_log = adv_loss.detach() # STAGE 1일 경우도 대비
            dist.all_reduce(adv_loss_log, op=dist.ReduceOp.AVG) 
        # 분리된 텐서의 값을 누적합니다.
        total_loss_epoch += total_loss_log.item()
        total_task_loss_epoch += task_loss_log.item()
        
        # STAGE 0의 adv_loss는 CPU 스칼라이므로 .item()을 바로 써도 됩니다.
        if UDA_ENABLED:
            total_adv_loss_epoch += adv_loss_log.item()
        else:
            total_adv_loss_epoch += adv_loss.item()  
        global_step += 1

        
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'task_loss': f'{task_loss.item():.4f}',
                'adv_loss': f'{adv_loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/task_loss": task_loss.item(),
                "train/adv_loss": adv_loss.item(),
                "train/learning_rate": current_lr
            }, step=global_step)

    avg_loss = total_loss_epoch / len(train_dataloader)

    if global_step >= WARMUP_STEPS:
        scheduler.step()
    
    validate_and_log(model, val_dataloader, device, LABEL_SIZE, global_step)

    if local_rank == 0:
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        wandb.log({"train/avg_loss": avg_loss, "epoch": epoch + 1})

        # 현재 훈련 중인 모델(model)을 저장
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_stage{STAGE}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'wandb_run_id': wandb.run.id
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    dist.barrier()  

if local_rank == 0:
    print("\n--- Training Finished ---")

dist.destroy_process_group()