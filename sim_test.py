import torch
import open_clip
from PIL import Image, ImageEnhance, Image as PILImage # ImageEnhance 추가, PILImage 추가
import numpy as np

# --- 설정 (Configuration) ---
# 사용할 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 모델 가중치 경로
COCA_CHECKPOINT_PATH = "/home/hyunseo/workspace/kari/SAR-RS-CoCa.pt"
CLIP_CHECKPOINT_PATH = "/home/hyunseo/workspace/kari/SAR-RS-CLIP.pt"

# 평가할 SAR 이미지 경로 (예시)
# 이 부분을 실제 평가하고 싶은 SAR 이미지 경로로 변경하세요.
IMAGE_PATH = "/home/hyunseo/workspace/kari/SeeSR/4.png"

# 밝기 조절 계수 (1.0 = 원본, 1.5 = 50% 밝게)
BRIGHTNESS_FACTOR = 1.0 # 기본값은 원본 밝기

# 리사이즈 크기 설정
RESIZE_DIM = (512, 512)

# --- 모델 로딩 함수 ---
# (load_coca_model 함수는 변경 없음)
def load_coca_model(checkpoint_path, device):
    """SAR-RS-CoCa 모델을 로드하는 함수"""
    print("Loading SAR-RS-CoCa model for captioning...")
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained=checkpoint_path,
        device=device,
        weights_only=False # 전체 모델 구조를 불러오기 위해 False로 설정
    )
    model.eval()
    print("CoCa model loaded.")
    return model, transform

# (load_clip_model 함수는 변경 없음)
def load_clip_model(checkpoint_path, device):
    """SAR-RS-CLIP 모델을 로드하는 함수"""
    print("Loading SAR-RS-CLIP model for similarity scoring...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name='ViT-L-14',
        pretrained='openai', # 기본 구조는 openai pretrained로 시작
        device=device,
        cache_dir='cache/weights/open_clip'
    )

    # 학습된 가중치 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 텍스트 토크나이저
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    print("CLIP model loaded.")
    return model, preprocess, tokenizer

# --- 메인 기능 함수 ---

def generate_caption(image_path, coca_model, coca_transform, device, resize_dim=(512, 512), brightness_factor=1.0):
    """주어진 이미지 경로에 대해 리사이즈, 밝기 조절 후 CoCa 모델로 캡션을 생성하는 함수"""
    print(f"\nGenerating caption for: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")

        # --- 리사이즈 적용 ---
        print(f"  Resizing image to {resize_dim}...")
        # PIL.Image.Resampling.BICUBIC 사용 (최신 PIL 권장)
        # 또는 PILImage.BICUBIC (이전 버전)
        try:
            image_resized = image.resize(resize_dim, resample=PILImage.Resampling.BICUBIC)
        except AttributeError: # 이전 PIL 버전 호환성
             image_resized = image.resize(resize_dim, resample=PILImage.BICUBIC)
        print("  Image resized.")
        # --- 리사이즈 끝 ---

        # --- 밝기 조절 시작 (리사이즈된 이미지에 적용) ---
        if brightness_factor != 1.0:
            print(f"  Adjusting brightness by factor: {brightness_factor}")
            enhancer = ImageEnhance.Brightness(image_resized)
            image_final = enhancer.enhance(brightness_factor) # 최종 사용할 이미지
            print("  Brightness adjusted.")
        else:
            image_final = image_resized # 밝기 조절 안 하면 리사이즈된 이미지가 최종
        # --- 밝기 조절 끝 ---
        image_final.save('./final_processed_image.png') # 최종 입력 이미지 저장 (확인용)


        # 최종 이미지를 텐서로 변환
        image_tensor = coca_transform(image_final).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            # generation_type을 다시 top_k나 top_p 등으로 설정할 수 있습니다.
            generated = coca_model.generate(image_tensor, generation_type="beam_search", seq_len=40)

        caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "").strip()

        print(f"  -> Generated Caption: '{caption}'")
        return caption
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e: # 다른 에러 처리 추가
        print(f"An error occurred during caption generation: {e}")
        return None

# (calculate_similarity 함수는 변경 없음)
def calculate_similarity(image_path, caption, clip_model, clip_preprocess, clip_tokenizer, device):
    """주어진 이미지와 캡션 간의 CLIP 유사도를 계산하는 함수 (원본 이미지 사용)"""
    if caption is None:
        return None

    print(f"Calculating similarity between original image and caption...")
    try:
        # 유사도 계산 시에는 원본 이미지를 사용합니다.
        image = Image.open(image_path).convert("RGB")
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        text_tokens = clip_tokenizer([caption]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)[0]
            image_features2 = clip_model.encode_image(image_tensor)[1]
            print('image_features shape', image_features.shape)
            print('image_features2 shape', image_features2.shape)

            
            text_features = clip_model.encode_text(text_tokens)

            # 정규화 (중요)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 코사인 유사도 계산 (dot product)
            similarity_score = (image_features @ text_features.T).item()

        print(f"  -> CLIP Similarity Score: {similarity_score:.4f}")
        return similarity_score
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # --- (모델 로드 부분은 동일) ---
    coca_model, coca_transform = load_coca_model(COCA_CHECKPOINT_PATH, DEVICE)
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model(CLIP_CHECKPOINT_PATH, DEVICE)

    # 2. 캡션 생성 (리사이즈 및 밝기 조절 적용)
    generated_caption = generate_caption(IMAGE_PATH, coca_model, coca_transform, DEVICE, resize_dim=RESIZE_DIM, brightness_factor=BRIGHTNESS_FACTOR)

    # 3. 유사도 계산 (원본 이미지와 생성된 캡션 간)
    if generated_caption:
        # 여기에 비교할 캡션을 직접 넣거나, 생성된 캡션을 사용할 수 있습니다.
        comparison_caption = generated_caption
        similarity = calculate_similarity(IMAGE_PATH, comparison_caption, clip_model, clip_preprocess, clip_tokenizer, DEVICE)

        if similarity is not None:
            print("\n--- Final Result ---")
            print(f"Image: {IMAGE_PATH}")
            print(f"Generated Caption (from resized/brightened image): '{generated_caption}'")
            print(f"Original Image-Caption Similarity: {similarity:.4f}")
            print("--------------------")