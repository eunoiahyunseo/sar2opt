from torchvision.transforms import v2
import torch




def make_transform(is_train: bool = True, data_type: str = "opt", dataset: str = "benv2", resize_size: int = 256):
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)
    
    if data_type == "opt":
        transforms_list = [
            v2.ToTensor(),
            v2.Resize((resize_size, resize_size)),
            v2.Lambda(lambda x: torch.clamp(x / 10000.0, 0.0, 1.0)) if dataset == "benv2" else v2.Lambda(lambda x: x),
        ]
    elif data_type == "sar":
        transforms_list = [
            v2.ToTensor(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.Lambda(lambda x: torch.clamp((x + 30.0) / 40.0, 0.0, 1.0)) if dataset == "benv2" else v2.Lambda(lambda x: x),
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