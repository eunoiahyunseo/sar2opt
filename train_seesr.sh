CUDA_VISIBLE_DEVICES="0" accelerate launch train_seesr.py \
--pretrained_model_name_or_path="/home/hyunseo/workspace/kari/SeeSR2/stable-diffusion-2-base" \
--output_dir="./output" \
--ram_ft_path='./DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=256 \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=32 \
--null_text_ratio=0.5 \
--dataloader_num_workers=4 \
--checkpointing_steps=10 \
--max_train_steps=50000 \
--report_to wandb \
--lr_scheduler "cosine" \
--lr_warmup_steps=100 \
--warmup_steps=10 \
# --root_folders 'preset/datasets/training_datasets' \