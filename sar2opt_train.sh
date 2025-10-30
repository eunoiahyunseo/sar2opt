# python test_seesr_turbo.py \
# --pretrained_model_path /home/hyunseo/workspace/kari/SeeSR/sd-turbo \
# --prompt '' \
# --seesr_model_path preset/models/seesr \
# # --ram_ft_path preset/models/DAPE.pth \
# --image_path preset/datasets/test_datasets \
# --output_dir /home/hyunseo/workspace/kari/SeeSR/output \
# --start_point lr \
# --num_inference_steps 2 \
# --guidance_scale 1.0 \
# --process_size 512 



CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch train_seesr.py \
--pretrained_model_name_or_path="/home/hyunseo/workspace/kari/SeeSR2/stable-diffusion-2-base" \
--output_dir="./output" \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=256 \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=1 \
--warmup_steps=10 \
# --validation_steps=500 \
# --root_folders '/home/hyunseo/workspace/kari/SeeSR/datasets/training_datasets' \
# --ram_ft_path 'preset/models/DAPE.pth' \