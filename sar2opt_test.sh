# python test_sar2opt.py \
#   --trained_model_path "./output/checkpoint-1" \
#   --pretrained_model_path "/home/hyunseo/workspace/kari/SeeSR/stable-diffusion-2-base" \
#   --vit_model_path "../SAR-RS-CLIP.pt" \
#   --image_path "./datasets/train_datasets/QXSLAB_SAROPT" \
#   --output_dir "./results"



python test_seesr.py \
--pretrained_model_path "/home/hyunseo/workspace/kari/SeeSR2/stable-diffusion-2-base" \
--seesr_model_path "./output/checkpoint-9000" \
--output_dir './output_image' \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--latent_tiled_size 9999 \
--latent_tiled_overlap 0
# --image_path p/reset/datasets/test_datasets \
# --prompt '' \
# --start_point lr \
# --process_size 256
# --ram_ft_path preset/models/DAPE.pth \