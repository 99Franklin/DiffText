export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export TRAIN_DIR="/home/lfu/project/preprocessing_scene_text/text_diffusion_data_v3"

NCCL_DEBUG=INFO accelerate launch --main_process_port 32345 --mixed_precision="fp16" src/engines/finetune_text_to_image_inpainting_with_single_adapter_with_fussion_te.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--train_data_dir=$TRAIN_DIR \
	--cache_dir="/data2/lfu/cache_dir/diffusers" \
	--dataloader_num_workers 0 \
	--use_ema \
	--resolution=512 --center_crop --random_flip \
	--train_batch_size=16 \
	--gradient_accumulation_steps=4 \
	--gradient_checkpointing \
	--enable_xformers_memory_efficient_attention \
	--num_train_epochs=50 \
	--learning_rate=1e-05 \
	--max_grad_norm=1 \
	--lr_scheduler="constant" --lr_warmup_steps=0 \
	--output_dir="output/0408_inpainting_with_adapter_with_second_fussion_te"
