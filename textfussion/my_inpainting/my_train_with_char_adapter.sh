export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export TRAIN_DIR="/data/lfu/datasets/scene_text_detection/for_diffusers_inpainting_char"

NCCL_DEBUG=INFO accelerate launch --mixed_precision="fp16" src/engines/finetune_text_to_image_inpainting_with_char_adapter.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--train_data_dir=$TRAIN_DIR \
	--cache_dir="/data/lfu/cache_dir/diffusers" \
	--dataloader_num_workers 16 \
	--use_ema \
	--resolution=512 --center_crop --random_flip \
	--train_batch_size=24 \
	--gradient_accumulation_steps=4 \
	--gradient_checkpointing \
	--enable_xformers_memory_efficient_attention \
	--num_train_epochs=2 \
	--learning_rate=1e-06 \
	--max_grad_norm=1 \
	--lr_scheduler="constant" --lr_warmup_steps=0 \
	--output_dir="output/0327_inpainting_with_char_adapter_char_text_guide_epoch_2e_lr_1e-6"
