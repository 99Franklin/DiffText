export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export OUTPUT_DIR="output/pretrain_8702_text_vae"

NCCL_P2P_DISABLE=1 accelerate launch train_vae.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--output_dir=$OUTPUT_DIR \
	--resolution=512 \
	--train_batch_size=4 \
	--gradient_accumulation_steps=1 \
	--gradient_checkpointing \
	--learning_rate=5e-6 \
	--num_train_epochs=3 \
	--lr_scheduler="constant" \
	--lr_warmup_steps=3000 \
	--dataloader_num_workers=8 \
	--mixed_precision=fp16 \
