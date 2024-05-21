unset LD_LIBRARY_PATH
source /usr/local/conda/bin/activate adaptor

# SDXL Model
export MODEL_NAME="huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/"
# CLIP Model
export ENCODER_NAME="IP-Adapter/sdxl_models/image_encoder"
# pretrained InstantID model
export ADAPTOR_NAME="InstantID/checkpoints/ip-adapter.bin"
export CONTROLNET_NAME="InstantID/checkpoints/ControlNetModel"

# Dataset
export ROOT_DATA_DIR="/"
export JSON_FILE="aigc_data/index_files/mt_portrait_dataset.json"

# Output
export OUTPUT_DIR="InstantID_SDXL/output/test"


echo "OUTPUT_DIR: $OUTPUT_DIR"
#accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
#CUDA_VISIBLE_DEVICES=0 \

accelerate launch --mixed_precision="fp16" train_instantId_sdxl.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --controlnet_model_name_or_path $CONTROLNET_NAME \
  --image_encoder_path $ENCODER_NAME \
  --pretrained_ip_adapter_path $ADAPTOR_NAME \
  --data_root_path $ROOT_DATA_DIR \
  --data_json_file $JSON_FILE \
  --output_dir $OUTPUT_DIR \
  --clip_proc_mode orig_crop \
  --mixed_precision="fp16" \
  --resolution 1024 \
  --learning_rate 1e-5 \
  --weight_decay=0.01 \
  --num_train_epochs 20 \
  --train_batch_size 2 \
  --dataloader_num_workers=8 \
  --checkpoints_total_limit 10 \
  --save_steps 10000


