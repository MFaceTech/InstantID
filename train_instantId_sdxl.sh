# SDXL Model
export MODEL_NAME="huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/"
# CLIP Model
export ENCODER_NAME="IP-Adapter/sdxl_models/image_encoder"
# pretrained InstantID model
export ADAPTOR_NAME="InstantID/checkpoints/ip-adapter.bin"
export CONTROLNET_NAME="InstantID/checkpoints/ControlNetModel"

# Dataset
export ROOT_DATA_DIR="/"
# This json file ' format:
# {"file_name": "/data/train_data/images_part0/84634599103.jpg", "additional_feature": "myolv1,a man with glasses and a
# tie on posing for a picture in front of a window with a building in the background, Andrew Law, johnson ting, a picture,
# mannerism", "bbox": [-31.329412311315536, 160.6865997314453, 496.19240215420723, 688.1674156188965],
# "landmarks": [[133.046875, 318], [319.3125, 318], [221.0625, 422], [153.515625, 535], [298.84375, 537]],
# "insightface_feature_file": "/data/feature_data/images_part0/84634599103.bin"}
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


