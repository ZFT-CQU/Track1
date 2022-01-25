INPUT_PATH="test"
OUTPUT_PATH="output"

CUDA_VISIBLE_DEVICES=0 python3 process.py \
                                  $INPUT_PATH \
                                  $OUTPUT_PATH \
                                  --batch-size 8 \
                                  --epochs 30 \
                                  --test