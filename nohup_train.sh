#!/bin/sh
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
TMP_RESULTS_DIR="$(pwd)/.tmp_results/${TIMESTAMP}"
ARGS_FILE="${TMP_RESULTS_DIR}/args.yaml"
TRAIN_LOG_FILE="${TMP_RESULTS_DIR}/log.txt"
CFG_FILE_PATH="$(pwd)/config/conv_ae_with_cnn.yaml" 
# CFG_FILE_PATH="$(pwd)/config/conv_autoencoder.yaml" 
# CFG_FILE_PATH="$(pwd)/config/simple_cnn.yaml" 
mkdir -p $TMP_RESULTS_DIR

export CUDA_VISIBLE_DEVICES=0
nohup python -u train.py \
  --args_file_path $ARGS_FILE \
  --cfg_file_path $CFG_FILE_PATH \
  --tmp_results_dir $TMP_RESULTS_DIR \
  --train_log_file_path $TRAIN_LOG_FILE \
  > $TRAIN_LOG_FILE &
#  > $TRAIN_LOG_FILE 2>&1 &
sleep 1s
tail -f $TRAIN_LOG_FILE
