#!/usr/bin/env bash

nvidia-smi

# 切换到当前文件所在目录，并使其能够显示结果的命令
cd $(dirname $0) 
set -x

# 定义变量
GPUS=""
PARAMS=""

# 解析命令行参数
while [ $# -gt 0 ]; do
    case "$1" in
        --GPUS)
            GPUS="$2"
            shift 2
            ;;
        *)
            PARAMS="$PARAMS $1"
            shift
            ;;
    esac
done

# 提取GPUS后的一位数字
GPU_NUMBER=$(echo "$GPUS" | sed 's/.*\([0-9]\)$/\1/')

# 拼接参数
PARAMS="$PARAMS"

# 输出结果
echo "GPUS: $GPU_NUMBER"
echo "PARAMS: $PARAMS"

# 训练命令dist_train.sh 为 bevfusion仅lidar的多卡训练启动文件脚本
/ai/volume/mmsegmentation/tools/dist_train_cul.sh /ai/volume/mmsegmentation/configs/culane/culane_xt.py ${GPU_NUMBER} ${PARAMS}