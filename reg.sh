#!/bin/bash

# 流式处理ClimbLab数据集的运行脚本

echo "开始流式处理ClimbLab数据集..."


# 运行流式处理脚本
# 参数说明:
# --output_file: 输出JSON文件路径
# --max_tokens: 最大token数量限制（1M = 1,000,000）
# --log_level: 日志级别

python3 /home/ubuntu/ELM/LMFlow/register.py \
    --output_file /home/ubuntu/ELM/LMFlow/data/rand/climblab_2M_tokens.json \
    --max_tokens 2000000 \
    --log_level INFO

echo "处理完成！"
