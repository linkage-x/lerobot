#!/bin/bash

set -euo pipefail

SCRIPT_NAME=$(basename "$0")

print_usage() {
  cat <<EOF
用法:
  $SCRIPT_NAME -d DATA_CFG -t TRAIN_CFG [-g GPU_IDS] [-n GPU_COUNT] [-y]

说明:
  -d DATA_CFG     数据集配置文件路径 (必填)
  -t TRAIN_CFG    训练配置文件路径   (必填)
  -g GPU_IDS      使用的 GPU ID, 逗号分隔, 默认: "0"
  -n GPU_COUNT    进程数 / GPU 数量, 默认: 1
  -y              不进行交互确认, 直接开始执行
  -h              显示本帮助

也可以保持兼容老用法:
  $SCRIPT_NAME DATA_CFG TRAIN_CFG [GPU_IDS] [GPU_COUNT]

示例:
  $SCRIPT_NAME \\
    -d src/lerobot/datasets/hirol/config/ip_1118_dee2dee.yaml \\
    -t src/lerobot/scripts/train_config/smolvla_ip_1118_dee2dee.json \\
    -g 0,1 -n 2
EOF
}

DATA_CFG=""
TRAIN_CFG=""
GPU_IDS="0"
GPU_COUNT=1
ASSUME_YES=0
DEST_DIR=""

while getopts ":d:t:g:n:o:yh" opt; do
  case "$opt" in
    d) DATA_CFG=$OPTARG ;;
    t) TRAIN_CFG=$OPTARG ;;
    g) GPU_IDS=$OPTARG ;;
    n) GPU_COUNT=$OPTARG ;;
    o) DEST_DIR=$OPTARG ;;
    y) ASSUME_YES=1 ;;
    h)
      print_usage
      exit 0
      ;;
    \?)
      echo "未知参数: -$OPTARG" >&2
      print_usage
      exit 1
      ;;
    :)
      echo "参数 -$OPTARG 需要一个值" >&2
      print_usage
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

# 兼容旧的纯位置参数用法: DATA_CFG TRAIN_CFG [GPU_IDS] [GPU_COUNT]
if [[ -z "$DATA_CFG" && $# -ge 1 ]]; then
  DATA_CFG=$1
  shift
fi
if [[ -z "$TRAIN_CFG" && $# -ge 1 ]]; then
  TRAIN_CFG=$1
  shift
fi
if [[ "$GPU_IDS" == "0" && $# -ge 1 ]]; then
  GPU_IDS=$1
  shift
fi
if [[ "$GPU_COUNT" == 1 && $# -ge 1 ]]; then
  GPU_COUNT=$1
  shift
fi

prompt_if_empty() {
  local var_name=$1
  local prompt_text=$2
  # 间接展开变量值
  local current_value
  current_value=$(eval "printf '%s' \"\${$var_name-}\"")

  if [[ -z "$current_value" ]]; then
    read -rp "$prompt_text: " input
    if [[ -z "$input" ]]; then
      echo "错误: $var_name 不能为空" >&2
      exit 1
    fi
    eval "$var_name=\"\$input\""
  fi
}

validate_positive_int() {
  local name=$1
  local value=$2
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    echo "错误: $name 必须为正整数, 当前为: $value" >&2
    exit 1
  fi
}

# 如未通过参数传入, 则交互式获取
prompt_if_empty DATA_CFG "请输入数据配置文件路径 (DATA_CFG)"
prompt_if_empty TRAIN_CFG "请输入训练配置文件路径 (TRAIN_CFG)"

validate_positive_int "GPU_COUNT" "$GPU_COUNT"

if [[ ! -f "$DATA_CFG" ]]; then
  echo "错误: 数据配置文件不存在: $DATA_CFG" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_CFG" ]]; then
  echo "错误: 训练配置文件不存在: $TRAIN_CFG" >&2
  exit 1
fi

echo "================ 配置信息 ================"
echo "DATA_CFG : $DATA_CFG"
echo "TRAIN_CFG: $TRAIN_CFG"
echo "GPU_IDS  : $GPU_IDS"
echo "GPU_COUNT: $GPU_COUNT"
echo "=========================================="

if [[ $ASSUME_YES -eq 0 ]]; then
  read -rp "确认以上配置并开始执行吗? [y/N]: " confirm
  case "$confirm" in
    y|Y)
      ;;
    *)
      echo "已取消执行。"
      exit 0
      ;;
  esac
fi

echo "[1/3] 加载数据集配置..."
echo "+ python -m src.lerobot.datasets.hirol.lerobot_loader -c \"$DATA_CFG\""
python -m src.lerobot.datasets.hirol.lerobot_loader -c "$DATA_CFG"

echo "[2/3] 开始训练..."
echo "+ CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes $GPU_COUNT src/lerobot/scripts/lerobot_train.py --config_path=\"$TRAIN_CFG\""
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes "$GPU_COUNT" src/lerobot/scripts/lerobot_train.py --config_path="$TRAIN_CFG"

echo "[3/3] 导出模型..."
echo "+ python utils/export_ckpts.py --data_cfg \"$DATA_CFG\" --train_cfg \"$TRAIN_CFG\""

# 暂时关闭 -e, 方便捕获导出脚本的输出和返回码
set +e
EXPORT_OUTPUT=$(python utils/export_ckpts.py --data_cfg "$DATA_CFG" --train_cfg "$TRAIN_CFG" 2>&1)
status=$?
set -e

echo "$EXPORT_OUTPUT"

if [[ $status -ne 0 ]]; then
  echo "导出模型失败, 退出码: $status" >&2
  exit $status
fi

# 从导出脚本输出中解析 "Created zip: <path>"
ZIP_PATH=$(printf '%s\n' "$EXPORT_OUTPUT" | awk -F': ' '/^Created zip:/ {print $2}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

if [[ -z "$ZIP_PATH" || ! -f "$ZIP_PATH" ]]; then
  echo "未能从导出脚本输出中解析到 zip 路径, 或文件不存在。"
  echo "请根据上面的输出手动查找 zip 文件。"
  echo "✅ 全流程完成 (zip 未自动移动)。"
  exit 0
fi

echo "检测到导出 zip: $ZIP_PATH"

# 如果未通过参数指定目标目录, 则可交互式选择是否移动
if [[ -z "$DEST_DIR" ]]; then
  if [[ $ASSUME_YES -eq 0 ]]; then
    read -rp "是否将该 zip 移动到某个目录? [y/N]: " move_confirm
    case "$move_confirm" in
      y|Y)
        read -rp "请输入目标目录 (默认: /data/ckpts): " input_dest
        DEST_DIR=${input_dest:-/data/ckpts}
        ;;
      *)
        echo "跳过移动 zip。"
        echo "✅ 全流程完成。"
        exit 0
        ;;
    esac
  else
    echo "未指定目标目录, 且使用了 -y, 跳过自动移动 zip。"
    echo "✅ 全流程完成 (zip 未移动)。"
    exit 0
  fi
fi

if [[ ! -d "$DEST_DIR" ]]; then
  if [[ $ASSUME_YES -eq 0 ]]; then
    read -rp "目标目录不存在, 是否创建: $DEST_DIR ? [y/N]: " create_confirm
    case "$create_confirm" in
      y|Y)
        mkdir -p "$DEST_DIR"
        ;;
      *)
        echo "未创建目标目录, 跳过移动 zip。"
        echo "✅ 全流程完成 (zip 未移动)。"
        exit 0
        ;;
    esac
  else
    echo "目标目录不存在: $DEST_DIR, 且使用了 -y, 无法自动创建。跳过移动 zip。" >&2
    echo "✅ 全流程完成 (zip 未移动)。"
    exit 0
  fi
fi

DEST_PATH="$DEST_DIR/$(basename "$ZIP_PATH")"

echo "移动 zip 至: $DEST_PATH"
echo "+ mv \"$ZIP_PATH\" \"$DEST_PATH\""
mv "$ZIP_PATH" "$DEST_PATH"

echo "✅ 全流程完成, zip 已移动到: $DEST_PATH"
