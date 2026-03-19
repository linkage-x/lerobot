#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/home/hph/Code/lerobot-replay}"
COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.yml}"
CONFIG_PATH="${CONFIG_PATH:-src/lerobot/configs/franka_research3_ee2ee_act_das.yaml}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/lerobotv3_0310_100ep}"
WORKERS_LIST="${WORKERS_LIST:-4 8}"
STEPS="${STEPS:-400}"
LOG_FREQ="${LOG_FREQ:-200}"
HOME_OVERRIDE="${HOME_OVERRIDE:-/home/hph}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_DIR="${BENCH_DIR:-outputs/benchmarks/train_workers/${TIMESTAMP}}"

mkdir -p "${WORKDIR}/${BENCH_DIR}"

printf 'workers,elapsed_s,steps_per_s,data_s,updt_s,log_path\n' > "${WORKDIR}/${BENCH_DIR}/summary.csv"

for workers in ${WORKERS_LIST}; do
  run_name="workers_${workers}"
  log_path="${WORKDIR}/${BENCH_DIR}/${run_name}.log"
  output_dir="${BENCH_DIR}/${run_name}_output"

  start_ts="$(date +%s)"
  sudo env HOME="${HOME_OVERRIDE}" docker compose --profile train -f "${COMPOSE_FILE}" run -T --rm \
    lerobot-internal \
    lerobot-train \
    --config_path="${CONFIG_PATH}" \
    --dataset.root="${DATASET_ROOT}" \
    --num_workers="${workers}" \
    --steps="${STEPS}" \
    --log_freq="${LOG_FREQ}" \
    --eval_freq=0 \
    --save_checkpoint=false \
    --output_dir="${output_dir}" 2>&1 | tee "${log_path}"
  end_ts="$(date +%s)"

  elapsed_s="$((end_ts - start_ts))"
  steps_per_s="$(awk -v steps="${STEPS}" -v elapsed="${elapsed_s}" 'BEGIN { if (elapsed > 0) printf "%.3f", steps / elapsed; else print "0.000" }')"
  metrics_line="$(grep 'step:' "${log_path}" | tail -n 1 || true)"
  data_s="$(printf '%s\n' "${metrics_line}" | sed -n 's/.* data_s:\([0-9.]*\).*/\1/p')"
  updt_s="$(printf '%s\n' "${metrics_line}" | sed -n 's/.* updt_s:\([0-9.]*\).*/\1/p')"

  printf '%s,%s,%s,%s,%s,%s\n' \
    "${workers}" \
    "${elapsed_s}" \
    "${steps_per_s}" \
    "${data_s:-NA}" \
    "${updt_s:-NA}" \
    "${log_path}" >> "${WORKDIR}/${BENCH_DIR}/summary.csv"
done

cat "${WORKDIR}/${BENCH_DIR}/summary.csv"
