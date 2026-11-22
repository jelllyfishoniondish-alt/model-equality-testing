#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "ref stage1 started $(TZ=Europe/Paris date)" | tee -a stage1_run.log
python -u run_meta.py --prompts p_stage1.txt --output ref_stage1.jsonl --repeats 1 --max-tokens 16 --parallel 4 2>&1 | tee -a stage1_run.log
echo "ref stage1 finished $(TZ=Europe/Paris date)" | tee -a stage1_run.log
