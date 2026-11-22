#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "ref four-type started $(TZ=Europe/Paris date)" | tee -a four_run.log
python -u run_meta.py --prompts p_four.txt --output ref_four.jsonl --repeats 2 --max-tokens 16 --parallel 4 2>&1 | tee -a four_run.log
echo "ref four-type finished $(TZ=Europe/Paris date)" | tee -a four_run.log
