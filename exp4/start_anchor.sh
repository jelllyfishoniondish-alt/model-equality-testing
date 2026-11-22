#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "ref anchor started $(TZ=Europe/Paris date)" | tee -a anchor_run.log
python -u run_meta.py --prompts p_anchor.txt --output ref_anchor.jsonl --repeats 1 --max-tokens 16 --parallel 4 2>&1 | tee -a anchor_run.log
echo "ref anchor finished $(TZ=Europe/Paris date)" | tee -a anchor_run.log
