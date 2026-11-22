#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "ref complete started $(TZ=Europe/Paris date)" | tee -a complete_run.log
python -u run_meta.py --prompts p_complete.jsonl --output ref_complete.jsonl --repeats 2 --max-tokens 16 --parallel 4 2>&1 | tee -a complete_run.log
echo "ref complete finished $(TZ=Europe/Paris date)" | tee -a complete_run.log
