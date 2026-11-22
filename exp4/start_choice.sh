#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "ref choice started $(TZ=Europe/Paris date)" | tee -a choice_run.log
python -u run_meta.py --prompts p_choice.txt --output ref_choice.jsonl --repeats 3 --max-tokens 16 --parallel 4 2>&1 | tee -a choice_run.log
echo "ref choice finished $(TZ=Europe/Paris date)" | tee -a choice_run.log
