#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "exp5 started $(TZ=Europe/Paris date)" | tee -a exp5_run.log
python -u run_meta.py --prompts p_translate.txt --output ans_translate.jsonl --repeats 10 --max-tokens 200 --parallel 4 2>&1 | tee -a exp5_run.log
python -u run_meta.py --prompts p_flips.txt --output ans_flips.jsonl --repeats 20 --max-tokens 256 --parallel 4 2>&1 | tee -a exp5_run.log
python -u run_meta.py --prompts p_content.txt --output ans_content.jsonl --repeats 4 --max-tokens 200 --parallel 4 2>&1 | tee -a exp5_run.log
echo "exp5 finished $(TZ=Europe/Paris date)" | tee -a exp5_run.log
