#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "ref violate started $(TZ=Europe/Paris date)" | tee -a violate_run.log
python -u run_meta.py --prompts p_violate.txt --output ref_violate.jsonl --repeats 5 --max-tokens 200 --parallel 4 2>&1 | tee -a violate_run.log
echo "ref violate finished $(TZ=Europe/Paris date)" | tee -a violate_run.log
