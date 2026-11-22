#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "ref replication started $(TZ=Europe/Paris date)" | tee -a refrep_run.log
python -u run_meta.py --prompts p_many15.txt --output ref_many.jsonl --repeats 2 --max-tokens 16 --parallel 4 2>&1 | tee -a refrep_run.log
echo "ref replication finished $(TZ=Europe/Paris date)" | tee -a refrep_run.log
