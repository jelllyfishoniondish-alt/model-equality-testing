#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp3
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "started $(TZ=Europe/Paris date)" | tee -a exp3_run.log
python -u run_exp3.py --repeats 30 2>&1 | tee -a exp3_run.log
echo "finished $(TZ=Europe/Paris date)" | tee -a exp3_run.log
