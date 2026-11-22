#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "started $(TZ=Europe/Paris date)" | tee -a exp4_run.log
python -u run_exp4.py --repeats 12 2>&1 | tee -a exp4_run.log
echo "finished $(TZ=Europe/Paris date)" | tee -a exp4_run.log
