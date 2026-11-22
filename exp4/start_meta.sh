#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
source ../.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
echo "meta started $(TZ=Europe/Paris date)" | tee -a meta_run.log
python -u run_meta.py --repeats 5 --logprobs --parallel 1 2>&1 | tee -a meta_run.log
echo "meta finished $(TZ=Europe/Paris date)" | tee -a meta_run.log
