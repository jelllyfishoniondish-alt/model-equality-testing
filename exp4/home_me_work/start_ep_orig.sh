#!/bin/bash
# runs from $HOME because /p/scratch/training2666 is out of inodes (project quota)
cd ~/me_work
source /p/scratch/training2666/chander2/argos/model_equality/.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
EPS=$(python -c "print(\",\".join(f\"ep-{i:02d}\" for i in range(1,34)))")
echo "ep orig started $(TZ=Europe/Paris date)" | tee -a ep_orig_run.log
python -u run_meta.py --prompts p_ep_orig.txt --output ep_orig.jsonl --endpoints $EPS --repeats 1 --max-tokens 256 --parallel 4 --allow-ep 2>&1 | tee -a ep_orig_run.log
echo "ep orig finished $(TZ=Europe/Paris date)" | tee -a ep_orig_run.log
