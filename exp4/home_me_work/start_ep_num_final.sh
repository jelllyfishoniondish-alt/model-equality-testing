#!/bin/bash
cd ~/me_work
source /p/scratch/training2666/chander2/argos/model_equality/.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
EPS=$(python -c "print(\",\".join(f\"ep-{i:02d}\" for i in range(1,34)))")
echo "ep num final started $(TZ=Europe/Paris date)" | tee -a ep_num_final_run.log
python -u run_meta.py --prompts p_num.txt --output ep_num_final.jsonl --endpoints $EPS --repeats 18 --max-tokens 16 --parallel 4 --allow-ep 2>&1 | tee -a ep_num_final_run.log
echo "ep num final finished $(TZ=Europe/Paris date)" | tee -a ep_num_final_run.log
