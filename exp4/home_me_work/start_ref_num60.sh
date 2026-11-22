#!/bin/bash
# runs from $HOME because /p/scratch/training2666 is out of inodes (project quota)
cd ~/me_work
source /p/scratch/training2666/chander2/argos/model_equality/.venv/bin/activate
: "${ME_API_KEY:?export ME_API_KEY=\"your team key\" first}"
EPS=$(python -c "print(\",\".join(f\"ep-{i:02d}\" for i in range(1,34)))")
echo "ref num60 started $(TZ=Europe/Paris date)" | tee -a ref_num60_run.log
python -u run_meta.py --prompts p_num.txt --output ref_num60.jsonl --repeats 60 --max-tokens 16 --parallel 4 2>&1 | tee -a ref_num60_run.log
echo "ref num60 finished $(TZ=Europe/Paris date)" | tee -a ref_num60_run.log
