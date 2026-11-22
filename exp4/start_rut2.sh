#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
export HF_HOME=/p/scratch/training2666/chander2/argos/.cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
V=/p/scratch/training2666/chander2/argos/unlearning/t3011/.venv/bin/python
echo "rut step2 started $(TZ=Europe/Paris date)" | tee -a rut_run.log
$V -u rut_score.py --max-answers 60 --models HuggingFaceTB/SmolLM2-360M-Instruct 2>&1 | grep -v -e Warning -e "Loading weights" -e torch_dtype | tee -a rut_run.log
echo "rut step2 finished $(TZ=Europe/Paris date)" | tee -a rut_run.log
