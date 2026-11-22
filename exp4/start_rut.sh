#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
export HF_HOME=/p/scratch/training2666/chander2/argos/.cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
V=/p/scratch/training2666/chander2/argos/unlearning/t3011/.venv/bin/python
echo "rut select started $(TZ=Europe/Paris date)" | tee -a rut_run.log
$V -u rut_score.py --select --max-answers 20 --models HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-360M-Instruct 2>&1 | grep -v Warning | tee -a rut_run.log
echo "rut select finished $(TZ=Europe/Paris date)" | tee -a rut_run.log
