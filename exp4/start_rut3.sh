#!/bin/bash
cd /p/scratch/training2666/chander2/argos/model_equality/exp4
export HF_HOME=/p/scratch/training2666/chander2/argos/.cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=32
V=/p/scratch/training2666/chander2/argos/unlearning/t3011/.venv/bin/python
echo "rut 1.7B started $(TZ=Europe/Paris date)" | tee -a rut_run.log
$V -u rut_score.py --select --max-answers 20 --models HuggingFaceTB/SmolLM2-1.7B HuggingFaceTB/SmolLM2-1.7B-Instruct 2>&1 | grep "###" | tee -a rut_run.log
$V -u rut_score.py --max-answers 60 --models HuggingFaceTB/SmolLM2-1.7B-Instruct 2>&1 | grep -v -e Warning -e "Loading weights" -e torch_dtype | tee -a rut_run.log
echo "rut 1.7B finished $(TZ=Europe/Paris date)" | tee -a rut_run.log
