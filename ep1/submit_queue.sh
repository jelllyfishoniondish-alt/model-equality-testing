#!/bin/bash
# Submit queue/v1..v5.csv one by one, 31 minutes apart; first not before 23:10 Paris time.
cd /p/scratch/training2666/chander2/argos/model_equality/ep1
source ../.venv/bin/activate
LOG=queue/scores.log
START=$(TZ=Europe/Paris date -d "today 23:10" +%s)
NOW=$(date +%s); [ $NOW -lt $START ] && { echo "$(TZ=Europe/Paris date) waiting until 23:10 Paris" >> $LOG; sleep $((START-NOW)); }
for f in queue/v1.csv queue/v2.csv queue/v3.csv queue/v4.csv queue/v5.csv; do
  cp "$f" queue/current.csv
  { echo "===== $(TZ=Europe/Paris date "+%F %T %Z")  $f"; python ../submission.py 2>&1; echo; } >> $LOG
  [ "$f" != queue/v5.csv ] && sleep 1860
done
echo "===== $(TZ=Europe/Paris date "+%F %T %Z") queue finished" >> $LOG
