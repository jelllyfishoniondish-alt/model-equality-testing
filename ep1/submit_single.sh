#!/bin/bash
# Overnight single-split queue: one submission every 31 min, stop at 10:30 Paris. Start only after the user says so.
cd /p/scratch/training2666/chander2/argos/model_equality/ep1
source ../.venv/bin/activate
LOG=queue/scores.log
STOP=$(TZ=Europe/Paris date -d "today 10:30" +%s); [ $STOP -lt $(date +%s) ] && STOP=$(TZ=Europe/Paris date -d "tomorrow 10:30" +%s)
for f in queue/gt/v6.csv queue/gt/v6b.csv queue/gt/s01.csv queue/gt/s13.csv queue/gt/s05.csv queue/gt/b1.csv queue/gt/b2.csv queue/gt/b3.csv queue/gt/b4.csv queue/gt/b5.csv; do
  while :; do
    [ $(date +%s) -ge $STOP ] && { echo "===== $(TZ=Europe/Paris date "+%F %T %Z") stop time reached" >> $LOG; exit 0; }
    cp "$f" queue/current.csv
    OUT=$(python ../submission.py 2>&1)
    { echo "===== $(TZ=Europe/Paris date "+%F %T %Z")  $f (single queue)"; echo "$OUT"; echo; } >> $LOG
    if echo "$OUT" | grep -q "Score (public split)"; then break; fi
    sleep 1860  # rejected (e.g. team cooldown after the lead submitted): retry this file at the next slot
  done
  sleep 1860
done
echo "===== $(TZ=Europe/Paris date "+%F %T %Z") single queue finished" >> $LOG
