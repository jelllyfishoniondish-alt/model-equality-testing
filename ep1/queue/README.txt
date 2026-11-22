Submission queue (ep1, 40 answers per probe, no new queries). Each differs from v1 in one thing.
v1  l2u, family 0.10, model 0.10   base: X / Y / Z separate families, outliers 02+23, 04, 06, 10, 20 alone (= submission_family.csv)
v2  l2u, family 0.10, model 0.05   same families, but Z split into models {05,16} and {09,11,13,15,19,27,29,31}
v3  families = v1; split models by within-family mmd_perm z (both probes): Y: 28 alone; Z: {05,16}, {13}, rest  (strongest evidence only)
v4  hypothesis A (ref prior: ~half of each family = disguised copies of one model, rest = distinct variants): per family the most central half by within-family mmd_perm z = one model, the rest each its own model
v5  hypothesis B (the opposite: endpoints that look most different are disguises): per family the 3 most distant + the most central = one model, the rest singletons; X (no outliers) all singletons
(v1-v3 all scored 0.1; previous v4/v5 moved to old/)
(old v3-v5 = tv / mmd / rank family variants moved to old/; v1 = 0.1 showed families are already right)
Commands: python cluster_ep.py --metric <m> --family-threshold <f> --model-threshold <m> --out queue/vN.csv
Scores: queue/scores.log (one block per submission). submit_queue.sh copies vN.csv to queue/current.csv and runs ../submission.py.
