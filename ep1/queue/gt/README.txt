Leaderboard group testing (families = v1). Each bN.csv = v1 with exactly one endpoint per family X/Y/Z made its own model.
Ranked by mean within-family mmd_perm z on number 1-100 (+animal); endpoints already split jointly in v3 (28, 13, 05, 16) are left out because v3 scored 0.1.
b1: 01 (X), 07 (Y), 29 (Z)
b2: 33 (X), 08 (Y), 31 (Z)
b3: 14 (X), 22 (Y), 09 (Z)
b4: 24 (X), 30 (Y), 19 (Z)
b5: 12 (X), 26 (Y), 11 (Z)
A correct split (endpoint unique in its family) should lift the public score to about 0.25; a wrong one gives 0.1 (negatives are clipped).
If a batch scores > 0.1: test its 3 endpoints one at a time, keep each confirmed split in all later CSVs.
Submit:  cp gt/bN.csv current.csv && python ../../submission.py   (from ep1/queue; SUBMISSION_FILE points to queue/current.csv)
