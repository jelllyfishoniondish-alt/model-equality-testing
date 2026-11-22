"""Build stage-2 prompts for the selected question indices: verbatim, 'Choice:' ending, shuffled options.
Writes p_stage2.txt and stage2_map.json (prompt line -> question index, variant, letter->option text)."""
import json, random, sys
sel = [int(x) for x in sys.argv[1].split(",")]
items = json.load(open("anchor_items.json"))
rng = random.Random(7); lines = []; mp = []
for qi in sel:
    it = items[qi]; opts = it["opts"]
    def fmt(o, end): return it["stem"] + " " + " ".join(f"{'ABCD'[k]}) {x}" for k, x in enumerate(o)) + f" Respond with only the letter. {end}"
    sh = opts[:]
    while sh == opts and len(opts) > 1: rng.shuffle(sh)
    for var, o, end in [("verbatim", opts, "Answer:"), ("choice", opts, "Choice:"), ("shuffled", sh, "Answer:")]:
        lines.append(fmt(o, end)); mp.append({"q": qi, "variant": var, "letters": {"ABCD"[k]: x for k, x in enumerate(o)}})
open("p_stage2.txt", "w").write("\n".join(lines) + "\n"); json.dump(mp, open("stage2_map.json", "w"))
print(len(lines), "prompts")
