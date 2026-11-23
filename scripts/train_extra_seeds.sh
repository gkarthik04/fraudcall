#!/bin/bash
# Extra seeds (3, 4) for the models behind H2 (T-asr, A, F). Stage A trains the branches, stage B warm-starts F from them.
# Finished runs are skipped. Primary tables still use seeds 0-2; seeds 3-4 feed results/paired_tests.
export PYTHONUTF8=1
run() { for s in "${@:2}"; do [ -f results/runs/$1/s$s/summary.json ] || python -m src.train.train --config configs/exp_$1.yaml --seed $s --threads 3 > logs/train_$1_s$s.log 2>&1 & done; }
run T-asr 3 4; run A 3 4; wait
run F 3 4; wait
echo extra seeds done
