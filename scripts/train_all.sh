#!/bin/bash
# Train every experiment with 3 seeds (seeds run in parallel, 3 threads each). Fusion models warm-start from branch checkpoints.
# Finished runs are skipped, so the script can be re-run after an interruption.
export PYTHONUTF8=1
python -m scripts.make_configs
run() { for s in 0 1 2; do [ -f results/runs/$1/s$s/summary.json ] || python -m src.train.train --config configs/exp_$1.yaml --seed $s --threads 3 > logs/train_$1_s$s.log 2>&1 & done; wait; }
for m in T-clean T-asr T-asr-aug A A-aug; do run $m; done
for m in F F-aug; do run $m; done
python -m src.eval.run_eval
for m in F-aug-win10 F-aug-win60 F-aug-tok128; do run $m; done
python -m src.eval.run_eval
