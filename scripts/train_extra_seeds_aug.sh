#!/bin/bash
# Seeds 3-4 for the augmented models (H3 five-seed check). Waits for the 512-token runs, then trains branches, then F-aug.
export PYTHONUTF8=1
until [ -f results/runs/T-asr-tok512/s0/summary.json ] && [ -f results/runs/T-asr-tok512/s1/summary.json ] && [ -f results/runs/T-asr-tok512/s2/summary.json ]; do sleep 30; done
run() { for s in "${@:2}"; do [ -f results/runs/$1/s$s/summary.json ] || python -m src.train.train --config configs/exp_$1.yaml --seed $s --threads 3 > logs/train_$1_s$s.log 2>&1 & done; }
run T-asr-aug 3 4; run A-aug 3 4; wait
echo branches done
run F-aug 3 4; wait
echo aug seeds done
