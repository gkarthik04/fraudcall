#!/bin/bash
# Full data build (resumable: finished calls/conditions are cached). One command: make build-data
export PYTHONUTF8=1
python -m src.data.icfd
python -m src.data.build synth --set 1 --subset all --workers 5
python -m src.data.build degrade --set 1 --subset all
python -m src.data.build synth --set 2 --subset all --workers 2
python -m src.data.build degrade --set 2 --subset all --workers 5
python -m src.data.build manifest
python -m src.features.cache
python -m src.data.stats --tag full
