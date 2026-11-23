# One command per stage. Run from the project root.
PY=python
build-data:      ## sample, synthesize, degrade, ASR, manifest, features, stats
	bash scripts/build_all.sh
train:           ## all experiments, 3 seeds each (branch models first, then fusion warm-started from them)
	bash scripts/train_all.sh
eval:            ## metric tables, figures
	$(PY) -m src.eval.run_eval && $(PY) -m src.eval.figures
demo:            ## web app: landing page + live demo + results at http://localhost:8000
	python -m uvicorn src.api.server:app --port 8000
demo-streamlit:  ## older single-page Streamlit demo
	streamlit run src/demo/app.py
test:
	$(PY) -m pytest tests -q
