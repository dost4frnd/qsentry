# QSentry — convenience targets. Run from the repo root.
PY ?= python
CFG ?= configs/experiment/full.yaml

.PHONY: help install smoke all data train eval figures tables test clean zip

help:
	@echo "make install   install dependencies"
	@echo "make smoke     fast end-to-end pipeline (tiny)"
	@echo "make all       full pipeline (CFG=configs/experiment/full.yaml)"
	@echo "make test      run unit tests"
	@echo "make clean     remove generated data/results/caches"
	@echo "make zip       build dist/qsentry-bundle.zip"

install:
	$(PY) -m pip install -r requirements.txt

smoke:
	bash scripts/run_smoke.sh

all:
	bash scripts/run_all.sh $(CFG)

data:
	$(PY) scripts/00_generate_data.py $(CFG)

train: data
	$(PY) scripts/01_preprocess.py $(CFG)
	$(PY) scripts/02_train.py $(CFG)

figures:
	$(PY) scripts/11_make_figures.py $(CFG)

tables:
	$(PY) scripts/12_make_tables.py $(CFG)

test:
	$(PY) -m pytest -q

clean:
	rm -rf data data_smoke results results_smoke dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

zip: clean
	mkdir -p dist
	zip -r dist/qsentry-bundle.zip . \
	  -x '*/__pycache__/*' '*.pyc' './.venv/*' './dist/*' './.git/*'
