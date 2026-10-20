PYTHON ?= .venv/bin/python
VIDEO ?= data/sample_tennis.mp4

.PHONY: setup demo extract train train-ball report summary test lint

setup:
	python -m venv .venv
	.venv/bin/python -m pip install -r requirements-dev.txt

demo:
	$(PYTHON) -m court_detector.movement_analyzer all --input "$(VIDEO)" \
		--output-video outputs/demo_annotated.mp4

extract:
	$(PYTHON) -m court_detector.movement_analyzer extract --input "$(VIDEO)" --no-ball

train:
	$(PYTHON) -m court_detector.movement_analyzer train

train-ball:
	$(PYTHON) training/train_ball.py --epochs 10 --batch 1

report:
	$(PYTHON) -m court_detector.movement_analyzer report

summary:
	$(PYTHON) -m court_detector.movement_analyzer summary

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check court_detector training tests
