PYTHON ?= python3
.DEFAULT_GOAL := start
VENV := .venv/bin/python
LEARN := $(VENV) -m learning.train
.PHONY: start setup build up stop logs smoke demo train evaluate

start: setup up
	@echo '周回学習: http://127.0.0.1:8766/'

setup: .venv/.installed

.venv/.installed: learning/requirements.txt
	$(PYTHON) -m venv .venv
	$(VENV) -m pip install -r learning/requirements.txt
	touch $@

build:
	./scripts/build.sh

up:
	@./scripts/build.sh --ensure
	@mkdir -p runs
	docker compose up --build -d --wait --wait-timeout 180

stop:
	docker compose down

logs:
	docker compose logs -f sim

smoke: setup
	$(VENV) scripts/smoke.py

demo: setup
	$(LEARN) baseline --frame runs/gate.png

train: setup
	$(LEARN) train --output runs/ppo.zip

evaluate: setup
	$(LEARN) evaluate --model runs/ppo.zip --frame runs/evaluation.png
