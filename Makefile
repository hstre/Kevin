.PHONY: install test lint demo

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

demo:
	python -m kevin "how do we make onboarding feel less like paperwork?" \
		--constraint "no extra headcount" --known "checklists" --known "welcome email" --trace
