.PHONY: install test lint demo serve

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

demo:
	python -m kevin "how do we make onboarding feel less like paperwork?" \
		--constraint "no extra headcount" --known "checklists" --known "welcome email" --trace

serve:
	KEVIN_RELOAD=1 python -m uvicorn kevin.api:app --reload --port 8000
