.PHONY: test test-fast test-coverage

test:
	PYTHONPATH=src python -m pytest

test-fast:
	PYTHONPATH=src python -m pytest -m "not slow"

test-coverage:
	PYTHONPATH=src python -m pytest --cov=actinet --cov-report=term-missing --cov-fail-under=80
