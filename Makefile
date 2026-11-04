.PHONY: studio infer test audit figures clean

PYTHON ?= python3
PYTEST ?= pytest

studio:
	$(PYTHON) release_30b/studio_30b.py --port 8789

infer:
	$(PYTHON) release_30b/inference_loader_30b.py --interactive

test:
	$(PYTEST) tests/ campaigns/rigorous_benchmarks/benchmarks/test_benchmarks.py -v

audit:
	$(PYTHON) -m src.lean.audit.runner --milestone M20

figures:
	$(PYTHON) scripts/generate_30b_publication_figures.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
