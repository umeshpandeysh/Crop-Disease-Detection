# Crop-Disease-Detection — Project Makefile
.PHONY: help install test lint train-cnn train-efficientnet clean

PYTHON ?= python

help:              ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:           ## Install all dependencies
	pip install -r requirements.txt

test:              ## Run the test suite
	pytest tests/ -v --tb=short

lint:              ## Lint source code with flake8
	flake8 src/ training/ tests/ --max-line-length=100 --ignore=E203,W503

train-cnn:         ## Train custom 4-block CNN (requires dataset)
	$(PYTHON) training/train.py --model cnn --epochs 30

train-efficientnet: ## Train EfficientNet-B0 (requires dataset)
	$(PYTHON) training/train.py --model efficientnet --epochs 30

clean:             ## Remove generated files
	rm -rf models/*.pt logs/ outputs/
	@echo "Cleaned."
