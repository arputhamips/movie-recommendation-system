# Makefile for Movie Recommendation System

.PHONY: help setup install clean test train demo lint format

help:
	@echo "Available commands:"
	@echo "  make setup      - Set up the development environment"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Clean up generated files"
	@echo "  make test       - Run unit tests"
	@echo "  make train      - Train all models"
	@echo "  make demo       - Run the demo script"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code with black"

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  On Windows: venv\\Scripts\\activate"
	@echo "  On Unix/Mac: source venv/bin/activate"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist
	rm -rf htmlcov .coverage
	rm -rf .pytest_cache

test:
	python -m pytest tests/ -v --cov=src --cov-report=html

train:
	# Train collaborative filtering
	python src/train.py --model collaborative --method user_based
	python src/train.py --model collaborative --method item_based
	
	# Train matrix factorization
	python src/train.py --model matrix_factorization --method svd
	python src/train.py --model matrix_factorization --method nmf
	
	# Train content-based
	python src/train.py --model content_based
	
	# Train hybrid
	python src/train.py --model hybrid
	
	# Train ensemble
	python src/train.py --model ensemble

demo:
	python demo.py

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

format:
	black src/ tests/ --line-length=100

download-data:
	@echo "Downloading MovieLens 100K dataset..."
	mkdir -p data/raw
	cd data/raw && \
	wget http://files.grouplens.org/datasets/movielens/ml-100k.zip && \
	unzip ml-100k.zip && \
	rm ml-100k.zip
	@echo "Data downloaded successfully!"

notebook:
	jupyter notebook notebooks/

evaluate:
	python src/train.py --cross_validate --model collaborative
	python src/train.py --cross_validate --model matrix_factorization
	python src/train.py --cross_validate --model hybrid

docker-build:
	docker build -t movie-recommender .

docker-run:
	docker run -p 5000:5000 movie-recommender

all: setup install download-data train test
