.PHONY: help install test test-cov run retrain docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install       Install production and development dependencies"
	@echo "  make test          Run pytest test suite"
	@echo "  make test-cov      Run pytest with coverage report and 90% quality gate"
	@echo "  make run           Start local FastAPI development server"
	@echo "  make retrain       Execute clean MLOps training pipeline"
	@echo "  make docker-build  Build multi-stage Docker image"
	@echo "  make docker-up     Start container with docker compose"
	@echo "  make docker-down   Stop docker compose containers"
	@echo "  make clean         Remove Python cache and temporary test artifacts"

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=predict --cov-report=term-missing --cov-fail-under=90

run:
	uvicorn predict.api.main:app --host 127.0.0.1 --port 8000 --reload

retrain:
	python ml_pipeline/train_model.py

docker-build:
	docker build -t maritime-predict-api:latest .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov
