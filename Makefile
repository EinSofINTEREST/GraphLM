.DEFAULT_GOAL := help

PYTHON ?= python3
PIP ?= pip

.PHONY: help install install-notebook test test-verbose coverage coverage-html lint fmt nb-clean clean

help: ## 사용 가능한 타겟 표시
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## 개발 의존성 포함 editable 설치 (pip install -e ".[dev]")
	$(PIP) install -e ".[dev]"

install-notebook: ## 노트북 의존성까지 포함 설치
	$(PIP) install -e ".[dev,notebook]"

test: ## pytest 실행 (slow/gpu/network 마커 제외)
	pytest -m "not slow and not gpu and not network"

test-verbose: ## pytest 상세 출력
	pytest -m "not slow and not gpu and not network" -vv

coverage: ## 커버리지 측정 (terminal + missing 라인)
	pytest --cov=graphlm --cov-report=term-missing --cov-fail-under=70

coverage-html: ## 커버리지 HTML 리포트 생성 (htmlcov/)
	pytest --cov=graphlm --cov-report=html --cov-report=term

lint: ## ruff check + nbqa
	bash scripts/lint.sh

fmt: ## ruff format 적용
	ruff format .

nb-clean: ## 노트북 output cell 정리 (nbstripout)
	@if command -v nbstripout >/dev/null 2>&1; then \
		find notebooks -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 \
			| xargs -0 -r nbstripout; \
		echo "노트북 출력 셀 정리 완료"; \
	else \
		echo "nbstripout 미설치. pip install nbstripout 후 재시도"; \
		exit 1; \
	fi

clean: ## 캐시 / 빌드 산출물 제거
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
