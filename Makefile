.DEFAULT_GOAL := help

PYTHON ?= python3
PIP ?= pip
UV ?= uv
VENV_DIR ?= .venv
KERNEL_NAME ?= graphlm
KERNEL_DISPLAY ?= GraphLM (uv .venv)

.PHONY: help venv install install-notebook kernel kernel-uninstall test test-verbose coverage coverage-html lint fmt nb-clean clean

help: ## 사용 가능한 타겟 표시
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## uv 로 Python 3.12 가상환경 생성 (.venv)
	$(UV) venv --python 3.12 $(VENV_DIR)
	@echo ""
	@echo "활성화:  source $(VENV_DIR)/bin/activate"
	@echo "다음:    make install-notebook && make kernel"

install: venv ## .venv 에 dev 의존성 + 패키지 editable 설치
	$(UV) pip install -e ".[dev]"

install-notebook: venv ## .venv 에 dev + notebook 의존성 + 패키지 editable 설치
	$(UV) pip install -e ".[dev,notebook]"

kernel: ## .venv 를 Jupyter 커널로 등록 (notebook 에서 선택 가능)
	$(VENV_DIR)/bin/python -m ipykernel install --user \
		--name=$(KERNEL_NAME) \
		--display-name="$(KERNEL_DISPLAY)"
	@echo ""
	@echo "등록된 커널 목록:"
	@$(VENV_DIR)/bin/jupyter kernelspec list

kernel-uninstall: ## 등록된 Jupyter 커널 제거
	$(VENV_DIR)/bin/jupyter kernelspec uninstall -y $(KERNEL_NAME) || true

test: ## pytest 실행 (slow/gpu/network 마커 제외)
	$(UV) run pytest -m "not slow and not gpu and not network"

test-verbose: ## pytest 상세 출력
	$(UV) run pytest -m "not slow and not gpu and not network" -vv

coverage: ## 커버리지 측정 (terminal + missing 라인)
	$(UV) run pytest --cov=graphlm --cov-report=term-missing --cov-fail-under=70

coverage-html: ## 커버리지 HTML 리포트 생성 (htmlcov/)
	$(UV) run pytest --cov=graphlm --cov-report=html --cov-report=term

lint: ## ruff check + nbqa
	$(UV) run bash scripts/lint.sh

fmt: ## ruff format 적용
	$(UV) run ruff format .

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
