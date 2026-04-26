
# Kudos: Adapted from Auto-documenting default target
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

.DEFAULT_GOAL := help

PROJECT = quantumflow
FILES = $(PROJECT) docs/conf.py examples/

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

init:  ## Install and initlize package ready for development
	uv sync --all-extras

about:	## Report versions of dependent packages
	@uv run python -m $(PROJECT).about

status:  ## git status
	@echo
	@git status --short --branch

test:		## Run unittests
	uv run pytest --disable-pytest-warnings

coverage:	## Report test coverage using current backend
	@echo
	uv run pytest --disable-pytest-warnings --cov --cov-report=term-missing
	@echo

lint:		## Lint check python source
	uv run ruff check

delint:   ## Run isort and black to delint project
	uv run ruff format

typecheck:	## Static typechecking
	uv run mypy $(PROJECT)

docs:		## Build documentation
	uv run make -C docs html

docs-open:  ## Build documentation and open in webbrowser
	uv run make -C docs html
	open docs/_build/html/index.html

docs-clean: 	## Clean documentation build
	uv run make -C docs clean

testall: about coverage lint typecheck docs build   ## Run all tests

pragmas:	## Report all pragmas in code
	@echo
	@echo "** Code that needs something done **"
	@grep 'TODO' --color -r -n $(FILES) || echo "No TODO pragmas"
	@echo
	@echo "** Code that needs fixing **"
	@grep 'FIXME' --color -r -n $(FILES) || echo "No FIXME pragmas"
	@echo
	@echo "** Code that needs documenting **"
	@grep 'DOCME' --color -r -n $(FILES) || echo "No DOCME pragmas"
	@echo
	@echo "** Code that needs more tests **"
	@grep 'TESTME' --color -r -n $(FILES) || echo "No TESTME pragmas"
	@echo
	@echo "** Implementation notes **"
	@grep 'NB:' --color -r -n $(FILES)  || echo "No NB implementation notes Pragmas"
	@echo	
	@echo "** Acknowledgments **"
	@grep 'kudos:' --color -r -n -i $(FILES) || echo "No kudos"
	@echo
	@echo "** Pragma for test coverage **"
	@grep 'pragma: no cover' --color -r -n $(FILES) || echo "No Typecheck Pragmas"
	@echo
	@echo "** flake8 linting pragmas **"
	@echo "(http://flake8.pycqa.org/en/latest/user/error-codes.html)"
	@grep '# noqa:' --color -r -n $(FILES) || echo "No flake8 pragmas"
	@echo
	@echo "** Typecheck pragmas **"
	@grep '# type:' --color -r -n $(FILES) || echo "No Typecheck Pragmas"

build: clean ## Build sdist and wheel
	uv build

requirements: ## Make requirements.txt
	uv export --no-hashes --no-emit-project -o requirements.txt

clean: ## Remove build artifacts
	git clean -fdX -- build dist quantumflow.egg-info


.PHONY: help
.PHONY: docs
.PHONY: build
