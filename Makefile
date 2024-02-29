
# Kudos: Adapted from Auto-documenting default target
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

.DEFAULT_GOAL := help

PROJECT = quantumflow
FILES = $(PROJECT) docs/conf.py setup.py examples/

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

init:  ## Install and initlize package ready for development
	pip install -e '.[dev]'

about:	## Report versions of dependent packages
	@python -m $(PROJECT).about

status:  ## git status
	@echo
	@git status --short --branch

test:		## Run unittests
	pytest --disable-pytest-warnings

coverage:	## Report test coverage using current backend
	@echo
	pytest --disable-pytest-warnings --cov
	@echo

lint:		## Lint check python source
	ruff check

delint:   ## Run isort and black to delint project
	ruff format

typecheck:	## Static typechecking 
	mypy $(PROJECT)

docs:		## Build documentation
	(cd docs; make html)

docs-open:  ## Build documentation and open in webbrowser
	(cd docs; make html)
	open docs/_build/html/index.html

docs-clean: 	## Clean documentation build
	(cd docs; make clean)

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

build: ## Setuptools build
	./setup.py clean --all
	./setup.py sdist bdist_wheel

requirements: ## Make requirements.txt
	pip freeze > requirements.txt

clean: ## Clean up after setuptools
	./setup.py clean --all


.PHONY: help
.PHONY: docs
.PHONY: build
