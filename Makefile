SHELL := /usr/bin/env bash
EXEC := python=3.10
PACKAGE := lullaby
RUN := python -m
INSTALL := $(RUN) pip install
ACTIVATE := source activate $(PACKAGE)
.DEFAULT_GOAL := help

## help      : print available commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## env       : setup environment and install dependencies (call in base).
.PHONY : env
env : $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py requirements.txt requirements-conda.sh
ifeq (0, $(shell conda env list | grep -wc $(PACKAGE)))
	@conda create -yn $(PACKAGE) $(EXEC)
endif
	@$(ACTIVATE); $(INSTALL) -e "."; bash requirements-conda.sh

## format    : format code with black (call in env).
.PHONY : format
format : env
	@black .

## test      : run testing pipeline (call in env).
.PHONY : test
test: style static

style : black

static : mypy pylint 

mypy : env
	@mypy \
	-p $(PACKAGE) \
	--ignore-missing-imports \
	--check-untyped-defs # or --disallow-untyped-defs

pylint : env
	@pylint $(PACKAGE) \
	--disable C0112,C0113,C0114,C0115,C0116 \
	|| pylint-exit $$?

black : env
	@black --check .

## run	   : run package with default arguments (call in env).
.PHONY : run
run : env
	@$(RUN) $(PACKAGE)

## uninstall : remove environment (call in base).
.PHONY : uninstall
uninstall :
	@conda env remove -yn $(PACKAGE); touch requirements.txt; touch requirements-conda.sh