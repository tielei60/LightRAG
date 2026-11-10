ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON := $(ROOT)/.venv/bin/python

.PHONY: start status stop preview import
start status stop:
	@$(PYTHON) $(ROOT)/product/manage.py $@

preview:
	@$(PYTHON) $(ROOT)/product/manage.py preview

import:
	@$(PYTHON) $(ROOT)/product/manage.py import
