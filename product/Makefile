ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON := $(ROOT)/.venv/bin/python
BATCH ?= first3

.PHONY: start status stop preview import evidence-preview evidence-import
start status stop:
	@$(PYTHON) $(ROOT)/product/manage.py --batch $(BATCH) $@

preview:
	@$(PYTHON) $(ROOT)/product/manage.py --batch $(BATCH) preview

import:
	@$(PYTHON) $(ROOT)/product/manage.py --batch $(BATCH) import

evidence-preview:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.evidence_import --batch "$(BATCH)"

evidence-import:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.evidence_import --batch "$(BATCH)" --apply
