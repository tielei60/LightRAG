ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON := $(ROOT)/.venv/bin/python
BATCH ?= master
BASE_BATCH ?=
APPROVED_FROM ?=

.PHONY: start status stop preview import evidence-preview evidence-import append-preview append-import verify qa
start status stop:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.manage --batch "$(BATCH)" $@

preview:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.manage --batch "$(BATCH)" preview

import:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.manage --batch "$(BATCH)" import

evidence-preview:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.evidence_import --batch "$(BATCH)"

evidence-import:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.evidence_import --batch "$(BATCH)" --apply

append-preview:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.evidence_import --batch "$(BATCH)" --append $(if $(BASE_BATCH),--base-batch "$(BASE_BATCH)") $(if $(APPROVED_FROM),--approved-from "$(APPROVED_FROM)")

append-import:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.evidence_import --batch "$(BATCH)" --append --apply $(if $(BASE_BATCH),--base-batch "$(BASE_BATCH)") $(if $(APPROVED_FROM),--approved-from "$(APPROVED_FROM)")

verify:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.verify_batch --batch "$(BATCH)" readback

qa:
	@cd "$(ROOT)" && "$(PYTHON)" -m product.verify_batch --batch "$(BATCH)" qa --repeat 3
