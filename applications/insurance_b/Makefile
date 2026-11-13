.PHONY: status storage-start storage-stop prepare ingest verify serve smoke test check

PYTHON ?= .venv/bin/python
B := $(PYTHON) -m applications.insurance_b.cli

status storage-start storage-stop ingest verify serve smoke:
	$(B) $@

prepare:
	$(B) prepare --parse

test:
	PYTHON=$(PYTHON) ./scripts/test.sh applications/insurance_b -q

check:
	.venv/bin/ruff check applications/insurance_b
	.venv/bin/ruff format --check applications/insurance_b
