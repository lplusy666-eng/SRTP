PYTHON ?= python
CONFIG ?= configs/demo.yaml

.PHONY: install install-full demo verify data features train scan api dashboard test doctor export-demo-inputs clean-test

install:
	$(PYTHON) -m pip install -e .

install-full:
	$(PYTHON) -m pip install -e ".[all,dev]"

doctor:
	power-econ doctor -c $(CONFIG)

data:
	power-econ collect -c $(CONFIG) --refresh

features:
	power-econ features -c $(CONFIG)

train:
	power-econ train -c $(CONFIG)

scan:
	power-econ scan -c $(CONFIG) --limit 20

demo:
	power-econ demo -c $(CONFIG)

verify:
	$(PYTHON) scripts/verify_all.py --config $(CONFIG)

api:
	power-econ serve -c $(CONFIG) --host 0.0.0.0 --port 8000

dashboard:
	power-econ dashboard -c $(CONFIG)

test:
	pytest
	$(PYTHON) -m compileall -q src tests

export-demo-inputs:
	$(PYTHON) scripts/export_demo_inputs.py --config configs/demo.yaml --output-dir data/input/demo_split

clean-test:
	rm -rf .test_artifacts .pytest_cache
