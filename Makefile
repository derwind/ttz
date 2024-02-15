.PHONY: default lint pylint mypy black pytest clean

default: lint

lint:
	ruff ttz

pylint:
	pylint -rn ttz

mypy:
	mypy ttz

black:
	black ttz setup.py

pytest:
	pytest test
