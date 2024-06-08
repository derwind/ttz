.PHONY: default lint pylint mypy black pytest clean

default: lint

lint:
	ruff check ttz

pylint:
	pylint -rn ttz

mypy:
	mypy ttz

black:
	black ttz setup.py

pytest:
	pytest test
