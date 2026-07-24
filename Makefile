help: 
	@echo ¬ make run
	@echo ¬ make install
	@echo ¬ make debug
	@echo ¬ make clean
	@echo ¬ make lint

run:
	@uv run python -m src

install:
	uv pip install torch transformers mypy flake8 pydantic

debug:
	uv run python -m pdb src/main.py

clean:
	@rm -rf **/__pycache__ .mypy_cache

lint:
	
	@mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs
	@flake8 .

.PHONY: run install debug clean lint