.PHONY: merge test

# merge player data with raw data
merge:
	python3 playerData.py

# run current test in tests.py
test:
	python3 tests.py