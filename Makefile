install:
	pipenv install --dev

download:
	kaggle competitions download -c costa-rican-household-poverty-prediction -p data
	unzip data/costa-rican-household-poverty-prediction.zip -d data/raw/
	rm data/costa-rican-household-poverty-prediction.zip

setup: install download

lab:
	pipenv run jupyter lab --notebook-dir notebooks
