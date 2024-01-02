.DEFAULT_GOAL := run
setup:
	pip install -r /data/requirements.txt

run: setup
	tritonserver --allow-http=true --model-repository=/data/repo

