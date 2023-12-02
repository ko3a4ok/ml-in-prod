build:
	docker build -f Dockerfile -t l6:latest .

train: build
	docker run -it l6:latest python l6/trainer.py

test:
	pytest --disable-warnings ./tests/

test_data:
	pytest --disable-warnings -rc ./tests/test_data.py

test_code:
	pytest --disable-warnings -rc ./tests/test_code.py

test_model:
	pytest --disable-warnings -ss ./tests/test_model.py

