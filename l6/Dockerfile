FROM tensorflow/tensorflow
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y build-essential cmake

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "bash" ]
