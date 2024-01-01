FROM huggingface/transformers-pytorch-cpu:4.18.0
RUN pip install --upgrade pip

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "bash" ]
