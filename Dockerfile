FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./setup.py /code/setup.py
COPY ./src/app /code/src/app
COPY ./src/text_generation /code/src/text_generation
COPY ./src/pairing/search /code/src/pairing/search
COPY ./src/pairing/model /code/src/pairing/model
COPY ./src/pairing/training /code/src/pairing/training
COPY ./src/pairing/utils /code/src/pairing/utils
COPY ./src/pairing/dataset/pairing/english.csv /code/src/pairing/dataset/pairing/english.csv
RUN python /code/setup.py install

COPY ./src/app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
