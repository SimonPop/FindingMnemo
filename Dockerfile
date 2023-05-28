FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./setup.py /code/setup.py
COPY ./finding_mnemo/app /code/finding_mnemo/app
COPY ./finding_mnemo/text_generation /code/finding_mnemo/text_generation
COPY ./finding_mnemo/pairing/search /code/finding_mnemo/pairing/search
COPY ./finding_mnemo/pairing/model /code/finding_mnemo/pairing/model
COPY ./finding_mnemo/pairing/training /code/finding_mnemo/pairing/training
COPY ./finding_mnemo/pairing/utils /code/finding_mnemo/pairing/utils
COPY ./finding_mnemo/pairing/dataset/pairing/english.csv /code/finding_mnemo/pairing/dataset/pairing/english.csv
RUN python /code/setup.py install

COPY ./finding_mnemo/app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
