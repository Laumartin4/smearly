FROM python:3.10.6-bullseye

COPY smearly /smearly
COPY requirements.txt /requirements.txt

RUN mkdir /models
COPY models/model-064.h5 /models/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install tensorflow-cpu

CMD uvicorn smearly.api.fast:app --host 0.0.0.0 --port $PORT
