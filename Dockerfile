FROM tensorflow/tensorflow:latest

COPY requirements.txt /requirements.txt

RUN pip3 install -r  requirements.txt

WORKDIR /

COPY . /

ENTRYPOINT ["python3", "train.py"]