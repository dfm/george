FROM python:3.7

COPY . ./george
RUN pip3 install --upgrade pip
RUN cd ./george && pip3 install .

RUN rm -rf ./george/george

RUN pip3 install pytest

RUN pytest -m ./george/tests