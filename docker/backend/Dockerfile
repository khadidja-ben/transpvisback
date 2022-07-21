# FROM ubuntu:latest
# FROM ubuntu:20.04
# RUN apt-get update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.9 python3.9-dev python3-pip
# FROM python:3

# # FROM ubuntu:20.04

# RUN apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/*
# RUN sudo apt update
# RUN sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
# RUN cd /tmp
# RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
# RUN tar -xf Python-3.8.3.tgz
# RUN cd python-3.8.3
# RUN ./configure --enable-optimizations
# RUN sudo make altinstall
# RUN sudo make install
# RUN python3 --version

FROM ubuntu:latest
# ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Africa \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt python3-dev && apt-get install build-essential



# RUN apt-get update && \
#     # apt-get upgrade && \
#     apt-get install software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install python3.9 && \
#     apt-get install -y python3-dev && \
#     apt-get install -y python3-pip

WORKDIR /app
COPY requirements.txt .
# RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.9 /usr/bin/python
# RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.9 /usr/bin/python3

# RUN pip install --upgrade pip
# RUN pip install Cython
# RUN pip3 install --upgrade pip==20.0.1
RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt
RUN pip3 install gunicorn==19.9.0

ADD ./serverTranspvisBack /app/serverTranspvisBack
ADD ./docker /app/docker
# ADD ./machineLearning /app/machineLearning

RUN mkdir -p /app/serverTranspvisBack/static