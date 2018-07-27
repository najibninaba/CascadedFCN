FROM ubuntu:latest

RUN apt-get update -y \
    && apt-get install -y python-pip \
    python-dev \
    build-essential \
    python-tk 
WORKDIR /CascadedFCN

COPY . /CascadedFCN
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["/usr/bin/python", "src/train.py"]

