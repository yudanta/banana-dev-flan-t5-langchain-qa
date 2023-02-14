FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# install os dependencies
RUN apt-get update &&  apt-get install  -y git

# install  python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# banana boilerrplate
ADD http_api.py .
ADD decorators.py .

# add model weight files or download
ADD download.py .
RUN python3 download.py

# add custom app.py code init() and inference here()
ADD app.py .

# expos http port 
EXPOSE 8000

# kick http server process
CMD python3 -u http_api.py