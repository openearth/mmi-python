FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y \
    python3 \
    python3-gdal \
    python3-pip \
    && apt-get clean

COPY ./ app/
RUN pip3 install ./app
EXPOSE 22222
CMD [ "mmi", "tracker" ]
