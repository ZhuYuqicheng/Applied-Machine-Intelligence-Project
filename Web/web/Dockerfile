ARG CONDA_CONTAINER=continuumio/miniconda3:4.9.2-alpine

FROM $CONDA_CONTAINER

ENV TZ=CET-1CEST,M3.5.0,M10.5.0/3
ENV TZ_DJANGO=Europe/Berlin

RUN apk add -U tzdata && \
    cp /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \
    echo "Europe/Berlin" > /etc/timezone

RUN pip install DateTime &&\
    pip install pmdarima

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority true && \
    conda install --quiet --yes \
    'pytorch'\
    'sktime' \
    'tzlocal' \
    'django' \
    'djangorestframework' \
    'django-cors-headers' \
    && \
  conda update --quiet --yes --all \
  && \
  conda clean --all -f -y


RUN mkdir web
COPY . /web
WORKDIR /web

EXPOSE 8888
CMD ["sh", "start-server.sh"]