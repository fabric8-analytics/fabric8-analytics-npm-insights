FROM quay.io/farrion/python3-ml:latest

LABEL maintainer="Avishkar Gupta <avgupta@redhat.com>"

COPY ./recommendation_engine /recommendation_engine
COPY ./requirements.txt /requirements.txt
COPY ./entrypoint.sh /bin/entrypoint.sh

RUN chmod 0777 /bin/entrypoint.sh

RUN pip3 install --upgrade pip
RUN pip install git+https://github.com/fabric8-analytics/fabric8-analytics-rudra#egg=rudra
RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/entrypoint.sh"]
