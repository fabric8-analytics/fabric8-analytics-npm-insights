FROM registry.centos.org/centos/centos:7

LABEL maintainer="Avishkar Gupta <avgupta@redhat.com>"

COPY ./recommendation_engine /recommendation_engine
COPY ./requirements.txt /requirements.txt
COPY ./requirements_new.txt /requirements_new.txt
COPY ./entrypoint.sh /bin/entrypoint.sh
COPY ./training /training

RUN yum install -y epel-release &&\
    yum install -y openssl-devel &&\
    yum install -y gcc git python36-pip python36-requests httpd httpd-devel python36-devel &&\
    yum clean all

RUN chmod 0777 /bin/entrypoint.sh

RUN pip3 install git+https://github.com/fabric8-analytics/fabric8-analytics-rudra#egg=rudra
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/entrypoint.sh"]
