FROM centos:7

LABEL maintainer="Avishkar Gupta <avgupta@redhat.com>"

COPY ./recommendation_engine /recommendation_engine
COPY ./requirements.txt /requirements.txt
COPY ./entrypoint.sh /bin/entrypoint.sh

RUN yum install -y epel-release &&\
    yum install -y gcc git python34-pip python34-requests httpd httpd-devel python34-devel &&\
    yum clean all

RUN chmod 0777 /bin/entrypoint.sh

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/entrypoint.sh"]
