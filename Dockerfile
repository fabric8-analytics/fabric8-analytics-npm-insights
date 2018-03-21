FROM centos:7

LABEL maintainer="Avishkar Gupta <avgupta@redhat.com>"

COPY ./recommendation_engine /recommendation_engine
COPY ./requirements.txt /requirements.txt
COPY ./entrypoint.sh /bin/entrypoint.sh

RUN yum -y install centos-release-scl
RUN yum-config-manager --enable centos-sclo-rh-testing
RUN yum -y install rh-python36 && yum -y install which
ENV PATH="/opt/rh/rh-python36/root/usr/bin/:${PATH}"
RUN chmod 0777 /bin/entrypoint.sh

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/entrypoint.sh"]
