FROM registry.centos.org/centos/centos:7

LABEL maintainer="Avishkar Gupta <avgupta@redhat.com>"

COPY ./recommendation_engine /recommendation_engine
#COPY ./rudra /rudra
COPY ./requirements.txt /requirements.txt
COPY ./requirements_new.txt /requirements_new.txt
COPY ./entrypoint.sh /bin/entrypoint.sh
COPY ./training /training

RUN yum -y install gcc openssl-devel bzip2-devel libffi-devel &&\
    cd /tmp &&\
    yum -y install -v httpd httpd-devel wget git make &&\
    wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz &&\
    tar xzf Python-3.7.4.tgz &&\
    cd Python-3.7.4 &&\
    ./configure --enable-optimizations &&\
    make altinstall &&\
    export PATH="/usr/local/bin:$PATH" &&\
    python3.7 -m pip install --upgrade pip --user &&\
    python3.7 -m pip install numpy==1.16.5 Jinja2==2.10.1 --user &&\
    python3.7 -m pip install tensorflow==2.0.0b1 pandas boto3 scipy daiquiri flask h5py --user &&\
    python3.7 -m pip install git+https://github.com/fabric8-analytics/fabric8-analytics-rudra --user &&\

#RUN yum install -y epel-release &&\
#    yum install -y openssl-devel &&\
#    yum install -y gcc gcc-c++ git python36-pip python36-requests httpd httpd-devel python36-devel python-dev &&\
#    yum clean all

#RUN pip3 install pandas boto3 numpy tensorflow scipy daiquiri flask h5py --user

RUN chmod 0777 /bin/entrypoint.sh

#RUN pip3 install git+https://github.com/fabric8-analytics/fabric8-analytics-rudra#egg=rudra
#RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/entrypoint.sh"]
