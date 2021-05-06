FROM registry.access.redhat.com/ubi8/python-36:latest

LABEL name="f8analytics cvae npm insights service" \
      description="Fabric8 analytic npm insights service for recommendation." \
      email-ids="dhpatel@redhat.com" \
      git-url="https://github.com/fabric8-analytics/fabric8-analytics-npm-insights" \
      git-path="/" \
      target-file="Dockerfile" \
      app-license="GPL-3.0"

ENV LANG=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1

COPY ./requirements.txt /opt/app-root

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r /opt/app-root/requirements.txt

COPY ./recommendation_engine /opt/app-root/src/recommendation_engine

ADD ./entrypoint.sh /bin/entrypoint.sh

ENTRYPOINT ["bash", "/bin/entrypoint.sh"]
