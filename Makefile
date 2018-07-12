  REGISTRY := quay.io
ifeq ($(TARGET),rhel)
  DOCKERFILE := Dockerfile.rhel
  REPOSITORY?=openshiftio/rhel-fabric8-analytics-f8a-npm-insights
else
  DOCKERFILE := Dockerfile
  REPOSITORY?=openshiftio/fabric8-analytics-f8a-npm-insights
endif
DEFAULT_TAG=latest

.PHONY: all docker-build fast-docker-build get-image-name get-image-repository

all: fast-docker-build

docker-build:
	docker build --no-cache -t $(REGISTRY)/$(REPOSITORY):$(DEFAULT_TAG) -f $(DOCKERFILE) .
	docker tag $(REGISTRY)/$(REPOSITORY):$(DEFAULT_TAG) npm-insights:testing

docker-build-test: docker-build
	docker build --no-cache -t chester-tests -f Dockerfile.tests .

fast-docker-build:
	docker build -t $(REGISTRY)/$(REPOSITORY):$(DEFAULT_TAG) -f $(DOCKERFILE) .

get-image-name:
	@echo $(REGISTRY)/$(REPOSITORY):$(DEFAULT_TAG)

get-image-repository:
	@echo $(REPOSITORY)

get-push-registry:
	@echo $(REGISTRY)
