TAG :=$(or ${CI_COMMIT_TAG}, latest)
DOCKER_IMAGE := rs-image-processing-service:${TAG}

build:
	docker build -t localhost:5000/${DOCKER_IMAGE} .

push:
	docker push localhost:5000/${DOCKER_IMAGE}