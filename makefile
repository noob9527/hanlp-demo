# Image name and tag
IMAGE_NAME = hanlp-server
TAG = latest

# Full image reference
IMAGE = $(IMAGE_NAME):$(TAG)

# Default port mapping
HOST_PORT = 5012
CONTAINER_PORT = 80

.PHONY: build run stop clean shell

# generate requirements.txt
# https://github.com/astral-sh/uv/issues/6007#issuecomment-2310990807
requirements:
	uv export --format requirements-txt > requirements.txt

# Build the Docker image
build: requirements
	docker build -t $(IMAGE) .

# Run the container
run:
	docker run -d \
		--name $(IMAGE_NAME) \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		$(IMAGE)

# Clean up images
clean: stop
	docker rmi $(IMAGE)

# Run container with interactive shell
shell:
	docker run -it --rm \
		$(IMAGE) /bin/bash
