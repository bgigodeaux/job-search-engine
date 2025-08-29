# Project variables
APP_NAME=smart-matcher
TAG=backbone
IMAGE=$(APP_NAME):$(TAG)
CONTAINER=$(APP_NAME)_container
PORT_API=8000
PORT_UI=8501

# Build Docker image
build:
	docker build -t $(IMAGE) .

# Run container (detached) mounting ./data into /app/data
run: stop
	docker run -d --rm \
		--name $(CONTAINER) \
		-p $(PORT_API):8000 -p $(PORT_UI):8501 \
		-v $(PWD)/data:/app/data \
		$(IMAGE)

# Follow container logs
logs:
	docker logs -f $(CONTAINER)

# Stop container
stop:
	-@docker stop $(CONTAINER) 2>/dev/null || true

# Rebuild & run fresh
up: build run logs

# Just remove the built image
clean:
	-@docker rmi $(IMAGE) 2>/dev/null || true