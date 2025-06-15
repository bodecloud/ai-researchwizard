docker buildx build -f Dockerfile.fullstack . --tag=bodencrouch/ai-researchwizard:latest
CONTAINER_ID=$(docker run -p 3000:3000 -p 8000:8000 -v .env:/app/.env --env-file .env bodencrouch/ai-researchwizard:latest)
docker container logs $CONTAINER_ID -f
