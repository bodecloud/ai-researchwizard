#docker buildx build -f Dockerfile.fullstack . --tag=bodencrouch/ai-researchwizard:latest

docker run -p 3000:3000 -p 8000:8000 --name=gptr --hostname=gptr --env-file .env bodencrouch/ai-researchwizard:latest

#CONTAINER_ID=$(docker run -d -p 3000:3000 -p 8000:8000 --container-name=gptr --hostname=gptr --env-file .env bodencrouch/ai-researchwizard:latest)
#docker container logs $CONTAINER_ID -f
