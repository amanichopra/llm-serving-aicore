setup-venv:
	python3 -m venv .venv

build-ollama-image:
	docker build --platform=linux/amd64 -t docker.io/amanichopra/ollama-serve:latest ./ollama

build-pose-estimation-serve-image:
	docker build --platform=linux/amd64 -t docker.io/amanichopra/pose-estimation-yolo:latest ./pose-estimation

run-ollama-image-local:
	docker run -it --rm -p 8080:8080 amanichopra/ollama-serve:latest

run-pose-estimation-serve-image-local:
	docker run -it --rm -p 8080:8080 amanichopra/pose-estimation-yolo:latest

push-ollama-image:
	docker push docker.io/amanichopra/ollama-serve:latest

push-pose-estimation-serve-image:
	docker push docker.io/amanichopra/pose-estimation-yolo:latest

cf-login:
	cf login -a https://api.cf.eu12.hana.ondemand.com -o SLS-ATI-ML_agent-poc -s dev --sso