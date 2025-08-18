setup-venv:
	python3 -m venv .venv

build-ollama-image:
	docker build --platform=linux/amd64 -t docker.io/amanichopra/ollama-serve:latest ./ollama

build-custom-serve-image:
	docker build --platform=linux/amd64 -t docker.io/amanichopra/custom-serve:latest ./custom-server

run-ollama-image-local:
	docker run -it --rm -p 8080:8080 amanichopra/ollama-serve:latest

run-custom-serve-image-local:
	docker run -it --rm -p 8080:8080 amanichopra/custom-serve:latest

push-ollama-image:
	docker push docker.io/amanichopra/ollama-serve:latest

push-custom-serve-image:
	docker push docker.io/amanichopra/custom-serve:latest

cf-login:
	cf login -a https://api.cf.eu12.hana.ondemand.com -o SLS-ATI-ML_agent-poc -s dev --sso