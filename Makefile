setup-venv:
	python3 -m venv .venv

build-ollama-image:
	docker build --platform=linux/amd64 -t docker.io/amanichopra/ollama-serve:latest ./ollama

run-ollama-image-local:
	docker run -it --rm -p 11434:11434 amanichopra/ollama-serve:latest

push-ollama-image:
	docker push docker.io/amanichopra/ollama-serve:latest

cf-login:
	cf login -a https://api.cf.eu12.hana.ondemand.com -o SLS-ATI-ML_agent-poc -s dev --sso