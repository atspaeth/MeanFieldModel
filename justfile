export hash := `git log -n 1 --pretty=format:"%h"`
export container := "atspaeth/mean-field:" + hash

help:
    @just --list

build:
    @git diff-index --quiet HEAD -- || (echo "Won't build with uncommited changes."; exit 1)
    docker build -t $container .

debug: build
    docker run --rm -it $container

pull:
    docker pull $container

push: build
    docker push $container
    @echo Current commit uploaded to $container
