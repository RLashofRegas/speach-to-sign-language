# install necessary dependencies for running the docker images

# install docker-compose for arm64 via pip
sudo apt-get update -y
sudo apt-get install -y curl
sudo apt-get install -y libssl-dev
sudo apt-get install -y libffi-dev
sudo apt-get install -y python-openssl
curl -sSL https://bootstrap.pypa.io/get-pip.py | sudo python
export DOCKER_COMPOSE_VERSION=1.24.0
sudo pip install docker-compose=="${DOCKER_COMPOSE_VERSION}"

# add current user to docker group to avoid needing to use sudo
sudo usermod -aG docker $USER
