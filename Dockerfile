# This is an Ubuntu image with system Python 3.10.
FROM nest/nest-simulator:3.4
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Make PyNEST available in python and install NESTML.
ENV PYTHONPATH=/opt/nest/lib/python3.10/site-packages
RUN pip install nestml==5.3.*

# Install braingeneerspy, always from the latest commit. Include IOT
# dependencies because we use Redis for job queues.
ADD "https://api.github.com/repos/braingeneers/braingeneerspy/commits?per_page=1" /tmp/latest_braingeneers_commit
RUN pip install "git+https://github.com/braingeneers/braingeneerspy#egg=braingeneerspy[analysis,iot]"

# Copy over the source files.
WORKDIR /root
COPY models models
COPY *.py .

ENTRYPOINT bash
