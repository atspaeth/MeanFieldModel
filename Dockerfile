# This is an Ubuntu image with system Python 3.10.
FROM nest/nest-simulator:3.4

# Make PyNEST available in python and install braingeneerspy.
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/opt/nest/lib/python3.10/site-packages
RUN pip install "braingeneerspy[analysis,iot]"

# Set up the working directory.
WORKDIR /root
COPY *.py .
ENTRYPOINT bash
