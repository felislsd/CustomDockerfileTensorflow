#
# This example Dockerfile illustrates a method to install
# additional packages on top of NVIDIA's TensorFlow container image.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3

# Set environment variable to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install NVIDIA drivers and OpenGL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        && \
    rm -rf /var/lib/apt/lists/*


# Copy the requirements.txt file into the container at /app
COPY . /scripts
WORKDIR /scripts

# Install the dependencies
RUN pip install -r requirements.txt
