FROM tensorflow/tensorflow:latest

LABEL maintainer="Fernando Silva <fernando.f.silva@outlook.com>"

# Copy all files to app folder
COPY .. /app

# Set working directory
WORKDIR /app

# Install packages
RUN apt-get update \
        && apt-get install -y \
        cmake \
        libboost-all-dev \
        libjpeg-dev \
        libsdl2-dev \
        libx11-6 \
        python-opengl \
        swig \
        xorg-dev \
        xvfb \
        zlib1g-dev \
        && rm -rf /var/apt/lists/*

## Upgrade pip
RUN pip3 install --upgrade pip

# Install
RUN pip install -r requirements.txt

# Run python script
ENTRYPOINT ["python", "/app/codes/main.py"]