# Select the base image
FROM nvcr.io/nvidia/pytorch:21.06-py3

# Select the working directory
WORKDIR  /thesis-latplan

# Install Python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt