# Use an official Python 3 image as the base image
FROM python:3

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on your host to the /app directory in the container
COPY . /app

# Create a virtual environment in the /app directory
RUN python3 -m venv /app/venv

# Activate the virtual environment and install the required packages from requirements.txt
RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install -r requirements.txt

# Set the default command to run your main.py script
CMD ["/app/venv/bin/python", "main.py"]
