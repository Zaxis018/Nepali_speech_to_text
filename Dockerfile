# Use the official Python image as the base image
FROM python:3.10

# Install system dependencies necessary for the project
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libasound-dev \
#     libportaudio2 \
#     libportaudiocpp0 \
#     portaudio19-dev \
#     ffmpeg \
#     && apt-get clean

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies, excluding `pyaudio`
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Manually install a pre-built pyaudio wheel
RUN pip install --no-cache-dir "https://files.pythonhosted.org/packages/c4/30/8314e22f19d7ae95f4a1e1064c87f9e27b360cfce5c7b5e7c0b4a7427f2c/PyAudio-0.2.11-cp310-cp310-manylinux2014_x86_64.whl"

# Copy the entire codebase into the container
COPY . .

# Expose the default port for Streamlit
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py"]
