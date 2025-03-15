
# 1. Use a base image with PyTorch (change if needed)
# If GPU support is needed, switch to nvidia/cuda:12.1.1-devel-ubuntu20.04
# If changes are made to the dockerfile run -> docker build -t xai_thesis_image .
FROM python:3.8.19  

# 2. Set working directory inside the container
WORKDIR /workspace

# 3. Install system dependencies: 
    # wget (allows downloading external datasets or pre-trained models)
    # htop (monitoring tool to check CPU/memory usage inside container)
RUN apt-get update && apt-get install -y \
    git \ 
    wget \
    unzip \
    htop \
    libgl1-mesa-glx  # Needed for OpenCV visualization

# 4. Copy the repository files (excluding files in .dockerignore)
COPY . /workspace

# 5. Clone external repositories (e.g., CheXlocalize)
RUN git submodule update --init --recursive /third_party/cheXlocalize

# 6. Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 7. Create a user (optional, for permission management)
RUN useradd -ms /bin/bash dockeruser
USER dockeruser

# 8. Set environment variables (for flexibility)
ENV DATA_PATH="/external_data"
ENV MODEL_PATH="/models"

# 9. Expose ports (e.g., 8888 Jupyter, 6060 TensorBoard)
EXPOSE 8888
EXPOSE 6006

# 10. Default command: Open a shell
# Can be overrun when running the container e.g. "docker run -it xai_thesis python train.py"
CMD ["/bin/bash"]
