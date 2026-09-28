# Use an official Python runtime as a parent image (3.11 gives us a newer CMake)
FROM python:3.11-slim

# Install C++ build tools and git (needed for FetchContent)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Generate large scale datasets on the fly (since we excluded them from Git to save space)
RUN python scripts/generate_million_constraints.py data/million.dat
RUN python scripts/generate_large_scale.py 20000 20000 0.005 data/bench_94_20k.dat
RUN python scripts/generate_standard_benchmarks.py

# Compile the C++ engine for Linux (Out-of-source build required by Eigen)
RUN mkdir build && cd build && cmake .. && make

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
