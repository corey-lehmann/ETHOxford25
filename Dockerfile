# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create a non-root user
RUN useradd -m streamlit
USER streamlit

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run streamlit when the container launches
ENTRYPOINT ["streamlit", "run"]
CMD ["TwitterAgent.py", "--server.port=8501", "--server.address=0.0.0.0"] 