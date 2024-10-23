# Use an official Python runtime as a parent image
FROM python:3.12.4-slim

# Set environment variables to avoid running as root and for buffer handling
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /evidai

# Copy the requirements file into the container
COPY requirements.txt /evidai/

# Install the dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project into the container
COPY . /evidai/

# Collect static files (you can modify this if static files are handled separately)
RUN python manage.py collectstatic --noinput

# Expose port 8000 (or your app’s port)
EXPOSE 8000

# Start the Django development server (this will change for production)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
