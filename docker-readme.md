Crypto Twitter Debate Analyzer

A sophisticated analysis system that identifies and analyzes debates and conflicts happening in Crypto Twitter.

Prerequisites:
- Docker
- Docker Compose (optional)

Environment Variables:
Create a .env file with the following variables:

API_KEY=your_api_key
MODEL=your_model
ENDPOINT=your_endpoint
REGION=your_region
AUTH=your_auth
TWITTER_AUTH=your_twitter_auth

Running with Docker Compose:

1. Build the container:
docker-compose build

2. Run the container:
docker-compose up

Running with Docker:

1. Build the container:
docker build -t twitter-analyzer .

2. Run the container:
docker run -p 8501:8501 \
  -e API_KEY=${API_KEY} \
  -e MODEL=${MODEL} \
  -e ENDPOINT=${ENDPOINT} \
  -e REGION=${REGION} \
  -e AUTH=${AUTH} \
  -e TWITTER_AUTH=${TWITTER_AUTH} \
  twitter-analyzer

Accessing the Application:
Once running, access the application at: http://localhost:8501 