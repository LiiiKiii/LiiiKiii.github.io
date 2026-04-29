# AI-Pedia Deployment Guide

## Prerequisites

- Docker 20.10 or newer
- Docker Compose 1.29 or newer
- At least 2 GB of available memory
- Optional: an OpenAI API key for summary generation

## Quick Deployment

### Option 1: Docker Compose

1. Clone the project

```bash
git clone <repository-url>
cd AI-Pedia/Project/Code
```

2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` if you want to enable optional LLM summaries.

3. Start the service

```bash
docker-compose up -d
docker-compose logs -f ai-pedia
```

4. Open the application

Visit `http://localhost:5000`

5. Stop the service

```bash
docker-compose down
```

### Option 2: Direct Docker Build

```bash
docker build -t ai-pedia:latest .

docker run -d \
  --name ai-pedia \
  -p 5000:5000 \
  -e OPENAI_API_KEY=your_api_key_here \
  -v $(pwd)/data:/app/data \
  ai-pedia:latest

docker logs -f ai-pedia
```

Stop and remove the container:

```bash
docker stop ai-pedia
docker rm ai-pedia
```

### Option 3: Local Python Run

1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables

```bash
export OPENAI_API_KEY=your_api_key_here
export FLASK_ENV=production
```

3. Start the application

```bash
python3 app.py
```

Then open `http://localhost:5000`.

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | No | - | Optional OpenAI API key for summaries. Without it, the fallback summary logic is used. |
| `FLASK_ENV` | No | `production` | Flask runtime environment. |
| `FLASK_DEBUG` | No | `False` | Flask debug mode. |
| `FLASK_PORT` | No | `5000` | Service port. |

### Pipeline Parameters

Project settings are centralized in `config.py`. Typical values include:

- keyword extraction count
- search limits per resource type
- recommendation count per type
- upload size and minimum document threshold

## Monitoring and Maintenance

### Health Check

The application exposes a `/health` endpoint:

```bash
curl http://localhost:5000/health
```

### Logs

Docker Compose:

```bash
docker-compose logs -f ai-pedia
```

Docker:

```bash
docker logs -f ai-pedia
```

### Data Backup

Project data is stored under `data/`:

```bash
tar -czf ai-pedia-backup-$(date +%Y%m%d).tar.gz data/
```

Restore with:

```bash
tar -xzf ai-pedia-backup-YYYYMMDD.tar.gz
```

## Security Notes

1. Protect API keys
   - Do not hard-code API keys in source files.
   - Prefer environment variables.
   - Rotate keys regularly if they are used.

2. Network exposure
   - Use a reverse proxy such as Nginx in public deployments.
   - Enable HTTPS if the service is exposed externally.

3. File handling
   - Keep upload permissions restricted.
   - Periodically clean `data/uploads` if needed.

## Troubleshooting

### Container exits immediately

```bash
docker-compose logs ai-pedia
netstat -an | grep 5000
```

If port `5000` is busy, remap the host port.

### OpenAI calls fail

Check whether the key is set correctly:

```bash
docker-compose exec ai-pedia env | grep OPENAI
```

### Upload succeeds but processing fails

Check writable directories and restart the service if needed:

```bash
docker-compose exec ai-pedia ls -la data/uploads/
docker-compose restart
```

## Updating a Deployment

```bash
git pull origin main
docker-compose up -d --build
```
