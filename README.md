# TB2 AI: Tension Board 2 Route Generator

TB2 AI is a web application that generates climbing routes for the Tension Board 2 (TB2) using PyTorch models trained on Tension Board 2 climb data. It generates climbs based on customizable parameters such as board angle, size, layout, difficulty grade, and wall type (Mirror or Spray).

## Features

* **Route Generation**: Generate climbing routes for target difficulty grades from V0 to V15+.
* **Web Dashboard**: Select parameters (layout, size, grade, temperature, and board type) and view generated climbs on an interactive board layout.
* **TB2 Configurations**: Supports Mirror and Spray layouts, multiple board sizes, and Standard/No-Match rules.
* **Deployment Options**: Dockerfile and Kubernetes manifests for containerized hosting.

## Technical Stack

* **Backend**: Python, Flask, Gunicorn
* **Machine Learning**: PyTorch (CPU inference)
* **Database / Data Store**: JSON Metadata (TB2 hold data, placements, and coordinates)
* **Frontend**: HTML, CSS, JavaScript
* **Containerization**: Docker
* **Orchestration**: Kubernetes, Cloudflare Tunnel Ingress

## Local Development

### 1. Prerequisites
* Python 3.11+

### 2. Setup
```bash
# Clone and enter the repository directory
cd "tb2ai"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run Development Server
```bash
export FLASK_APP=wsgi.py
export FLASK_ENV=development
flask run --port=8000
```
Visit http://localhost:8000.

## Docker Deployment

### Build the Image
```bash
docker build -t timshn/tb2ai:1.0 .
```

### Run Locally
```bash
docker run -d -p 8000:8000 --name tb2-ai timshn/tb2ai:1.0
```

## Kubernetes Deployment

### Deploy Resources
```bash
kubectl apply -f k8s/
```
Deploys the namespace, a 3-replica deployment, NodePort service, and Cloudflare Ingress.

### Useful Commands
```bash
# Check pod status
kubectl get pods -n tb2ai

# Check ingress status
kubectl get ingress -n tb2ai
```

## API Documentation

### POST /api/v1/generate

Generates a climb route based on input configuration.

#### Request Headers
* `Content-Type: application/json`

#### Request Body Schema
```json
{
  "layout_id": 11,
  "size_id": 8,
  "grade": "V5",
  "temperature": 0.7,
  "angle": 40,
  "is_nomatch": false,
  "max_len": 20,
  "beam_width": 4
}
```
* `layout_id`: (Integer) Mirror (10) or Spray (11). Default is 11.
* `size_id`: (Integer) Board size ID. Default is 8. Options: 8 (12x8), 9 (10x8), 6 (12x12), 7 (10x12), 10 (12x16).
* `grade`: (String) Target grade (V0 to V16). Default is "V5".
* `temperature`: (Float) Sampling variance. Default is 0.7.
* `angle`: (Integer) Board angle. Default is 40.
* `is_nomatch`: (Boolean) No-match rules flag. Default is false.

#### Example Response
```json
{
  "frames": "p1421r5p1455r6p1502r7"
}
```

## License
Educational and recreational climbing board research purposes.
