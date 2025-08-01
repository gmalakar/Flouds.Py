> **Note:**  
> This project is under active development and we are looking for more collaborators to help improve and extend Flouds AI!  
> If you're interested in contributing, please reach out or open a pull request.

# Flouds AI

**Flouds AI** is an enterprise-grade Python NLP service framework for text summarization and embedding, built with FastAPI and ONNX runtime. It features comprehensive monitoring, security, and performance optimizations for scalable deployment.

---

## ✨ Key Features

### 🤖 **AI Capabilities**
- **Advanced Text Summarization**: Seq2seq models with ONNX optimization and automatic sentence capitalization
- **High-Performance Embeddings**: Sentence and document embeddings with configurable chunking strategies
- **Batch Processing**: Async batch operations for high-throughput scenarios
- **Model Optimization**: Optimized ONNX models with automatic fallback and KV caching

### 🚀 **Enterprise Features**
- **Performance Monitoring**: Real-time system metrics, memory tracking, and performance profiling
- **Advanced Health Checks**: Kubernetes-ready liveness/readiness probes with detailed diagnostics
- **Request Validation**: Size limits, timeout handling, and comprehensive error responses
- **Rate Limiting**: Configurable per-IP rate limiting with automatic cleanup
- **Security**: CORS protection, trusted host validation, and request sanitization

### ⚙️ **Configuration & Deployment**
- **Environment-Aware Config**: Development/production configs with environment variable overrides
- **Docker Ready**: Multi-stage builds with GPU support and optimized images
- **Comprehensive Logging**: Structured logging with rotation and configurable levels
- **Resource Management**: Memory/CPU threshold monitoring with automatic alerts

---

## 📁 Project Structure

```
app/
  config/
    appsettings.json              # Enterprise configuration
    appsettings.development.json  # Development overrides
    onnx_config.json             # ONNX model configurations
    config_loader.py             # Enhanced config management
  middleware/
    rate_limit.py                # Advanced rate limiting
    request_validation.py        # Request size/timeout validation
  models/                        # Pydantic request/response models
  routers/                       # FastAPI route handlers
  services/                      # Core NLP service logic
  utils/
    performance_monitor.py       # System performance tracking
    memory_monitor.py           # Memory usage monitoring
    model_cache.py              # LRU model caching
  main.py                       # Enhanced FastAPI application
onnx_loaders/                   # Model export utilities
tests/                          # Comprehensive test suite
.env.example                    # Environment configuration template
```

---

## ⚙️ Configuration

### Enhanced Configuration System

Flouds AI features a sophisticated configuration system with environment-specific overrides and comprehensive settings.

**Key Configuration Files:**
- `appsettings.json` - Enterprise configuration
- `appsettings.development.json` - Development overrides
- `.env` - Environment variables (copy from `.env.example`)

### Core Configuration Sections

```json
{
    "app": {
        "name": "Flouds AI",
        "version": "1.0.0",
        "cors_origins": ["*"],
        "max_request_size": 10485760,
        "request_timeout": 300
    },
    "server": {
        "host": "0.0.0.0",
        "port": 19690,
        "session_provider": "CPUExecutionProvider",
        "keepalive_timeout": 5,
        "graceful_timeout": 30
    },
    "rate_limiting": {
        "enabled": true,
        "requests_per_minute": 200,
        "requests_per_hour": 5000
    },
    "monitoring": {
        "enable_metrics": true,
        "memory_threshold_mb": 1024,
        "cpu_threshold_percent": 80
    },
    "security": {
        "enabled": false,
        "clients_db_path": "clients.db"
    }
}
```

### Environment Variables

**Optional:**
- `FLOUDS_API_ENV` - Environment (Development/Enterprise)
- `APP_DEBUG_MODE` - Enable debug logging (0/1)
- `SERVER_PORT` - Override server port
- `FLOUDS_RATE_LIMIT_PER_MINUTE` - Rate limit override
- `FLOUDS_CORS_ORIGINS` - Comma-separated CORS origins
- `FLOUDS_SECURITY_ENABLED` - Enable client authentication
- `FLOUDS_ENCRYPTION_KEY` - Base64 encoded encryption key for client credentials (if not set, auto-generated and stored in `.encryption_key` file under the folder where the db file resides)

**Note**: `FLOUDS_ONNX_ROOT`, `FLOUDS_ONNX_CONFIG_FILE`, and `FLOUDS_CLIENTS_DB` are automatically set by deployment scripts.

---

### onnx_config.json

The `onnx_config.json` file (located in `app/config/onnx_config.json`) is used to configure all ONNX models that you want to use in your application.  
Each entry in this file corresponds to a model you have downloaded and placed in your ONNX model folder.

- **Key**: The model name (e.g., `"t5-small"`, `"sentence-t5-base"`)
- **Value**: A dictionary describing the model's configuration, including:
  - `dimension`, `max_length`, `embedder_task` or `summarization_task`
  - `inputnames`, `outputnames`, `decoder_inputnames`
  - ONNX model file paths (`encoder_onnx_model`, `decoder_onnx_model`)
  - Optimized model paths (`encoder_optimized_onnx_model`, `decoder_optimized_onnx_model`)
  - Performance flags (`use_optimized`, `legacy_tokenizer`)
  - Special tokens, generation config, and other options

**Example snippet:**
```json
"t5-small": {
    "dimension": 512,
    "max_length": 512,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "summarization_task": "s2s",
    "legacy_tokenizer": true,
    "use_optimized": false,
    "inputnames": {
        "input": "input_ids",
        "mask": "attention_mask"
    },
    "outputnames": {
        "output": "last_hidden_state"
    },
    "decoder_inputnames": {
        "encoder_output": "encoder_hidden_states",
        "input": "input_ids",
        "mask": "encoder_attention_mask"
    },
    "encoder_onnx_model": "encoder_model.onnx",
    "decoder_onnx_model": "decoder_model.onnx",
    "encoder_optimized_onnx_model": "encoder_model_optimized.onnx",
    "decoder_optimized_onnx_model": "decoder_model_optimized.onnx",
    "special_tokens_map_path": "special_tokens_map.json",
    "num_beams": 4,
    "early_stopping": true,
    "use_seq2seqlm": false
}
```

- The structure of your ONNX model folder should match the configuration in this file.

#### ONNX Model Folder Structure

Model paths are organized by task name. For example, for summarization models with `summarization_task: "s2s"`, the path will be:

```
/onnx/models/s2s/t5-small/
    encoder_model.onnx
    decoder_model.onnx
    special_tokens_map.json
```

Here, `s2s` is the task name for summarization, and `t5-small` is the model name.  
For embedding models, the folder will use the `embedder_task` value.

**Note:**  
The `summarization_task` or `embedder_task` values are typically set to the same value as the `--model_for` argument used when exporting models to ONNX (see `onnx_loaders/load_scripts.txt`).  
Just make sure the path you specify in `summarization_task` or `embedder_task` matches the folder structure where your ONNX models and config files are stored.

#### `use_seq2seqlm` Option

- **`use_seq2seqlm: true`**  
  Uses `ORTModelForSeq2SeqLM` and its `.generate()` method for summarization (recommended for supported models).
- **`use_seq2seqlm: false`** (default)  
  Uses the lower-level `ort.InferenceSession` for both encoding and decoding.

#### Embedder Model Notes

- For embedding models, if your ONNX model file is named `model.onnx`, you do **not** need to specify the model name in the config (`encoder_onnx_model` is optional).
- If your model file has a different name, set `encoder_onnx_model` to the correct filename.
- The `logits` flag:  
  - If you know the encoder output is logits, set `"logits": true` in your config.
  - If not set, the process will try to detect if the output is logits and process accordingly.

---

## 🚀 Quick Start

### Development Setup

1. **Clone and Install Dependencies**
```bash
git clone <repository-url>
cd Flouds.Py
pip install -r app/requirements.txt  # Production only
# OR for development:
pip install -r requirements-dev.txt  # Production + development
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your ONNX model path
```

3. **Export ONNX Models** (see [ONNX Export Guide](#exporting-models-to-onnx))
```bash
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/all-MiniLM-L6-v2" --optimize
```

4. **Run Development Server**
```bash
# Windows
set FLOUDS_API_ENV=Development
set APP_DEBUG_MODE=1
set FLOUDS_ONNX_ROOT=path/to/your/onnx
python -m app.main

# Linux/macOS
export FLOUDS_API_ENV=Development
export APP_DEBUG_MODE=1
export FLOUDS_ONNX_ROOT=/path/to/your/onnx
python -m app.main
```

### Enterprise Deployment

```bash
# Using Docker (recommended)
docker run -p 19690:19690 \
  -v /path/to/onnx:/flouds-ai/onnx \
  -e FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
  gmalakar/flouds-ai-cpu:latest

# Direct deployment
export FLOUDS_API_ENV=Enterprise
export FLOUDS_ONNX_ROOT=/path/to/onnx
python -m app.main
```

---

## Exporting Models to ONNX

To use ONNX models, you need to export them from HuggingFace format.  
Use the scripts in `onnx_loaders/load_scripts.txt` as examples:

```plaintext
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/all-MiniLM-L6-v2" --optimize
python onnx_loaders/export_model.py --model_for "s2s" --model_name "t5-small" --optimize --task "seq2seq-lm"
python onnx_loaders/export_model.py --model_for "fe" --model_name "PleIAs/Pleias-Pico" --optimize
python onnx_loaders/export_model.py --model_for "s2s" --model_name "PleIAs/Pleias-Pico" --optimize --task "seq2seq-lm"
python onnx_loaders/export_model.py --model_for "fe" --model_name "sentence-transformers/sentence-t5-base" --optimize --use_t5_encoder
python onnx_loaders/export_model.py --model_for "s2s" --model_name "google/pegasus-cnn_dailymail" --task "seq2seq-lm"
python onnx_loaders/export_model.py --model_for "s2s" --model_name "Falconsai/text_summarization" --task "seq2seq-lm" --model_folder "falconsai_text_summarization"
```

- `"fe"` = feature extraction (embedding)
- `"s2s"` = sequence-to-sequence (summarization)
- `--optimize` enables ONNX graph optimizations
- `--task` specifies the HuggingFace task type

---

## 🔐 Authentication

### Client-Based Authentication

Flouds AI uses a modern client-based authentication system with encrypted storage.

**Token Format**: `client_id|client_secret`

#### Auto-Generated Admin
On first startup, an admin client is automatically created:
- Credentials logged to console and saved to `admin_credentials.txt`
- Use admin token to manage other clients

#### Client Management CLI

```bash
# Add new API client
python generate_token.py add my-app --type api_user

# Add admin client
python generate_token.py add admin-user --type admin

# List all clients
python generate_token.py list

# Remove client
python generate_token.py remove my-app
```

#### Admin API Endpoints

```bash
# Generate client key (admin only)
curl -H "Authorization: Bearer admin|<admin_secret>" \
  -X POST "http://localhost:19690/api/v1/admin/generate-key" \
  -d '{"client_id": "app1"}'

# List clients (admin only)
curl -H "Authorization: Bearer admin|<admin_secret>" \
  "http://localhost:19690/api/v1/admin/clients"

# Remove client (admin only)
curl -H "Authorization: Bearer admin|<admin_secret>" \
  -X DELETE "http://localhost:19690/api/v1/admin/remove-client/app1"
```

---

## 📚 API Usage

### REST API Endpoints

Flouds AI provides a comprehensive REST API with automatic documentation at `/docs`.

#### Text Summarization

```bash
# Single summarization
curl -X POST "http://localhost:19690/api/v1/summarizer/summarize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer client_id|client_secret" \
  -d '{
    "model": "t5-small",
    "input": "Your long text to summarize here...",
    "temperature": 0.7
  }'

# Batch summarization
curl -X POST "http://localhost:19690/api/v1/summarizer/summarize_batch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer client_id|client_secret" \
  -d '{
    "model": "t5-small",
    "inputs": ["Text 1...", "Text 2..."],
    "temperature": 0.5
  }'
```

#### Text Embedding

```bash
# Single embedding
curl -X POST "http://localhost:19690/api/v1/embedder/embed" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer client_id|client_secret" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "input": "Text to embed",
    "projected_dimension": 128
  }'

# Batch embedding
curl -X POST "http://localhost:19690/api/v1/embedder/embed_batch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer client_id|client_secret" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "inputs": ["Text 1", "Text 2"],
    "projected_dimension": 128
  }'
```

### Python SDK Usage

```python
# Summarization
from app.models.summarization_request import SummarizationRequest
from app.services.summarizer_service import TextSummarizer

request = SummarizationRequest(
    model="t5-small",
    input="Your text to summarize",
    temperature=0.7
)
response = TextSummarizer.summarize(request)
print(response.results.summary)

# Embedding
from app.models.embedding_request import EmbeddingRequest
from app.services.embedder_service import SentenceTransformer

request = EmbeddingRequest(
    model="all-MiniLM-L6-v2",
    input="Text to embed",
    projected_dimension=128
)
response = SentenceTransformer.embed_text(
    text=request.input,
    model_to_use=request.model,
    projected_dimension=request.projected_dimension
)
print(response.results)
```

---

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Pull latest image
docker pull gmalakar/flouds-ai-cpu:latest

# Run with your ONNX models
docker run -d \
  --name flouds-ai \
  -p 19690:19690 \
  -v /path/to/your/onnx:/flouds-ai/onnx \
  -e FLOUDS_ONNX_ROOT=/flouds-ai/onnx \
  -e FLOUDS_API_ENV=Enterprise \
  --restart unless-stopped \
  gmalakar/flouds-ai-cpu:latest
```

### Build Custom Image

```bash
# CPU version
docker build -t flouds-ai-cpu .

# GPU version (requires NVIDIA Docker)
docker build --build-arg GPU=true -t flouds-ai-gpu .
```

### Enterprise Docker Compose

```yaml
version: '3.8'
services:
  flouds-ai:
    image: gmalakar/flouds-ai-cpu:latest
    ports:
      - "19690:19690"
    volumes:
      - ./onnx:/flouds-ai/onnx
      - ./logs:/flouds-ai/logs
    environment:
      - FLOUDS_ONNX_ROOT=/flouds-ai/onnx
      - FLOUDS_API_ENV=Enterprise
      - FLOUDS_RATE_LIMIT_PER_MINUTE=500
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:19690/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

### Automated Deployment Scripts

Flouds AI includes comprehensive deployment scripts for easy Docker container management.

#### PowerShell Script (Windows) - `start-flouds-ai.ps1`

**Basic Usage:**
```powershell
# Start with default settings
.\start-flouds-ai.ps1

# Start with custom environment file
.\start-flouds-ai.ps1 -EnvFile .env.production

# Force restart existing container
.\start-flouds-ai.ps1 -Force

# Build image locally and start
.\start-flouds-ai.ps1 -BuildImage -Force

# Use GPU image
.\start-flouds-ai.ps1 -GPU

# Always pull latest image
.\start-flouds-ai.ps1 -PullAlways
```

**Parameters:**
- `-EnvFile` - Path to environment file (default: `.env`)
- `-InstanceName` - Container name (default: `flouds-ai-instance`)
- `-ImageName` - Docker image name (default: `gmalakar/flouds-ai-cpu`)
- `-Tag` - Image tag (default: `latest`)
- `-GPU` - Use GPU image instead of CPU
- `-Force` - Force restart if container exists
- `-BuildImage` - Build image locally before starting
- `-PullAlways` - Always pull latest image from registry

#### Bash Script (Linux/macOS) - `start-flouds-ai.sh`

**Basic Usage:**
```bash
# Start with default settings
./start-flouds-ai.sh

# Start with custom environment file
./start-flouds-ai.sh --env-file .env.production

# Force restart existing container
./start-flouds-ai.sh --force

# Build image locally and start
./start-flouds-ai.sh --build --force

# Development mode with GPU
./start-flouds-ai.sh --gpu --development
```

**Parameters:**
- `--env-file` - Path to environment file (default: `.env`)
- `--instance` - Container name (default: `flouds-ai-instance`)
- `--image` - Docker image name (default: `gmalakar/flouds-ai-cpu`)
- `--tag` - Image tag (default: `latest`)
- `--gpu` - Use GPU image instead of CPU
- `--force` - Force restart if container exists
- `--build` - Build image locally before starting
- `--pull-always` - Always pull latest image
- `--development` - Run in development mode

#### Build Script - `build-flouds-ai.ps1`

**Usage:**
```powershell
# Build CPU image
.\build-flouds-ai.ps1

# Build GPU image with custom tag
.\build-flouds-ai.ps1 -Tag v1.2.0 -GPU

# Build and push to registry
.\build-flouds-ai.ps1 -PushImage

# Force rebuild existing image
.\build-flouds-ai.ps1 -Force
```

**Parameters:**
- `-Tag` - Image tag (default: `latest`)
- `-GPU` - Build with GPU support
- `-PushImage` - Push to Docker registry after build
- `-Force` - Force rebuild even if image exists

#### Required Environment File Structure

Create a `.env` file with the following required variables:

```bash
# Required: ONNX model configuration
FLOUDS_ONNX_CONFIG_FILE_AT_HOST=/path/to/your/onnx_config.json
FLOUDS_ONNX_MODEL_PATH_AT_HOST=/path/to/your/onnx/models

# Optional: Log persistence
FLOUDS_LOG_PATH_AT_HOST=/path/to/logs

# Optional: Docker platform
DOCKER_PLATFORM=linux/amd64
```

#### Script Features

**Automatic Setup:**
- ✅ Docker availability checking
- ✅ Environment file validation
- ✅ Directory permission management
- ✅ Network creation and management
- ✅ Container lifecycle management

**Volume Mapping:**
- ✅ ONNX models directory (read-only)
- ✅ Configuration files (read-only)
- ✅ Log directory (read-write)
- ✅ Automatic directory creation

**Error Handling:**
- ✅ Comprehensive validation
- ✅ Graceful error messages
- ✅ Automatic cleanup on failure
- ✅ Interactive confirmations

#### Example Complete Workflow

```powershell
# 1. Setup environment
cp .env.example .env
# Edit .env with your paths

# 2. Build custom image (optional)
.\build-flouds-ai.ps1 -Tag production

# 3. Start container
.\start-flouds-ai.ps1 -Tag production -Force

# 4. View logs
docker logs -f flouds-ai-instance

# 5. Test API
curl http://localhost:19690/api/v1/health
```

---

## 📊 Monitoring & Health Checks

### Health Endpoints

```bash
# Basic health check
curl http://localhost:19690/api/v1/health

# Detailed system information
curl http://localhost:19690/api/v1/health/detailed

# Kubernetes probes
curl http://localhost:19690/api/v1/health/ready   # Readiness probe
curl http://localhost:19690/api/v1/health/live    # Liveness probe
```

### Performance Monitoring

Flouds AI includes comprehensive monitoring:

- **Real-time Metrics**: Memory usage, CPU utilization, request timing
- **Resource Thresholds**: Configurable alerts for memory/CPU limits
- **Request Analytics**: Slow request detection and logging
- **Model Cache Monitoring**: Track cached models and sessions

### Logging

```bash
# View logs using deployment scripts
# PowerShell
.\start-flouds-ai.ps1  # Will prompt to view logs after startup

# Bash
./start-flouds-ai.sh   # Will prompt to view logs after startup

# Manual log viewing
docker logs -f flouds-ai-instance

# Log files (direct deployment)
tail -f logs/flouds-ai-$(date +%Y-%m-%d).log
```

### Container Management

```bash
# Using deployment scripts (recommended)
.\start-flouds-ai.ps1 -Force  # Restart container
./start-flouds-ai.sh --force   # Restart container

# Manual Docker commands
docker stop flouds-ai-instance
docker start flouds-ai-instance
docker restart flouds-ai-instance
docker rm flouds-ai-instance
```

---

## 🔧 Advanced Configuration

### Complete Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FLOUDS_ONNX_ROOT` | ONNX models directory | - | ✅ |
| `FLOUDS_API_ENV` | Environment (Development/Enterprise) | Enterprise | ❌ |
| `APP_DEBUG_MODE` | Enable debug logging (0/1) | 0 | ❌ |
| `SERVER_PORT` | Server port | 19690 | ❌ |
| `SERVER_HOST` | Server host | 0.0.0.0 | ❌ |
| `FLOUDS_RATE_LIMIT_PER_MINUTE` | Rate limit per minute | 200 | ❌ |
| `FLOUDS_RATE_LIMIT_PER_HOUR` | Rate limit per hour | 5000 | ❌ |
| `FLOUDS_MAX_REQUEST_SIZE` | Max request size (bytes) | 10485760 | ❌ |
| `FLOUDS_MODEL_SESSION_PROVIDER` | ONNX provider | CPUExecutionProvider | ❌ |
| `FLOUDS_CORS_ORIGINS` | CORS origins (comma-separated) | * | ❌ |
| `FLOUDS_ENCRYPTION_KEY` | Base64 encoded encryption key | Auto-generated file | ❌ |

### Performance Tuning

```bash
# High-performance configuration
export FLOUDS_MODEL_CACHE_SIZE=10
export FLOUDS_MODEL_CACHE_TTL=7200
export FLOUDS_RATE_LIMIT_PER_MINUTE=1000
export FLOUDS_MAX_REQUEST_SIZE=52428800  # 50MB

# GPU acceleration
export FLOUDS_MODEL_SESSION_PROVIDER=CUDAExecutionProvider
```

---

## 🎯 Model Management

### ONNX Model Optimization

Flouds AI provides advanced model optimization features:

- **Automatic Optimization**: Graph-optimized models with 20-50% performance improvement
- **KV Caching**: Faster seq2seq inference with decoder caching
- **Model Fallback**: Automatic fallback to regular models if optimized versions fail
- **Legacy Support**: Handles tokenizer compatibility across transformers versions

### Model Configuration

```json
{
  "t5-small": {
    "use_optimized": true,
    "legacy_tokenizer": false,
    "use_seq2seqlm": true,
    "dimension": 512,
    "max_length": 512
  }
}
```

### Performance Features

- **LRU Model Cache**: Intelligent model caching with TTL
- **Thread-Safe Operations**: Concurrent request handling
- **Memory Management**: Automatic cleanup and resource monitoring
- **Batch Optimization**: Efficient batch processing for high throughput

---

## 🔌 API Reference

### API Versioning

Flouds AI uses API versioning with the `/api/v1` prefix for all endpoints:

- **Current Version**: v1
- **Base URL**: `http://localhost:19690/api/v1`
- **Documentation**: `/api/v1/docs`
- **OpenAPI Spec**: `/api/v1/openapi.json`

### Core Endpoints

| Endpoint | Method | Description | Rate Limited |
|----------|--------|-------------|-------------|
| `/api/v1/summarize` | POST | Single text summarization | ✅ |
| `/api/v1/summarize_batch` | POST | Batch text summarization | ✅ |
| `/api/v1/embed` | POST | Single text embedding | ✅ |
| `/api/v1/embed_batch` | POST | Batch text embedding | ✅ |
| `/api/v1/health` | GET | Basic health check | ❌ |
| `/api/v1/health/detailed` | GET | Detailed system info | ❌ |
| `/api/v1/health/ready` | GET | Readiness probe | ❌ |
| `/api/v1/health/live` | GET | Liveness probe | ❌ |
| `/api/v1/docs` | GET | Interactive API docs | ❌ |

### Response Headers

- `X-Processing-Time`: Request processing time in seconds
- `X-RateLimit-Remaining-Minute`: Remaining requests this minute
- `X-RateLimit-Remaining-Hour`: Remaining requests this hour

---

## 🧪 Testing & Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_embedder_service.py -v
```

### Code Quality

```bash
# Format code
black app/ tests/
isort app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
```

### Development Tools

- **Pre-commit hooks**: Automatic code formatting and linting
- **Comprehensive test suite**: Unit tests with mocking
- **Performance profiling**: Built-in performance monitoring
- **Hot reload**: Development server with auto-reload

---

## 🤝 Contributing

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/Flouds.Py.git
   cd Flouds.Py
   ```

2. **Setup Development Environment**
   ```bash
   pip install -r requirements-dev.txt  # Includes production + dev dependencies
   pre-commit install  # Install pre-commit hooks
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-amazing-feature
   ```

4. **Make Changes & Test**
   ```bash
   pytest  # Run tests
   black app/ tests/  # Format code
   ```

5. **Submit Pull Request**
   - Ensure all tests pass
   - Add tests for new features
   - Update documentation
   - Follow conventional commit messages

---

## 📈 Performance Benchmarks

### Typical Performance

| Operation | Model | Throughput | Latency |
|-----------|-------|------------|----------|
| Summarization | t5-small | 50 req/min | 1.2s |
| Embedding | all-MiniLM-L6-v2 | 200 req/min | 0.3s |
| Batch Summarization (10x) | t5-small | 150 req/min | 4.0s |
| Batch Embedding (10x) | all-MiniLM-L6-v2 | 500 req/min | 1.2s |

*Benchmarks on Intel i7-10700K, 32GB RAM, CPU-only*

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model ecosystem
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [Uvicorn](https://www.uvicorn.org/) - ASGI server

---

## 👨‍💻 Maintainer

**Goutam Malakar**  
📧 Contact: [Create an issue](https://github.com/your-repo/issues) for support

---

*Built with ❤️ for the AI community*