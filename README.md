# HomeMatch - Personalized Real Estate Agent

A RAG (Retrieval-Augmented Generation) system that provides personalized real estate recommendations using Large Language Models and vector databases.

## Overview

HomeMatch transforms standard real estate listings into personalized narratives that resonate with potential buyers' unique preferences and needs. The system leverages:

- **FastAPI** for the web framework and API
- **ChromaDB** for vector database and semantic search
- **OpenAI API** for language model integration
- **Jinja2** for HTML templating

## Features

- 🤖 **AI-Powered Understanding**: Natural language processing of user preferences
- 🔍 **Semantic Search**: Vector-based matching of properties to user needs
- ✨ **Personalized Descriptions**: Custom listing descriptions based on user preferences
- 🌐 **Web Interface**: Simple and intuitive user interface
- 📊 **RESTful API**: Well-documented API endpoints

## Project Structure

```
homewatch_project/
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
├── app.py                  # Main FastAPI application
├── implementation_plan.md   # Detailed implementation plan
├── prompt_file.md          # System prompt and requirements
├── database/               # Vector database storage
├── modules/                # Core application modules (to be implemented)
├── templates/              # HTML templates
│   └── index.html         # Main web interface
├── static/                 # Static files (CSS, JS)
│   └── style.css          # Application styles
└── README.md              # This file
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv homewatch_env
source homewatch_env/bin/activate  # On Windows: homewatch_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your OpenAI API key
# OPENAI_BASE_URL=https://openai.vocareum.com/v1
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the Application

```bash
# Start the development server
python app.py

# Or using uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

- `GET /` - Main web interface
- `GET /health` - Health check endpoint
- `GET /api/listings` - Get all available listings (to be implemented)
- `POST /api/search` - Search listings based on preferences (to be implemented)
- `POST /api/personalize` - Personalize listing descriptions (to be implemented)

## Implementation Status

### ✅ Phase 1: Project Foundation (COMPLETED)
- [x] Environment setup with virtual environment
- [x] Dependency management and installation
- [x] Git repository initialization
- [x] Basic FastAPI application structure
- [x] Web interface with HTML templates and CSS
- [x] Project documentation

### 🚧 Phase 2: Core Data Layer (IN PROGRESS)
- [ ] Synthetic data generation module
- [ ] Vector database implementation
- [ ] Listing storage and retrieval

### ⏳ Phase 3: Search and Personalization Engine (PENDING)
- [ ] User preference processing
- [ ] Semantic search implementation
- [ ] Personalization module

### ⏳ Phase 4: API and Web Interface Enhancement (PENDING)
- [ ] Complete API endpoint implementation
- [ ] Enhanced web interface
- [ ] Error handling and validation

### ⏳ Phase 5: Integration and Testing (PENDING)
- [ ] System integration
- [ ] Testing and quality assurance
- [ ] Performance optimization

### ⏳ Phase 6: Documentation and Deployment (PENDING)
- [ ] Complete documentation
- [ ] Deployment preparation
- [ ] Final testing and validation

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code formatting
- Use type hints where applicable
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Git Workflow
```bash
# Add and commit changes
git add .
git commit -m "feat: implement user preference processing"

# Push to repository
git push origin main
```

### Testing
```bash
# Run tests (when implemented)
pytest

# Run with coverage
pytest --cov=.
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_BASE_URL` | OpenAI API base URL | https://openai.vocareum.com/v1 | Yes |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `APP_NAME` | Application name | HomeMatch | No |
| `APP_VERSION` | Application version | 1.0.0 | No |
| `DEBUG` | Debug mode | True | No |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | ./database/chroma_db | No |

## Dependencies

### Core Dependencies
- `fastapi==0.104.1` - Web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `openai>=1.10.0` - OpenAI API client
- `chromadb==0.4.18` - Vector database
- `python-dotenv==1.0.0` - Environment variable management

### Development Dependencies
- `pytest==7.4.3` - Testing framework
- `pytest-asyncio==0.21.1` - Async testing support

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure virtual environment is activated
2. **Port already in use**: Change port in `app.py` or kill existing process
3. **OpenAI API errors**: Verify API key is set correctly in `.env`

### Logs and Debugging

```bash
# Enable debug logging
export LOG_LEVEL=debug
python app.py

# Check application logs
tail -f logs/app.log  # When logging is implemented
```

## Contributing

1. Follow the implementation plan in `implementation_plan.md`
2. Create feature branches for new functionality
3. Write tests for new features
4. Update documentation as needed
5. Submit pull requests for review

## License

This project is for educational purposes as part of the "Building Generative AI Solutions" course.

## Contact

For questions about this implementation, refer to the course materials or implementation plan.
