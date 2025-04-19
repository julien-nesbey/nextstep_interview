# NextStep Interview Assistant

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![SocketIO](https://img.shields.io/badge/SocketIO-5.5.1-red.svg)](https://socket.io/)

## 🚀 Overview

NextStep is an AI-powered interview assistant that helps users practice for job interviews with real-time question generation and emotion analysis. The application leverages various AI models to provide a comprehensive interview simulation experience.

## ✨ Features

- AI-generated interview questions based on selected job roles
- Real-time emotion analysis during interviews
- Speech-to-text and text-to-speech capabilities
- Detailed performance analytics
- Fluency scoring
- Interview summaries and feedback

## 🛠️ Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/julien-nesbey/nextstep_interview.git
cd nextstep_interview
```

### Step 2: Create a Virtual Environment

#### On Windows

```bash
python -m venv .venv
```

#### On macOS/Linux

```bash
python3 -m venv .venv
```

### Step 3: Activate the Virtual Environment

#### On Windows

```bash
.venv\Scripts\activate
```

#### On macOS/Linux

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the root directory with the following content:

```
ELEVEN_LABS_API_KEY=your_eleven_labs_key
ELEVEN_LABS_VOICE_ID=your_voice_id
LLM_MODEL=your_llm_model
LLM_API_KEY=your_llm_api_key
DEBUG_MODE=false
```

## 🚀 Running the Application

After installing dependencies and configuring environment variables:

```bash
python app.py
```

The application will be available at `http://localhost:5000`.

## 🐳 Docker Support

You can also run the application using Docker:

### Build the Docker Image

```bash
docker build -t nextstep-interview .
```

### Run the Docker Container

```bash
docker run -p 5000:5000 -e LLM_API_KEY=your_key -e ELEVEN_LABS_API_KEY=your_key nextstep-interview
```

## 📂 Project Structure

- `app.py`: Main application file
- `models/`: Contains AI models for emotion detection and face recognition
- `prompts/`: Template prompts for AI generation
- `utils/`: Utility functions
- `globals.py`: Global variables and configurations
- `.env`: Environment variables
- `requirements.txt`: Python dependencies

## 🧪 Development

### Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

### Updating Dependencies

If you need to add or update dependencies:

1. Install the new package:

   ```bash
   pip install package_name
   ```

2. Update the requirements file:

   ```bash
   pip freeze > requirements.txt
   ```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Socket.IO](https://socket.io/) for real-time communication
- [Together AI](https://www.together.ai/) for LLM services
- [Eleven Labs](https://elevenlabs.io/) for text-to-speech capabilities
- [OpenCV](https://opencv.org/) for computer vision functionality
