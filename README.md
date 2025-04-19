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
- Visual Studio Code (recommended for development)

### Step 1: Clone the Repository

```bash
git clone https://github.com/julien-nesbey/nextstep_interview.git
cd nextstep_interview
```

### Step 2: Set Up VS Code

It's recommended to set up VS Code before proceeding with the virtual environment creation:

1. Open the project in VS Code:

   ```bash
   code .
   ```

2. Install the Python extension if you haven't already:
   - Click on the Extensions icon in the sidebar (or press `Ctrl+Shift+X`)
   - Search for "Python"
   - Install the official Python extension by Microsoft

### Step 3: Create a Virtual Environment

Now create a virtual environment in your project directory:

#### On Windows

```bash
python -m venv .venv
```

#### On macOS/Linux

```bash
python3 -m venv .venv
```

### Step 4: Select the Python Interpreter in VS Code

Before activating the environment, it's important to select the correct Python interpreter in VS Code:

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS) to open the Command Palette
2. Type "Python: Select Interpreter" and select it
3. Choose the interpreter with `.venv` in its path:
   - Windows: Something like `Python 3.12.x ('.venv':venv)`
   - macOS/Linux: Something like `Python 3.12.x ('.venv'/bin/python)`

Alternatively, you can click on the Python version display in the bottom status bar and select the appropriate interpreter.

![VS Code Python Interpreter Selection](https://code.visualstudio.com/assets/docs/python/environments/interpreters-list.png)

#### Why Using the Correct Interpreter Matters

- Ensures all imported packages come from your isolated environment
- Prevents dependency conflicts between projects
- Enables proper IntelliSense and code navigation for project dependencies
- Allows debugging with the correct Python version and packages

### Step 5: Activate the Virtual Environment

Now activate the virtual environment in your terminal:

#### On Windows

```bash
.venv\Scripts\activate
```

#### On macOS/Linux

```bash
source .venv/bin/activate
```

You should see the virtual environment name in your terminal prompt, indicating it's active.

### Step 6: Install Dependencies

With the virtual environment activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the root directory with the following content:

```
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
docker run -p 5000:5000 -e LLM_API_KEY=your_key nextstep-interview
```

## 📂 Project Structure

- `app.py`: Main application file with Flask and SocketIO setup
- `models/`: Contains AI models for emotion detection and face recognition
- `prompts/`: Template prompts for AI question and analysis generation
- `utils/`: Utility functions for interview processing and data management
- `.env`: Environment variables for API keys and configuration
- `requirements.txt`: Python dependencies
- `graphQL/`: GraphQL schema and resolvers (if applicable)
- `common/`: Shared utilities and helper functions

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

### Common Development Tasks

- Testing the emotion detection: Run the app and connect to it via browser
- Debugging question generation: Check the app logs and prompts directory
- Modifying interview flow: Edit relevant sections in app.py
- Adding new model capabilities: Place models in the models directory and update imports

## 🔍 Troubleshooting

### Common Issues

- **"ModuleNotFoundError"**: Ensure you've activated the virtual environment and installed all dependencies
- **Model loading errors**: Check if all required model files exist in the models directory
- **API key errors**: Verify your .env file contains all necessary API keys
- **Socket connection issues**: Make sure no other application is using port 5000

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Socket.IO](https://socket.io/) for real-time communication
- [Together AI](https://www.together.ai/) for LLM services
- [Eleven Labs](https://elevenlabs.io/) for text-to-speech capabilities
- [OpenCV](https://opencv.org/) for computer vision functionality
- [Transformers](https://huggingface.co/docs/transformers/index) for emotion analysis models
