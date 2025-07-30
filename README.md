# Question Rephraser Live

This is the **live version** of the question-rephraser project, specifically designed for deployment on Streamlit Cloud. The main difference from the original project is the removal of the `sentencepiece` dependency, which can cause compatibility issues with Streamlit Cloud.

**🌐 Live Demo**: [https://question-rephraser.streamlit.app/](https://question-rephraser.streamlit.app/)

**📁 Original Project**: [https://github.com/salemhamdani/question-rephraser](https://github.com/salemhamdani/question-rephraser)

## Overview

A Streamlit web application that provides a chat-like interface for testing trained question rephrasing models.

## Key Features

- **Model Selection**: Choose from various pre-trained models including BART only for now.
- **Real-time Rephrasing**: Input disfluent questions and get instant clean versions
- **Multiple Model Support**: Works with both Hugging Face Hub models and locally trained models

## Why This Version?

The original question-rephraser project includes `sentencepiece>=0.1.97` in its dependencies, which can cause deployment issues on Streamlit Cloud. This live version removes that dependency while maintaining all core functionality.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd question-rephraser-live
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Features

- **Model Loading**: Automatically discovers and loads available models from Hugging Face Hub
- **Local Model Support**: Can load models from local `experiments/` directory
- **Real-time Processing**: Instant question rephrasing with progress indicators
- **Error Handling**: Graceful fallback when models are unavailable
- **Responsive UI**: Clean, modern interface optimized for web deployment



## Model Compatibility

The app supports:
- **BART models** (recommended for question rephrasing)
- **Local trained models** (from experiments folder)

## Contributing

This is the live deployment version of the main question-rephraser project. For core development, please refer to the [original question-rephraser repository](https://github.com/salemhamdani/question-rephraser).

## License

Same as the original question-rephraser project.