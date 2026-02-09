# Setup Guide for LLM Chatbot Assistant

This guide will help you set up and run the chatbot application on your local Windows machine.

## Prerequisites

1. **Python 3.8 or higher** - Make sure Python is installed on your system
   - Check by running: `python --version` or `python3 --version`
   - Download from [python.org](https://www.python.org/downloads/) if needed

2. **pip** - Python package manager (usually comes with Python)
   - Check by running: `pip --version`

## Installation Steps

### Step 1: Create a Virtual Environment (Recommended)

It's best practice to use a virtual environment to isolate your project dependencies:

```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

Install all required packages from the `requirements.txt` file:

```powershell
pip install -r requirements.txt
```

**Note:** This will download and install:
- Flask (web framework)
- Flask-SocketIO (real-time communication)
- Transformers (Hugging Face library for LLMs)
- PyTorch (deep learning framework)
- Additional dependencies

**Important:** The first time you run this, it will download the GPT-Neo-1.3B model (~5GB) from Hugging Face. This may take several minutes depending on your internet connection.

### Step 3: Run the Application

Start the Flask server:

```powershell
python app.py
```

You should see output like:
```
Loading model and tokenizer...
Using device: cpu
Model loaded.
 * Running on http://127.0.0.1:5000
```

### Step 4: Access the Chatbot

Open your web browser and navigate to:
```
http://localhost:5000
```

or

```
http://127.0.0.1:5000
```

## Troubleshooting

### Issue: Port already in use
If you see an error about the port being in use, you can change the port in `app.py`:
```python
socketio.run(app, debug=True, port=5001)
```

### Issue: Model download is slow
The GPT-Neo-1.3B model is approximately 5GB. The first run will download it automatically. Be patient, or ensure you have a stable internet connection.

### Issue: Out of memory errors
If you encounter memory issues:
- The model will run on CPU by default (slower but uses less memory)
- Consider using a smaller model or reducing `max_length` in the `model.generate()` call
- Close other applications to free up RAM

### Issue: CUDA/GPU not detected
The app will automatically use CPU if no GPU is available. For GPU support on Windows:
- Install CUDA toolkit if you have an NVIDIA GPU
- PyTorch will automatically detect and use CUDA if available

## Project Structure

```
LLM-Chatbot-assistant/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Chatbot web interface
├── README.md             # Project description
└── SETUP.md             # This file
```

## Stopping the Application

Press `Ctrl+C` in the terminal to stop the Flask server.

## Next Steps

- Customize the model by changing `model_name` in `app.py`
- Adjust response length by modifying `max_length` in the `model.generate()` call
- Modify the UI by editing `templates/index.html`

