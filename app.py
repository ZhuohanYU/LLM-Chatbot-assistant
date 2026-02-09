from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
import os
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
import pandas as pd
from io import BytesIO

# Disable SSL verification for Hugging Face downloads (workaround for corporate/proxy environments)
# WARNING: This reduces security but is necessary when SSL certificates can't be verified
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure environment to disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Monkey patch requests to disable SSL verification globally for Hugging Face
_original_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return _original_request(self, method, url, **kwargs)

requests.Session.request = _patched_request

# Initialize Flask app
app = Flask(__name__)
# Note: SocketIO is imported but we're using regular Flask for simplicity
# socketio = SocketIO(app)  # Commented out - not needed for basic HTTP requests

# Model configuration
# Using smaller 125M model to reduce memory requirements (~500MB vs 5.31GB)
model_name = "EleutherAI/gpt-neo-125M"
model = None
tokenizer = None
device = None
model_loading = False
model_loaded = False

# Memory to store conversation history with timestamps
conversation_memory = []
# Chat history for Excel export (with timestamps)
chat_history_records = []

def load_model():
    """Load the model and tokenizer in the background"""
    global model, tokenizer, device, model_loading, model_loaded
    
    if model_loading or model_loaded:
        return
    
    model_loading = True
    
    # Device detection: CUDA (NVIDIA GPU) > MPS (Mac) > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    print("Loading model and tokenizer... (Using smaller 125M model - ~500MB)")
    print("This should work with limited memory!")
    
    try:
        # Load tokenizer first (smaller memory footprint)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Set pad token if not present (some models don't have one)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")
        
        # Memory-efficient model loading
        # Use low_cpu_mem_usage to reduce peak memory usage
        # Load directly to CPU first to avoid memory spikes
        print("Loading model with memory-efficient settings...")
        
        # Use memory-efficient loading
        # For CPU systems, we need to be extra careful with memory
        loading_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if device == "cuda":
            # For GPU, use float16 to reduce memory by half
            loading_kwargs["torch_dtype"] = torch.float16
            loading_kwargs["device_map"] = "auto"
        else:
            # For CPU, use float32 but load more carefully
            loading_kwargs["torch_dtype"] = torch.float32
        
        print("Loading model (this may take a while due to memory constraints)...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **loading_kwargs)
        
        # If not using device_map, move to device manually
        if device != "cuda" or "device_map" not in loading_kwargs:
            print("Moving model to device...")
            model = model.to(device)
        
        # Enable evaluation mode to reduce memory
        model.eval()
        
        # Clear any cached memory
        if device == "cpu":
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model_loading = False
        print("\nTroubleshooting tips:")
        print("1. Close other applications to free up RAM")
        print("2. Make sure you have at least 2GB free RAM available")
        print("3. If still failing, the system may need more available memory")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/export_excel")
def export_excel():
    """Export chat history to Excel file"""
    if not chat_history_records:
        return jsonify({"error": "No chat history to export"}), 404
    
    try:
        # Create DataFrame from chat history
        df = pd.DataFrame(chat_history_records)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Chat History')
            
            # Get the workbook and worksheet to format
            worksheet = writer.sheets['Chat History']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 100)  # Cap at 100
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        
        # Generate filename with timestamp
        filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": f"Failed to export Excel: {str(e)}"}), 500

@app.route("/chat_history")
def get_chat_history():
    """Get chat history as JSON"""
    return jsonify({
        "total_records": len(chat_history_records),
        "history": chat_history_records
    })

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear chat history"""
    global conversation_memory, chat_history_records
    conversation_memory = []
    chat_history_records = []
    return jsonify({"message": "Chat history cleared successfully"})

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_memory, model_loaded, model_loading, chat_history_records

    # Check if model is loaded
    if not model_loaded:
        if not model_loading:
            # Start loading the model in a background thread
            thread = threading.Thread(target=load_model)
            thread.daemon = True
            thread.start()
        
        return jsonify({
            "response": "Model is still loading. Please wait a moment and try again. This may take several minutes on the first run."
        }), 503

    # Get user input
    user_message = request.json["message"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append user message to memory
    conversation_memory.append({"role": "user", "content": user_message})

    # Generate response with better formatting
    # Build context with proper formatting for the model
    context = "The following is a conversation between a user and an AI assistant.\n\n"
    for msg in conversation_memory[-5:]:  # Only use last 5 messages to avoid context overflow
        if msg['role'] == 'user':
            context += f"User: {msg['content']}\n"
        else:
            context += f"Assistant: {msg['content']}\n"
    context += "Assistant:"
    
    # Prepare input IDs for the model
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    
    # Limit input length to prevent issues
    max_input_length = 512
    if input_ids.shape[1] > max_input_length:
        # Keep only the most recent tokens
        input_ids = input_ids[:, -max_input_length:]

    # Fix for MPS compatibility (force int32 for attention_mask)
    if device == "mps":
        attention_mask = torch.ones(input_ids.shape, dtype=torch.int32, device=device)
    else:
        attention_mask = torch.ones(input_ids.shape, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,  # Limit new tokens generated (not total length)
            min_length=input_ids.shape[1] + 10,  # Minimum response length
            temperature=0.7,  # Lower temperature for more coherent responses
            do_sample=True,  # Enable sampling
            top_p=0.9,  # Nucleus sampling for better quality
            top_k=50,  # Top-k sampling
            repetition_penalty=1.2,  # Penalize repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
        )
    
    # Decode only the new tokens (response part)
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Clean up the response
    response = response.strip()
    
    # Stop at newlines or if response seems to repeat
    if '\n' in response:
        response = response.split('\n')[0].strip()
    
    # Limit response length
    if len(response) > 500:
        response = response[:500].rsplit(' ', 1)[0] + "..."

    # Append assistant response to memory
    conversation_memory.append({"role": "assistant", "content": response})
    
    # Save to chat history records for Excel export
    chat_history_records.append({
        "Timestamp": timestamp,
        "Question": user_message,
        "Answer": response
    })

    return jsonify({"response": response})

if __name__ == "__main__":
    # Start loading the model in the background
    thread = threading.Thread(target=load_model)
    thread.daemon = True
    thread.start()
    
    # Try different ports if 5000 is in use
    port = 5000
    host = '127.0.0.1'  # localhost
    
    print("Starting Flask server...")
    print(f"Server will be available at http://{host}:{port}")
    print("If port 5000 is in use, try http://localhost:5001 or http://localhost:8080")
    print("Model is loading in the background. You can access the interface now.")
    
    try:
        # Use regular Flask run instead of SocketIO
        app.run(debug=True, host=host, port=port, threaded=True)
    except OSError as e:
        if "Address already in use" in str(e) or "address is already in use" in str(e).lower() or "10048" in str(e):
            print(f"\nPort {port} is already in use. Trying port 5001...")
            try:
                app.run(debug=True, host=host, port=5001, threaded=True)
            except OSError:
                print(f"\nPort 5001 also in use. Trying port 8080...")
                app.run(debug=True, host=host, port=8080, threaded=True)
        else:
            print(f"\nError starting server: {e}")
            import traceback
            traceback.print_exc()
            raise
