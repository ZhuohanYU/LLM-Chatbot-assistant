# Fixing "Paging File Too Small" Error

If you're getting the error "The paging file is too small for this operation to complete", you need to increase Windows virtual memory (paging file).

## Quick Fix: Increase Windows Paging File

### Method 1: Automatic (Recommended)

1. **Open System Properties:**
   - Press `Win + Pause/Break` OR
   - Right-click "This PC" → Properties

2. **Go to Advanced System Settings:**
   - Click "Advanced system settings" on the left

3. **Open Performance Settings:**
   - Under "Performance", click "Settings..."

4. **Open Virtual Memory:**
   - Click the "Advanced" tab
   - Under "Virtual memory", click "Change..."

5. **Configure Paging File:**
   - **Uncheck** "Automatically manage paging file size for all drives"
   - Select your system drive (usually C:)
   - Select "Custom size"
   - Set:
     - **Initial size (MB):** 16384 (16 GB)
     - **Maximum size (MB):** 32768 (32 GB)
   - Click "Set"
   - Click "OK"

6. **Restart your computer** for changes to take effect

### Method 2: Manual Calculation

If you want to calculate based on your RAM:
- **Minimum:** 1.5 × your RAM size
- **Maximum:** 3 × your RAM size

For example, if you have 8GB RAM:
- Minimum: 12,288 MB (12 GB)
- Maximum: 24,576 MB (24 GB)

## Alternative Solutions

### Option 1: Use a Smaller Model

If increasing the paging file doesn't work or you have limited disk space, consider using a smaller model. Edit `app.py` and change:

```python
model_name = "EleutherAI/gpt-neo-125M"  # Much smaller model
```

### Option 2: Close Other Applications

Before running the chatbot:
- Close unnecessary applications
- Close browser tabs
- Stop other Python processes
- Free up as much RAM as possible

### Option 3: Use 8-bit Quantization (Advanced)

This requires installing additional packages:
```powershell
pip install bitsandbytes
```

Then modify the model loading in `app.py` to use 8-bit quantization (reduces memory by ~75%).

## Verify the Fix

After increasing the paging file and restarting:
1. Run the app again: `python app.py`
2. The model should load without the paging file error

## Current Model Size

- **GPT-Neo-1.3B:** ~5.31 GB
- **Recommended RAM:** 8GB+ 
- **Recommended Paging File:** 16GB+ (for comfortable operation)


