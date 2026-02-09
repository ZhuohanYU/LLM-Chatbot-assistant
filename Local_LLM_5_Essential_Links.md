# Running Local LLM - 5 Essential Links

## 1. HuggingFace Transformers Quick Tour
**URL**: https://huggingface.co/docs/transformers/quicktour

**What it does**:
- Shows how to load and use any transformer model in 3 lines of code
- Demonstrates text generation, question answering, and chat completion
- Includes examples for DeepSeek, Llama, and other popular models
- Covers both GPU and CPU usage

**Key example**:
```python
from transformers import pipeline
chatbot = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
chatbot("Hello, how are you?")
```

---

## 2. HuggingFace Text Generation Pipeline Documentation
**URL**: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline

**What it does**:
- Complete API reference for text generation pipeline
- Explains all parameters: temperature, top_p, top_k, max_new_tokens
- Shows how to control generation behavior
- Includes batch processing and streaming examples

**Key example**:
```python
generator = pipeline('text-generation', model='gpt2')
generator("I am going to", max_length=30, num_return_sequences=3)
```

---

## 3. Google Colab - Running LLMs with Free GPU
**URL**: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_generation.ipynb

**What it does**:
- Interactive notebook with free GPU access
- Pre-configured environment, no installation needed
- Step-by-step walkthrough from loading model to generating text
- Can test different models (Llama, Qwen, DeepSeek) without local resources

**Key features**:
- Run immediately in browser
- Free T4 GPU for faster inference
- Save and share your own notebooks
- Experiment with different models and parameters

---

## 4. Ollama Documentation - Simplest Local Deployment
**URL**: https://github.com/ollama/ollama/blob/main/README.md

**What it does**:
- One-command installation on Mac, Linux, Windows
- Download and run models with single command: `ollama run deepseek-r1:1.5b`
- Built-in REST API for application integration
- Model library with optimized versions of popular LLMs

**Key examples**:
```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Run model
ollama run deepseek-r1:1.5b

# Use in Python
import requests
requests.post('http://localhost:11434/api/generate', 
    json={'model': 'deepseek-r1:1.5b', 'prompt': 'Hello'})
```

---

## 5. HuggingFace Model Hub - DeepSeek R1 Model Card
**URL**: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

**What it does**:
- Official model page with complete usage instructions
- Shows exact code to load and run the model
- Includes model specifications, performance benchmarks, and limitations
- Community discussions and example implementations in "Community" tab

**Key example**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

---

## Quick Comparison

| Link | Best For | Technical Level | Setup Time |
|------|----------|----------------|------------|
| HF Quick Tour | Learning basics | Beginner | 5 min |
| Pipeline Docs | Parameter tuning | Intermediate | N/A (reference) |
| Google Colab | Testing without local install | Beginner | Instant |
| Ollama | Production deployment | Beginner | 5 min |
| Model Card | Specific model details | Intermediate | N/A (reference) |

---

## Recommended Learning Path

**Path 1 - Fastest Start (15 minutes)**:
1. Google Colab link (run examples immediately)
2. Ollama link (install locally)

**Path 2 - Comprehensive (1 hour)**:
1. HuggingFace Quick Tour (understand fundamentals)
2. Pipeline Documentation (learn parameters)
3. Model Card (choose right model)
4. Google Colab (practice)
5. Ollama (deploy)

**Path 3 - Production Ready (1 day)**:
1. HuggingFace Quick Tour
2. Model Card (evaluate options)
3. Pipeline Documentation (optimize performance)
4. Ollama (production deployment)
5. Test and iterate

---

## Additional Context

**All links provide**:
- Working code examples
- Copy-paste ready snippets
- Free tools (no paid API required)
- Active community support
- Regular updates

**Prerequisites**:
- Python 3.8+ installed
- Basic command line knowledge
- 8GB+ RAM recommended
- GPU optional (CPU works but slower)


---

Last Updated: February 2026
