![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)
![LLaMA](https://img.shields.io/badge/LLaMA-3.1--8B-green)
![LoRA](https://img.shields.io/badge/Fine--tuning-LoRA-purple)
![Gradio](https://img.shields.io/badge/Interface-Gradio-ff7c00)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

[![Try RezAI](https://img.shields.io/badge/Try%20RezAI-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/rezaenayati/RezAi)

# 🚀 RezAI - Personal AI Interview Twin 
**RezAI** is a fine-tuned LLaMA 3.1 8B model trained on over 140 authentic Q&A pairs that reflect my technical background, personality, and lived experiences. Think of it as my digital twin, a personalized LLM model that knows how I think, code, and communicate.
Whether you’re curious about my projects, want to explore my approach to machine learning, or just need a breakdown of my favorite tech stacks and design decisions, RezAI can speak on my behalf. It doesn’t just answer like me, it is me, in model form.

## 🎯 Demo
RezAI is live and ready to chat! You can test it in two places:

- 🤖 [**Hugging Face Spaces (Gradio)**](https://huggingface.co/spaces/rezaenayati/RezAi) – Public and easy to access.
- 🌐 [**My Personal Website**](https://rezaenayati.co/) – **Recommended** for the smoothest and most polished UI experience (seriously, it looks way better here).
  
## Features
- **Authentic Voice**: Responds in my natural speaking style and tone
- **Technical Depth**: Discusses iOS development, machine learning, and AI research
- **Personal Insights**: Shares background growing up between Iran and the US
- **Real Experience**: Details about Pizza Guys app, DJAI internship, and research projects
- **Conversational**: Maintains context across multi-turn conversations

## 🛠️ Technical Implementation

### 🧠 Model Architecture
- **Base Model**: Meta LLaMA 3.1-8B Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation) 
- **Training Framework**: Unsloth for 2x faster training
- **Dataset**: 140+ personal Q&A pairs in ChatML format
- **Final Loss**: 0.3-0.4 (optimized for personality capture)

### ⚙️ Training Process
1. **Dataset Creation**: Manually crafted responses to technical, behavioral, and personal questions
2. **Data Preprocessing**: Converted to ChatML format for conversation training
3. **Fine-tuning**: LoRA training with carefully tuned hyperparameters
4. **Evaluation**: Tested for authenticity, accuracy, and hallucination prevention. In addition to BERTScore and Embedding Similarity.

### Deployment
- **Platform**: HuggingFace Spaces with ZeroGPU
- **Interface**: Gradio & personal website

## Key Learning Insights

- **Overfitting Detection**: Model initially hallucinated fake details ("Ari from Irvine")
- **Loss vs. Quality**: Lower loss doesn't always mean better responses
- **Temperature Tuning**: Found optimal balance between creativity and accuracy
- **Dataset Quality**: 140 high-quality examples > 1000+ generic ones
