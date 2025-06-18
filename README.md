![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)
![LLaMA](https://img.shields.io/badge/LLaMA-3.1--8B-green)
![LoRA](https://img.shields.io/badge/Fine--tuning-LoRA-purple)
![Gradio](https://img.shields.io/badge/Interface-Gradio-ff7c00)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

![DPO Trained](https://img.shields.io/badge/DPO-Fine--Tuned-4B1D95?style=flat-square&logo=OpenAI&logoColor=white)
> 🎓 **Reinforced via Direct Preference Optimization** on 127 prompt triplets to align responses with my personal tone, accuracy, and answer style.

# RezAi - Personal AI Interview Twin (Fine-tuned LLM + DPO)
[![Try RezAi](https://img.shields.io/badge/Try%20RezAI-Live%20Demo-blue?style=for-the-badge)](https://rezaenayati.co/)
**RezAi** is a fine-tuned LLaMA 3.1 8B model trained on over 140 authentic Q&A pairs that reflect my technical background, personality, and lived experiences. Think of it as my digital twin, a personalized LLM model that knows how I think, code, and communicate.
Whether you’re curious about my projects, want to explore my approach to machine learning, or just need a breakdown of my favorite tech stacks and design decisions, RezAI can speak on my behalf. It doesn’t just answer like me, it is me, in model form.

## 🎯 Demo
RezAI is live and ready to chat! You can test it in two places:

- 🤖 [**Hugging Face Spaces (Gradio)**](https://huggingface.co/spaces/rezaenayati/RezAi) – Public and easy to access.
- 🌐 [**My Personal Website**](https://rezaenayati.co/) – **Recommended** for the smoothest and most polished UI experience.
  
## Features
- **🗣️ Authentic Voice**: Mimics my speaking style and personality
- **💻 Technical Depth**: Discusses iOS development, ML, and NLP with clarity
- **🌍 Personal Insights**: Reflects my journey growing up between Iran and LA
- **📈 Real Experience**: Talks about past projects like DJAI, Pizza Guys, RezAI Pro
- **🧠 Multi-turn Memory**: Maintains context across longer conversations
- **🧾 Meta Prompting**: Guided by a system message that defines tone, behavior, and ethics — ensures responses always sound like *me*, especially in interview scenarios.

## 🛠️ Technical Implementation

### 🧠 Model Architecture
- **Base Model**: Meta LLaMA 3.1-8B Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation) 
- **Training Framework**: Unsloth for 2x faster training
- **Dataset**: 140+ personal Q&A pairs in ChatML format
- **Final Loss**: 0.3-0.5 (optimized for personality capture)

### ⚙️ Training Process
1. **Dataset Creation**: Manually crafted responses to technical, behavioral, and personal questions
2. **Data Preprocessing**: Converted to ChatML format for conversation training
3. **Fine-tuning**: LoRA training with carefully tuned hyperparameters
4. **Evaluation**: Tested for authenticity, accuracy, and hallucination prevention. In addition to BERTScore and Embedding Similarity.

### Deployment
- **Platform**: HuggingFace Spaces with ZeroGPU
- **Interface**: Gradio & personal website

# 🔁 DPO 
To take RezAI beyond basic instruction-following, I implemented a DPO fine-tuning stage focused on aligning the model’s responses with my personal preferences.

## 💡 Why DPO?
While initial fine-tuning captured my tone and knowledge, it scored low on BERTScore F1 (around 6.5) to make the model further capture my tone and knowledge and to avoid generic or incossistent answers, I did DPO to further make it authentic not just correct. 

## 🛠️ How it worked
1. **Triplet Generation**
    * For each of the original 127 questions, I generated 3 diverse responses at different temperatures (0.3, 0.6, 0.9).
2. **Automated Ranking with BERTScore**
    * Each candidate was scored against my original hand-written answer. The best and worst responses were selected to form (prompt, chosen, rejected) triplets.
3. **DPO Training**
    * Using Hugging Face’s trl.DPOTrainer, I fine-tuned the LoRA adapter to prefer the “chosen” completions over the “rejected” ones, without needing a separate reward model.
4. **Outcome**
    * RezAI now more consistently reflects my personality, values, and answer style, especially in edge cases or open-ended prompts.

## 👤 Who Should Try RezAi?
- *Recruiters* curious about my work and thinking process
- Collaborators interested in my approach to software and AI
- Anyone building personal LLMs or fine-tuned digital twins
- The idea for this project is my syartup. 

