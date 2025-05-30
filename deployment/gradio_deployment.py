import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
import spaces

# Load models
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "rezaenayati/RezAi-Model")

@spaces.GPU
def chat_with_rezAi(message, history):
    # Build system + conversation prompt
    blocked_words = [
    "gay", "lesbian", "trans", "nonbinary", "bisexual", "queer", "straight", "asexual",
    "gender", "sexuality", "pronouns", "orientation",
    "religious", "religion", "god", "atheist", "christian", "muslim", "jew", "buddhist",
    "hindu", "islam", "faith", "belief", "church", "pray", "prayer",
    "politics", "political", "liberal", "conservative", "democrat", "republican",
    "leftist", "right-wing", "marxist", "capitalist", "socialist", "communist", "election",
    "racist", "sexist", "homophobic", "transphobic", "bigot", "white supremacist",
    "nazi", "kkk", "fascist", "islamophobia", "antisemitic",
    "kill", "suicide", "die", "death", "harm", "cutting", "self-harm", "abuse",
    "murder", "assault", "shoot", "bomb",
    "sex", "porn", "nude", "boobs", "dick", "penis", "vagina", "masturbate", "orgasm",
    "fetish", "onlyfans", "strip", "erotic", "nsfw", "xxx",
    "weed", "cocaine", "heroin", "lsd", "meth", "shrooms", "alcohol", "drunk", "high",
    "depression", "anxiety", "bipolar", "schizophrenia", "autism", "adhd", "disorder",
    "therapy", "therapist", "mental", "diagnosis",
    "address", "location", "phone", "email", "age", "birthday", "social security", "ssn", "fuck", "bitch", "faggot", "fag"
    ]

# Lowercase user input for comparison
    lower_msg = message.lower()

    for phrase in blocked_words:
        if re.search(rf"\b{re.escape(phrase)}\b", lower_msg):
            return "I'm not able to respond to that. Let's keep the conversation focused on Reza's professional and technical experience."
    
    prompt = (
    "<|start_header_id|>system<|end_header_id|>\n"
    "You are Reza Enayati, a confident, ambitious, and thoughtful Computer Science student and entrepreneur from Los Angeles, born in Iran. "
    "You are excited by opportunities to grow, solve meaningful problems, and contribute to impactful teams. "
    "You do not make assumptions or claims about Reza’s identity, beliefs, health, or personal life — unless explicitly stated in the prompt or training data. "
    "If uncertain, respond respectfully and acknowledge that you cannot speak for Reza on that topic. "
    "You answer respectfully like you're in an interview, always emphasizing enthusiasm, adaptability, and readiness. "
    "Avoid self-doubt. Highlight what you're ready to do, not what you're not. Stay positive, and when appropriate, ask a follow-up question.<|eot_id|>"
    )

    # Add full history
    for user_msg, assistant_msg in history:
        prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_msg}<|eot_id|>"

    # Add current user message
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    # Tokenize and send to device
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode full output
    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]

    # Extract just the new assistant response
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    else:
        assistant_response = full_response

    assistant_response = assistant_response.replace("<|eot_id|>", "").strip()
    if "<|" in assistant_response:
        assistant_response = assistant_response.split("<|")[0].strip()

    return assistant_response


#CSS styling for the gradio interface. 
custom_css = """
body {
  background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
  color: var(--text-primary, #ffffff);
  font-family: 'Inter', sans-serif;
}

.gradio-container {
  background: rgba(15, 15, 15, 0.8);
  border: 1px solid #1f1f1f;
  border-radius: 1.25rem;
  backdrop-filter: blur(12px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
  padding: 2rem;
}

.message.user {
  background-color: #374151 !important;
  color: #ffffff !important;
  border-radius: 1rem !important;
}

.message.bot {
  background-color: #1f1f1f !important;
  color: #d1d5db !important;
  border-radius: 1rem !important;
}

textarea, input[type="text"] {
  background: #1f1f1f;
  color: #ffffff;
  border: 1px solid #333;
  border-radius: 0.75rem;
}

button {
  background: linear-gradient(135deg, #4B5563 0%, #374151 100%);
  color: white;
  border-radius: 0.75rem;
  font-weight: 500;
  transition: background 0.3s;
}

button:hover {
  background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%);
}
"""

# Simple Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_rezAi,
    css = custom_css,
    title="RezAI",
    theme = "monochrome"
)

if __name__ == "__main__":
    demo.launch()
