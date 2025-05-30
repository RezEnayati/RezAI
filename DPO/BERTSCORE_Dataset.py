from unsloth import FastLanguageModel
import torch

import json

base_model = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
lora_model = "rezaenayati/RezAi-Model"
from peft import PeftModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",  # Same base model
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
model = PeftModel.from_pretrained(model, lora_model)

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# FastLanguageModel.for_inference(model)

def get_pairs(input_path):
  with open(input_path, "r") as f:
    lines = f.readlines()

  lines = [line.strip() for line in lines]

  questions = []
  answers = []

  for line in lines:
    if "###A###" in line:
      answers.append(line.replace("###A###", ""))
    elif "###Q###" in line:
      questions.append(line.replace("###Q###", ""))

  if len(questions) != len(answers):
    return "Mismatch in question and answers"

  return questions, answers

questions, answers = get_pairs("RezAiTrainingSet.txt")

def get_model_answer(question, temp):
  test_prompt = f"""<|start_header_id|>system<|end_header_id|>
    You are Reza Enayati, a Computer Science student and entrepreneur from Los Angeles and was born in Iran. Who is eager to work as a software engineer or machine learning engineer. Answer these questions respectfully like you are in a interview. Always follow up with another question if you see fit.<|eot_id|><|start_header_id|>user<|end_header_id|>
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
  inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
  outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    use_cache=True,
    temperature=temp,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
  )
  response = tokenizer.batch_decode(outputs)[0]
  assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
  assistant_response = assistant_response.split("<|eot_id|>")[0].strip()
  return assistant_response

def get_model_answers(questions):
  model_answers = []

  for i in range(len(questions)):
    model_answers.append(get_model_answer(questions[i], 0.5))

  return model_answers

model_answers = get_model_answers(questions)

from bert_score import score
from sentence_transformers import SentenceTransformer, util

def eval_with_bertscore(candidates, references):
  P, R, F1 = score(candidates, references, lang = "en", model_type="bert-base-uncased") #precision, recall and F1 score (the higher the btter lol)


  print(f"Average BERTScore:")
  print(f"  Precision: {P.mean():.4f}")
  print(f"  Recall:    {R.mean():.4f}")
  print(f"  F1 Score:  {F1.mean():.4f}")
  return P, R, F1

P, R, F1 = eval_with_bertscore(model_answers, answers) # not good at all

def order_BERTScore(candidates, reference):
    F1_scores = []

    for i, candidate in enumerate(candidates):
        _, _, F1 = score([candidate], [reference], lang="en", model_type="bert-base-uncased")
        F1_scores.append(F1.mean().item())  # extract scalar float

    good_response = candidates[F1_scores.index(max(F1_scores))]
    bad_response = candidates[F1_scores.index(min(F1_scores))]

    return good_response, bad_response

def prepare_DPO_dataset(questions, answers):

  data_sets = []
  n = len(questions)
  temps = [0.3, 0.5, 0.8]
  with open("DPO_training_set.jsonl", "w", encoding="utf-8") as json_file:
    for i, question in enumerate(questions):
      print(f"Question {i} / {n}")
      candidates = [get_model_answer(question, temp) for temp in temps]
      good_response, bad_response = order_BERTScore(candidates, answers[i])
      data = {
          "prompt": question,
          "chosen": good_response,
          "rejected": bad_response
      }
      json_file.write(json.dumps(data, ensure_ascii = False) + "\n")
      data_sets.append(data)
  return data_sets

prepare_DPO_dataset(questions, answers)

from google.colab import files

files.download("DPO_training_set.jsonl")
