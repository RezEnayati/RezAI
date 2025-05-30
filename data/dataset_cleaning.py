import json

def clean_prepare_chatMl_save_json(input_path, output_path):
  with open(input_path, "r") as f:
    lines =  f.readlines()
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

  chatML = []

  n = len(questions)

  for i in range(n):
    chatML.append({
          "messages": [
              {
                  "role": "system",
                  "content": "You are Reza Enayati, a Computer Science student and entrepreneur from Los Angeles, Who is eager to work as a software engineer or a machine learning engineer, answer these questions as if you are in an interview."
              },
              {
                  "role": "user",
                  "content": questions[i]
              },
              {
                  "role": "assistant",
                  "content": answers[i]
              }
          ]
    })

    with open(output_path, "w", encoding = "utf-8") as json_file:
      json.dump(chatML, json_file, indent = 2, ensure_ascii=False)

clean_prepare_chatMl_save_json("RezAiTrainingSet.txt", "test_json")
