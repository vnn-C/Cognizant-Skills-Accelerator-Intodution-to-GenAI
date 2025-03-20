import os
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#variables
model = load_model("LSTM/lstm_model.h5")
source_folder = "LSTM\gutenberg_texts"
seq_len = 50
dest_folder = "LSTM\output"
#files_to_read in lstm_main.ipynb and generate_story.py should have the same value
files_to_read = 10
temperature = 0.7
max_words = 5000


#generate tokenizer
data_list = []
count = 0
for file_name in os.listdir(source_folder):
    if count == files_to_read:
        break
    else:
        count+=1
    try:
        file_path = os.path.join(source_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read().lower()
        data_list.append(data)

    except Exception as e:
        print(f"Error with reading {file_name}: {e}")

data_str = " ".join(data_list)
tokenizer = Tokenizer(char_level=False, filters="!\"#$%&()*+-/:;<=>?@[\\]^_`{|}~", num_words=max_words)
#tokenizer = Tokenizer(char_level=False, num_words=max_words)

tokenizer.fit_on_texts([data_str])
sequences = tokenizer.texts_to_sequences([data_str])[0]

vocab_size = len(tokenizer.word_index) + 1
vocab = list(tokenizer.word_index.keys())


#generate story
story_len = 0
#while loop for starting input
while True:
    start_text = input("How do you want to start your story?")
    if isinstance(start_text, str):
        break
    elif len(list(start_text.split())) > 50:
        print("The beginning of your story should be 50 words or less")
    else:
        print("Enter a valid beginning")

#while loop for story length
while True:
    len_choice = input("How many words do you want in your story?")
    if not len_choice.isdigit():
        print("Please enter a valid number")
    elif int(len_choice) < 0:
        print("Please enter a number greater than 0")
    else:
        story_len = int(len_choice)
        break

print("Proceeding to your story...")

generated_story = start_text

#preprocessing starting input
tokenized_start = tokenizer.texts_to_sequences([start_text.lower()])

if not tokenized_start or len(tokenized_start[0]) == 0:
    print("No valid tokens in start text.")
else:
    start_sequence = pad_sequences(tokenized_start, maxlen=seq_len, padding="pre")

#start_sequence = pad_sequences(tokenized_start, maxlen=seq_len, padding="pre")

#try statement for generating text
for i in range(story_len):
    try:
        pred_text = model.predict(start_sequence, verbose=0)

#softmax operation to implement temperature
        preds = np.asarray(pred_text).astype("float64")
        preds = np.log(preds + 1e-7) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)



        pred_text_index = np.random.choice(len(preds[0]), p=preds[0])


        pred_word = tokenizer.index_word.get(pred_text_index, "Unknown")
        generated_story+=" " + pred_word
        if i % 20 == 0:
            generated_story+="\n"

        start_sequence = pad_sequences([start_sequence[0].tolist() + [pred_text_index]], maxlen=seq_len, padding="pre")
    except Exception as e:
        print(f"Error with generating text: {e}")

#print(generated_story)
print("Saving generated story...")
curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")

new_file_path = os.path.join(dest_folder, f"story_{curr_date}.txt")

try:
    new_file = open(new_file_path, "x")
    new_file.write(generated_story)
    new_file.close()
    
except Exception as e:
    print(f"Error with creating story file: {e}")
