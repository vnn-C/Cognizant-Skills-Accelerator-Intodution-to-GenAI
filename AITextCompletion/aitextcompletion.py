#AI model used: https://huggingface.co/openai-community/gpt2
from transformers import pipeline


print("Starting script")

text_result = pipeline("text-generation", model="gpt2", max_length=300, temperature=0.8)

while True:
    try:
        prompt = input("Enter a prompt, or enter x to exit.")
        if len(prompt) > 100:
            print("Please enter a prompt shorter than 100 words.")
            continue

        elif prompt == "x":
            break
        elif not prompt.strip():
            print("Please enter a valid prompt.")
            continue
        else:
            result = text_result(prompt)[0]['generated_text']
            print(result)

    except Exception as e:
        print("Error: " + str(e))
        break

print("Exiting script")