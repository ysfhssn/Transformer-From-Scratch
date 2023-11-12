from transformers import MarianTokenizer, TFAutoModelForSeq2SeqLM

# Load pre-trained Marian model and tokenizer for English to French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function for translation
def translate_text(text):
    input_ids = tokenizer.encode(text, return_tensors='tf')

    # Generate translation
    translation_ids = model.generate(input_ids)

    # Decode the generated translation
    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_text

# Example usage
english_text = "Hello, how are you?"
translated_text = translate_text(english_text)
print(translated_text)
