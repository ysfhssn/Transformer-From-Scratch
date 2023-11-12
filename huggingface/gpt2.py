import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Function for autoregressive text generation
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='tf')

    for _ in range(max_length):
        # Generate the next token
        logits = model(input_ids)['logits']
        predicted_id = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)

        # Concatenate the predicted token to the input for the next iteration
        input_ids = tf.concat([input_ids, predicted_id[:, tf.newaxis]], axis=-1)

    # Decode the generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Once upon a time"
generated_story = generate_text(prompt)
print(generated_story)
