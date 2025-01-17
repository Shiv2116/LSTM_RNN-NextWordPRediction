import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Get input and output details for the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

## Function to predict the next word using TensorFlow Lite model
def predict_next_word_tflite(interpreter, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Take the last max_sequence_len-1 tokens
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')  # Pad the sequence
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], token_list.astype(np.float32))
    interpreter.invoke()
    
    # Get the output tensor
    predicted = interpreter.get_tensor(output_details[0]['index'])
    predict_next_word_idx = np.argmax(predicted, axis=1)  # Get the index of the next token
    
    for word, index in tokenizer.word_index.items():
        if index == predict_next_word_idx:
            return word
    return None

## Streamlit app
st.title('Next Word Prediction')

input_text = st.text_input('Enter the sequence of words:', "To be or not to")
if st.button('Predict Next Word'):
    max_sequence_len = input_details[0]['shape'][1] + 1  # Get max_sequence_len from the TFLite model input shape
    next_word = predict_next_word_tflite(interpreter, tokenizer, input_text, max_sequence_len)
    st.write(f'The next word is: {next_word}')
