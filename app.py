import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


## function to predict the next word

def predict_next_word(model,tokenizer, text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] ## take the last max_sequence_len-1 tokens
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre') ## pad the sequence
    predicted = model.predict(token_list, verbose=0) ## predict the next token
    predict_next_word = np.argmax(predicted,axis=1) ## get the index of the next token

    for word,index in tokenizer.word_index.items():
        if index == predict_next_word:
            return word
    return None

## streamlit app
st.title('Next Word Prediction')

input_text = st.text_input('Enter the sequence of words:', "To be or not to")
if st.button('Predict Next Word'):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'The next word is: {next_word}')