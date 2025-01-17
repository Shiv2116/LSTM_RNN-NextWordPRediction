# Next Word Prediction Using Gutenberg Corpus (NLP with NLTK and LSTM RNN)

## Overview
This project implements a Next Word Prediction model using the Gutenberg corpus from the Natural Language Toolkit (NLTK). The model is built with a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) to predict the next word in a sequence based on the preceding context.

## Features
- Utilizes the Gutenberg corpus for training the language model.
- Implements data preprocessing using the NLTK library for tokenization, cleaning, and vocabulary building.
- Uses LSTM RNNs for handling sequential data efficiently.
- Predicts the most probable next word for a given input sequence.

## Motivation
Next Word Prediction is a crucial task in Natural Language Processing (NLP) with applications in autocomplete systems, predictive text, and conversational AI. This project showcases the use of deep learning techniques to build an effective word prediction model.

## Requirements
To run the project, ensure you have the following installed:

- Python 3.8+
- TensorFlow 2.x
- NLTK
- NumPy
- Keras
- Matplotlib (for visualizations)
- Jupyter Notebook (optional, for experimentation)

You can install the dependencies using:
```bash
pip install tensorflow nltk numpy keras matplotlib
```

## Dataset
The Gutenberg corpus is used as the dataset, available directly through the NLTK library. The corpus includes classic texts such as works by Shakespeare, Jane Austen, and more. These texts are preprocessed to create a sequence-based dataset for the LSTM model.

## Workflow

1. **Data Preprocessing**:
   - Import the Gutenberg corpus using NLTK.
   - Clean and tokenize the text to remove punctuation, convert to lowercase, and split into words.
   - Generate sequences of words for training by creating sliding windows of a fixed size.

2. **Model Development**:
   - Use an LSTM RNN architecture with an embedding layer to handle the input sequences.
   - Train the model on the prepared sequences to predict the next word.

3. **Evaluation**:
   - Measure the model's accuracy using a validation set.
   - Visualize the training and validation loss over epochs.

4. **Prediction**:
   - Input a sequence of words to generate the next word prediction.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Shiv2116/LSTM_RNN-NextWordPRediction
   cd LSTM_RNN-NextWordPRediction
   ```

2. Run the Python script:
   ```bash
   streamlit run app.py
   ```
   Alternatively, open the Jupyter Notebook and follow the steps interactively.

3. Input a sequence of words to test the model's predictions.

## Results
The model achieves reasonable accuracy in predicting the next word in various contexts from the Gutenberg corpus. However, results may vary based on training time, corpus size, and model architecture.

## Future Improvements
- Expand the dataset by incorporating additional corpora.
- Experiment with different RNN architectures (e.g., GRU) or Transformer-based models.
- Fine-tune hyperparameters for improved accuracy.
- Implement beam search or top-k sampling for enhanced prediction quality.

## Acknowledgments
- [NLTK Library](https://www.nltk.org/)
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)
- Gutenberg corpus contributors

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---
Feel free to raise an issue or contribute to the project by submitting a pull request!

