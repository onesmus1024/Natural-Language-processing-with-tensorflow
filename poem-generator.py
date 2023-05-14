import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Define the corpus of text data
corpus = ['a red rose is a symbol of love',
          'the sky is blue',
          'the sun is shining bright',
          'the cat in the hat',
          'the quick brown fox jumped over the lazy dog']

# Define hyperparameters
EMBEDDING_DIM = 100
MAX_SEQ_LENGTH = 10
MAX_VOCAB_SIZE = 30

# Tokenize the corpus and create sequences
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

# Pad the sequences to a fixed length
padded_sequences = pad_sequences(
    sequences, maxlen=MAX_SEQ_LENGTH, padding='post')

# Train the word2vec model on the padded sequences
word2vec_model = Word2Vec(sentences=[[str(word) for word in sequence] for sequence in padded_sequences], window=5, min_count=1, workers=4)

# Define the generator model
input_layer = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,))
embedding_layer = tf.keras.layers.Embedding(
    input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM)(input_layer)
lstm_layer = tf.keras.layers.LSTM(128)(embedding_layer)
output_layer = tf.keras.layers.Dense(
    MAX_VOCAB_SIZE, activation='softmax')(lstm_layer)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define the poem generation function


def generate_poem(seed_text, num_lines, num_words_per_line):
    # Convert the seed text to a sequence of integers
    seed_seq = tokenizer.texts_to_sequences([seed_text])[0]

    # Pad the sequence to a fixed length
    padded_seed_seq = pad_sequences(
        [seed_seq], maxlen=MAX_SEQ_LENGTH, padding='post')

    # Generate the poem
    poem = []
    for i in range(num_lines):
        line = seed_text
        for j in range(num_words_per_line):
            # Predict the next word using the generator model
            pred = model.predict(padded_seed_seq)[0]
            next_word_index = np.argmax(pred)

            # Convert the next word index back to a word
            next_word = tokenizer.index_word[next_word_index]

            # Append the next word to the line and update the seed sequence
            line += ' ' + next_word
            seed_seq = np.roll(seed_seq, -1)
            seed_seq[-1] = next_word_index
            padded_seed_seq = pad_sequences(
                [seed_seq], maxlen=MAX_SEQ_LENGTH, padding='post')

        # Add the line to the poem
        poem.append(line)

    return '\n'.join(poem)


# Test the poem generation function
print(generate_poem('the', 4, 5))
