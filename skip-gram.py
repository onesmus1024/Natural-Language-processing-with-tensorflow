import os
import tensorflow.keras.backend as K
from urllib.request import urlretrieve
import zipfile
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


window_size = 1


def download_data(url, data_dir):
    """Download a file if not present, and make sure it's the right
    size."""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'bbc-fulltext.zip')
    if not os.path.exists(file_path):
        print('Downloading file...')
        filename, _ = urlretrieve(url, file_path)
    else:
        print("File already exists")
    extract_path = os.path.join(data_dir, 'bbc')
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(
            os.path.join(data_dir, 'bbc-fulltext.zip'),
            'r'
        ) as zipf:
            zipf.extractall(data_dir)
    else:
        print("bbc-fulltext.zip has already been extracted")


url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
download_data(url, 'data')


def read_data(data_dir):

    news_stories = []
    print("Reading files")
    for root, dirs, files in os.walk(data_dir):
        for fi, f in enumerate(files):
            if 'README' in f:
                continue
            print("."*fi, f, end='\r')
            with open(os.path.join(root, f), encoding='latin-1') as f:
                story = []
                for row in f:
                    story.append(row.strip())
                story = ' '.join(story)
                news_stories.append(story)
    print(f"\nDetected {len(news_stories)} stories")
    return news_stories


news_stories = read_data(os.path.join('data', 'bbc'))
print(f"{sum([len(story.split(' ')) for story in news_stories])} words found in the total news set")
print('Example words (start): ', news_stories[0][:50])
print('Example words (end): ', news_stories[-1][-50:])

# Create the tokenizer
tokenizer = Tokenizer(
    num_words=15000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_\'{ | }~\t\n',
    lower=True, split=' ', oov_token='',
)

# Fit the tokenizer on the documents
tokenizer.fit_on_texts(news_stories)

n_vocab = len(tokenizer.word_index.items())+1
print(f"Vocabulary size: {n_vocab}")
print("\nWords at the top")
print('\t', dict(list(tokenizer.word_index.items())[:10]))
print("\nWords at the bottom")
print('\t', dict(list(tokenizer.word_index.items())[-10:]))


print(f"Original: {news_stories[0][:100]}")
print(
    f"Sequence IDs: {tokenizer.texts_to_sequences([news_stories[0][:100]])[0]}")
news_sequences = tokenizer.texts_to_sequences(news_stories)


sample_word_ids = news_sequences[0][:5]
sample_phrase = ' '.join([tokenizer.index_word[wid]
                         for wid in sample_word_ids])
print(f"Sample phrase: {sample_phrase}")
print(f"Sample word IDs: {sample_word_ids }\n")


inputs, labels = tf.keras.preprocessing.sequence.skipgrams(
    sample_word_ids,
    vocabulary_size=len(tokenizer.word_index.items())+1,
    window_size=window_size,
    negative_samples=0,
    shuffle=False
)

inputs, labels = np.array(inputs), np.array(labels)


print("Sample skip-grams")
for inp, lbl in zip(inputs, labels):
    print(
        f"\tInput: {inp}({[tokenizer.index_word[wi] for wi in inp]}) /Label: {lbl}")


negative_sampling_candidates, true_expected_count, sampled_expected_count = tf.random.log_uniform_candidate_sampler(
    true_classes=inputs[:1, 1:],  # [b, 1] sized tensor
    num_true=1,  # number of true words per example
    num_sampled=10,
    unique=True,
    range_max=n_vocab,
    name="negative_sampling"
)


sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
    n_vocab, sampling_factor=1e-05
)


def skip_gram_data_generator(sequences, window_size, batch_size, negative_samples, vocab_size, seed=None):
    rand_sequence_ids = np.arange(len(sequences))
    np.random.shuffle(rand_sequence_ids)

    for si in rand_sequence_ids:
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequences[si],
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0.0,
            shuffle=False,
            sampling_table=sampling_table,
            seed=seed
        )
    targets, contexts, labels = [], [], []

    for target_word, context_word in positive_skip_grams:
        context_class = tf.expand_dims(tf.constant([context_word],
                                                   dtype="int64"), 1)
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,
            num_true=1,
            num_sampled=negative_samples,
            unique=True,
            range_max=vocab_size,
            name="negative_sampling")

        # Build context and label vectors (for one target word)
        context = tf.concat(
            [tf.constant([context_word], dtype='int64'),
             negative_sampling_candidates],
            axis=0
        )
        label = tf.constant([1] + [0]*negative_samples,
                            dtype="int64")
        # Append each element from the training example to global
        # lists.
        targets.extend([target_word]*(negative_samples+1))
        contexts.append(context)
        labels.append(label)

    contexts, targets, labels = np.concatenate(contexts),np.array(targets), np.concatenate(labels)
    # If seed is not provided generate a random one
    if not seed:
        seed = random.randint(0, 10e6)
    np.random.seed(seed)
    np.random.shuffle(contexts)
    np.random.seed(seed)
    np.random.shuffle(targets)
    np.random.seed(seed)
    np.random.shuffle(labels)

    for eg_id_start in range(0, contexts.shape[0], batch_size):

        yield (
            targets[eg_id_start: min(eg_id_start+batch_size,
                                     inputs.shape[0])],
            contexts[eg_id_start: min(eg_id_start+batch_size,
                                      inputs.shape[0])]
        ), labels[eg_id_start: min(eg_id_start+batch_size,
                                   inputs.shape[0])]


batch_size = 4096  # Data points in a single batch
embedding_size = 128  # Dimension of the embedding vector.
window_size = 1  # We use a window size of 1 on either side of target word
negative_samples = 4  # Number of negative samples generated per example
epochs = 5  # Number of epochs to train for
# We pick a random validation set to sample nearest neighbors
valid_size = 16  # Random set of words to evaluate similarity on.
# We sample valid datapoints randomly from a large window without always
# being deterministic
valid_window = 250
# When selecting valid examples, we select some of the most frequent words
# as well as some moderately rare words as well
np.random.seed(54321)
random.seed(54321)
valid_term_ids = np.array(random.sample(range(valid_window), valid_size))
valid_term_ids = np.append(
    valid_term_ids, random.sample(range(1000, 1000+valid_window),
                                  valid_size),
    axis=0
)

K.clear_session()

input_1 = tf.keras.layers.Input(shape=(), name='target')
input_2 = tf.keras.layers.Input(shape=(), name='context')


# Two embeddings layers are used one for the context and one for the
# target
target_embedding_layer = tf.keras.layers.Embedding(
    input_dim=n_vocab, output_dim=embedding_size,
    name='target_embedding'
)
context_embedding_layer = tf.keras.layers.Embedding(
    input_dim=n_vocab, output_dim=embedding_size,
    name='context_embedding'
)


# Lookup outputs of the embedding layers
target_out = target_embedding_layer(input_1)
context_out = context_embedding_layer(input_2)

# Computing the dot product between the two
out = tf.keras.layers.Dot(axes=-1)([context_out, target_out])

# Defining the model
skip_gram_model = tf.keras.models.Model(inputs=[input_1, input_2],
                                        outputs=out, name='skip_gram_model')
# Compiling the model
skip_gram_model.compile(loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=True), optimizer='adam', metrics=['accuracy'])


skip_gram_model.summary()


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_term_ids, model_with_embeddings, tokenizer):

        self.valid_term_ids = valid_term_ids
        self.model_with_embeddings = model_with_embeddings
        self.tokenizer = tokenizer
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        """ Validation logic """
        # We will use context embeddings to get the most similar words

        # Other strategies include: using target embeddings, mean
        # embeddings after avaraging context/target
        embedding_weights = self.model_with_embeddings.get_layer(
            "context_embedding"
        ).get_weights()[0]
        normalized_embeddings = embedding_weights / \
            np.sqrt(np.sum(embedding_weights**2, axis=1, keepdims=True))
        # Get the embeddings corresponding to valid_term_ids
        valid_embeddings = normalized_embeddings[self.valid_term_ids,
                                                 :]
        # Compute the similarity between valid_term_ids and all the
        # embeddings
        # V x d (d x D) => V x D
        top_k = 5  # Top k items will be displayed
        similarity = np.dot(valid_embeddings, normalized_embeddings.T)
        # Invert similarity matrix to negative
        # Ignore the first one because that would be the same word as the
        # probe word
        similarity_top_k = np.argsort(-similarity, axis=1)[:, 1:
                                                           top_k+1]
        # Print the output
        for i, term_id in enumerate(valid_term_ids):
            similar_word_str = ', '.join(
                [self.tokenizer.index_word[j] for j in similarity_top_k[i, :] if j > 1])
            print(f"{self.tokenizer.index_word[term_id]}: {similar_word_str}")
        print('\n')


skipgram_validation_callback = ValidationCallback(
    valid_term_ids, skip_gram_model, tokenizer)
for ei in range(epochs):
    print(f"Epoch: {ei+1}/{epochs} started")
    news_skip_gram_gen = skip_gram_data_generator(
        news_sequences, window_size, batch_size, negative_samples,
        n_vocab
    )
    skip_gram_model.fit(
        news_skip_gram_gen, epochs=1,
        callbacks=skipgram_validation_callback,
    )
