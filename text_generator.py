#---imports---#
import tensorflow as tf
import csv
import numpy as np
import os
import time

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # creates mask to stop unknown outputs
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(values=[-float('inf')]*len(skip_ids),
                                      indices=skip_ids,
                                      dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_single_step(self, inputs, states=None):
        # Convert strings to token IDs
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states

#---preprocessing---#

# reads text from csv file taking only the item names
def read_item_data(file_path, column_index):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > column_index:
                data.append(row[column_index])
    return data

# outputs text that has been put into a string list
def convert_to_list():
    file_path = "Runescape_Item_Names.csv"
    column_index = 1
    csv_data = read_item_data(file_path, column_index)
    del csv_data[0]     # to delete heading entry
    return csv_data

# writes list to text file for proper formatting
def write_to_file(list_of_items):
    file_name = 'items.txt'
    file = open(file_name, 'w')
    for each in list_of_items:
        file.write(each+"\n")
    file.close()
    return file, file_name

def decode_py2(name):
    py2 = open(name, 'rb').read().decode(encoding='utf-8')
    print(f'Length of text: {len(py2)} characters')
    vocab = sorted(set(py2))
    print(f'{len(vocab)} unique characters')
    print("ok")
    return py2, vocab

def tokenization(file):
    example_text = ['abcdefg', 'qrstuvxyz']
    chars = tf.strings.unicode_split(example_text, input_encoding='UTF-8')
    print(chars)
    id_chars = tf.keras.layers.StringLookup(vocabulary=file, mask_token=None)
    id_token = id_chars(chars)
    print(id_token)
    chars_id = tf.keras.layers.StringLookup(vocabulary=id_chars.get_vocabulary(), invert=True, mask_token=None)
    chars_token = chars_id(id_token)
    print(chars_token)

    return id_chars, chars_id, chars

def text_from_ids(chars_id, ids):
    return tf.strings.reduce_join(chars_id(ids), axis=-1)

def split_data(sequence):
    input_data = sequence[:-1]
    target_data = sequence[1:]
    return input_data, target_data

def formatting_data(text, data_id, data_chars):
    all_ids = data_id(tf.strings.unicode_split(text, 'UTF-8'))
    print(all_ids)
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    for ids in ids_dataset.take(10):
        print(data_chars(ids).numpy().decode('utf-8'))

    seq_len = 100
    sequences = ids_dataset.batch(seq_len+1, drop_remainder=True)
    for i in sequences.take(1):
        print(data_chars(i))

    dataset = sequences.map(split_data)

    for input_example, target_example in dataset.take(1):
        print("Input :", text_from_ids(data_chars, input_example).numpy())
        print("Target:", text_from_ids(data_chars, target_example).numpy())

    return dataset

def create_training_batches(dataset):
    batch_size = 32
    buffer_size = 1000

    dataset = (dataset
               .shuffle(buffer_size)
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.experimental.AUTOTUNE))
    print(dataset)
    return dataset


def main():
    #---PREPROCESSING---#
    csv_data = convert_to_list()
    text, path = write_to_file(csv_data)
    decoded, vocab = decode_py2(path)
    id_from_chars, chars_from_id, chars = tokenization(vocab)
    dataset = formatting_data(decoded, id_from_chars, chars_from_id)
    training_sets = create_training_batches(dataset)

    #---CREATING THE MODEL---#

    vocab_size = len(id_from_chars.get_vocabulary())
    embedding_dim = 256
    rnn_units = 1024

    model = MyModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units
    )

    for input_example_batch, target_example_batch in training_sets.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print(sampled_indices)

    print("Input:\n", text_from_ids(chars_from_id, input_example_batch[0]).numpy())
    print()
    print("Next Char Predictions:\n", text_from_ids(chars_from_id, sampled_indices).numpy())

    #---SETUP FOR TRAINING THE MODEL---#

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", example_batch_mean_loss)
    print(tf.exp(example_batch_mean_loss).numpy())

    model.compile(optimizer='adam', loss=loss)

    #---CONFIGURE CHECKPOINTS---#
    directory = './training_checkpoints'
    checkpoints_prefix = os.path.join(directory, "ckpt_{epoch}")
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoints_prefix,
        save_weights_only=True)
    #---EXECUTE TRAINING---#
    epochs = 30
    history = model.fit(training_sets, epochs=epochs, callbacks=[checkpoints_callback])

    #---GENERATING TEXT---#
    one_step = OneStep(model, chars_from_id, id_from_chars)

    start_execute = time.time()
    states = None
    next_char = tf.constant([' '])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step.generate_single_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end_execute = time.time()
    with open("output.txt", "a") as f:
        print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80, file=f)
    print('\nRun time:', end_execute - start_execute,file=f)

    tf.saved_model.save(one_step, 'one_step')
    one_step_reloaded = tf.saved_model.load('one_step')

    states = None
    next_char = tf.constant([' '])
    result = [next_char]

    for n in range(100):
        next_char, states = one_step_reloaded.generate_single_step(next_char, states=states)
        result.append(next_char)

    print(tf.strings.join(result)[0].numpy().decode("utf-8"))

main()