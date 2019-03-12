




"""
1. GRU
2. Bi-directional
3. CNN
"""

"""
LSTM
* **ミニバッチ数 Samples**. One sequence is one sample. A batch is comprised of one or more samples.
* **タイムステップ（セル数）Time Steps**. One time step is one point of observation in the sample.
* **特徴数 Features**. One feature is one observation at a time step.
"""



id_worrd_dict
worrd_id_dict

MAX_TWEET_LEN
VOCAB_SIZE
EMBEDDING_DIM

TIME_STEPS
MINI_BATCH_SIZE
NUM_EPOCHS
CALLBACK = ModelCheckpoint(filepath=best_weight.h5, save_best_only=True,)
# EarlyStopping
HIDDEN_UNITS
DROPOUT_RATE

from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, SpatialDropout1D, GRU, Dense, Bidirectional, Dropout, Reshape
from sklearn.model_selection import train_test_split

# Reshape, RepeatVector
inputs = Input(shape=(MAX_TWEET_LEN,))
embedding_layer = Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_TWEET_LEN, input_length=VOCAB_SIZE)(inputs)
reshape = Reshape(target_shape=(MINI_BATCH_SIZE, TIME_STEPS, EMBEDDING_DIM))(embedding_layer)
gru_layer = GRU(units=HIDDEN_UNITS, activation="tanh", return_state,)(reshape)
dropout_layer = SpatialDropout1D(rate=DROPOUT_RATE)(GRU_layer)
outputs = Dense(units=2, activation="sigmoid")(dropout_layer)

model = Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3)

partial_x_train = x_train[:]
partial_y_train = y_train[:]
x_val = x_train[:]
y_val = y_train[:]

model.fit(x=partial_x_train, y=partial_y_train, validation_data=(x_val, y_val),
          batch_size=MINI_BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, callbacks=CALLBACK, )
model.evaluate(x=x_test, y=y_test, batch_size=MINI_BATCH_SIZE, verbose=1,)
