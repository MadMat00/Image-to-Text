import os
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Flatten, RepeatVector
from tensorflow.keras.applications import VGG16
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

embedding_dim = 100
max_vocab_size = 100
max_sequence_length = 50
num_images = 1000

callbacks = [ModelCheckpoint('data/model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
               EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')]

def process_image(image_path):
    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image
    except Exception as e:
        print(f'Error processing image {image_path}: {e}')
        return None

def download_image(row_data):
    index, row = row_data
    try:
        ext = row['Attachments'].split('.')[-1]
        image_path = f'data/images/{row["ID"]}.{ext}'
        if not os.path.exists('data/images'):
            os.makedirs('data/images')
        if os.path.exists(image_path):
            return
        response = requests.get(row['Attachments'], stream=True)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
    except Exception as e:
        print(f'Error downloading image {row["ID"]}: {e}')

def download_images(num_images=100):
    df = pd.read_csv('data/image_prompts_df.csv')
    df = df.head(num_images)
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(download_image, df.iterrows()), total=df.shape[0], desc='Downloading images'))

def create_df(images_path='data/images', num_images=30000):
    images = []
    prompts = []
    df = pd.read_csv('data/image_prompts_df.csv')
    df.set_index('ID', inplace=True)
    for image_file in tqdm(os.listdir(images_path)[:num_images], desc='Loading images'):
        image_id = image_file.split('.')[0]
        if image_id in df.index.tolist():
            images.append(image_file)
            prompts.append(df.loc[image_id, 'clean_prompts'])
    print(f'Loaded {len(images)} images and {len(prompts)} prompts')
    return images, prompts

def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf8') as f:
        for line in tqdm(f, desc='Loading GloVe embeddings'):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_embedding_matrix(tokenizer, embeddings_index):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tqdm(tokenizer.word_index.items(), desc='Creating embedding matrix'):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_model(vocab_size, embedding_matrix):
    input_shape = (224, 224, 3)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    cnn_input = Input(shape=input_shape)
    cnn_features = base_model(cnn_input)
    cnn_features = Flatten()(cnn_features)
    cnn_features = Dense(256, activation='relu')(cnn_features)

    repeated_cnn_features = RepeatVector(max_sequence_length)(cnn_features)

    text_input = Input(shape=(max_sequence_length,))
    text_features = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], 
                              input_length=max_sequence_length, trainable=False)(text_input)
    text_features = LSTM(256, return_sequences=True)(text_features)

    combined_features = Concatenate(axis=-1)([repeated_cnn_features, text_features])

    decoder = Dense(256, activation='relu')(combined_features)
    outputs = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[cnn_input, text_input], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def main():
    #download_images(num_images=num_images)
    images, prompts = create_df(num_images=num_images)
    
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(prompts)
    sequences = tokenizer.texts_to_sequences(prompts)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

    glove_embeddings = load_glove_embeddings('data/embedding/glove.6B.100d.txt')
    embedding_matrix = create_embedding_matrix(tokenizer, glove_embeddings)
    
    encoded_images = []
    for image in tqdm(images, desc='Processing images'):
        processed_image = process_image(f'data/images/{image}')
        if processed_image is not None:
            encoded_images.append(processed_image)

    encoded_images = np.vstack(encoded_images)

    vocab_size = len(tokenizer.word_index) + 1
    target_sequences = np.zeros((len(padded_sequences), max_sequence_length, vocab_size))
    for i, sequence in enumerate(padded_sequences):
        for t, word_idx in enumerate(sequence):
            if t < max_sequence_length - 1:
                target_sequences[i, t, word_idx] = 1

    model = create_model(vocab_size, embedding_matrix)
    model.fit([encoded_images, padded_sequences[:, :-1]], target_sequences[:, 1:, :], epochs=100, verbose=1, batch_size=256, callbacks=callbacks)

    model.save('data/model.h5')

    predictions = model.predict([encoded_images, padded_sequences[:, :-1]])
    print(predictions[0])

if __name__ == '__main__':
    main()
