import pandas as pd


def extract_features_from_chunks(input_dir='chunks'):
    """
    Extract MFCC features from each audio chunk in the input directory.
    :param input_dir: Directory containing audio chunks.
    :return: A Pandas DataFrame containing the features for each chunk.
    """
    features = []

    # List all audio files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_dir, filename)

            # Load the audio file
            audio, sr = librosa.load(file_path, sr=None)

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

            # Calculate the mean of each MFCC (simplifying the feature vector)
            mfccs_mean = mfccs.mean(axis=1)

            # Append the features to our list, including the file name for identification
            features.append([filename] + mfccs_mean.tolist())

    # Create a DataFrame
    feature_columns = ['filename'] + [f'mfcc_{i}' for i in range(1, 14)]
    df_features = pd.DataFrame(features, columns=feature_columns)

    return df_features


# Example usage
df = extract_features_from_chunks()
print(df.head())
