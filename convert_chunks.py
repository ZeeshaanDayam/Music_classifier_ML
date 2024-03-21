import librosa
import numpy as np
import os
import soundfile as sf  # Import soundfile


def divide_song_into_chunks(filename, chunk_length=10, output_dir='chunks'):
    """
    Divide a song into fixed-length chunks and save them to the output directory.
    :param filename: Path to the input audio file.
    :param chunk_length: Length of each chunk in seconds.
    :param output_dir: Directory to save the output chunks.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the full audio file
    audio, sr = librosa.load(filename, sr=None)

    # Calculate the number of samples per chunk
    samples_per_chunk = sr * chunk_length

    # Calculate the total number of chunks
    total_chunks = int(np.ceil(len(audio) / samples_per_chunk))

    # Process each chunk
    for i in range(total_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = audio[start:end]

        # Save each chunk to a file using soundfile.write
        chunk_filename = os.path.join(
            output_dir, f"{os.path.basename(filename)}_chunk{i}.wav")
        sf.write(chunk_filename, chunk, sr)  # Use soundfile.write here
        print(f"Saved chunk {i+1}/{total_chunks} to {chunk_filename}")


# Example usage
divide_song_into_chunks('Progressive Rock Songs/-04- Knots.mp3')
