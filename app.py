import torch
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write

# Load pre-trained Tacotron 2 model for generating spectrograms
tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
tacotron2.eval()

# Load pre-trained WaveGlow model for generating audio
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow.eval()

# Function to synthesize speech from text
def synthesize_speech(text, tacotron_model, waveglow_model):
    # Convert text to a sequence of tokens
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(torch.int64)

    # Generate mel-spectrogram using Tacotron 2
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, _ = tacotron_model.infer(sequence)

    # Convert mel-spectrogram to audio using WaveGlow
    with torch.no_grad():
        audio = waveglow_model.infer(mel_outputs_postnet)

    return audio

# Text input for speech synthesis
text = "Hello, this is an example of voice cloning using deep learning."

# Generate audio using the text input
audio = synthesize_speech(text, tacotron2, waveglow)

# Convert audio to numpy array and save as a .wav file
audio_numpy = audio[0].cpu().numpy()
rate = 22050  # Sampling rate
write("output_voice.wav", rate, audio_numpy)

# Print a success message
print("Audio has been successfully generated and saved as 'output_voice.wav'")
