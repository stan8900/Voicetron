import torch
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write

# Использование предобученной модели Tacotron 2 для генерации голосовых спектрограмм
tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
tacotron2.eval()

# Предобученная модель WaveGlow для синтеза аудио
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow.eval()

# Функция для синтеза речи
def synthesize_speech(text, tacotron_model, waveglow_model):
    # Преобразуем текст в спектрограмму с помощью Tacotron 2
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(torch.int64)

    # Генерация мел-спектрограммы
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, _ = tacotron_model.infer(sequence)

    # Генерация аудио с помощью WaveGlow
    with torch.no_grad():
        audio = waveglow_model.infer(mel_outputs_postnet)

    return audio

# Текст для синтеза
text = "Hello, this is an example of voice cloning using deep learning."

# Синтезируем аудио
audio = synthesize_speech(text, tacotron2, waveglow)

# Сохраняем аудио в файл
audio_numpy = audio[0].cpu().numpy()
rate = 22050  # частота дискретизации
write("output_voice.wav", rate, audio_numpy)

print("Аудио успешно сгенерировано и сохранено в 'output_voice.wav'")
