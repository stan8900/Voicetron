# Voice Cloning with Tacotron 2 and WaveGlow

This project demonstrates real-time voice cloning using pre-trained models like **Tacotron 2** and **WaveGlow**. You can synthesize speech from text and fine-tune the models for specific voices, accents, emotions, and more.

## Features
- Voice cloning with a short audio clip
- Real-time speech synthesis from text
- Pre-trained models for easy setup
- Easily extendable for voice style transfer (e.g., emotions, accents)

## Requirements
- Python 3.7+
- PyTorch
- CUDA (optional, but recommended for GPU support)

## Setup Instructions

### 1. Clone the repository
```
git clone https://github.com/stan8900/Voicetron.git
cd https://github.com/stan8900/Voicetron.git
```
### 2. Create and activate a virtual environment
It's recommended to create a virtual environment for managing project dependencies. You can use venv for this.


On macOS/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```
python -m venv venv
venv\Scripts\activate
```
### 3. Install dependencies
Install the necessary Python packages using pip:

```
pip install torch
pip install scipy
pip install soundfile

```
For GPU (CUDA) support, install the appropriate version of torch with CUDA:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

Make sure you replace cu118 with the appropriate version if you are using a different CUDA version.

### 4. Running the script
Once the environment is set up and dependencies are installed, you can run the script to synthesize speech. Make sure your virtual environment is activated.
```
python3 voice_cloning.py

```