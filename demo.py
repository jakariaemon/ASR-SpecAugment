import librosa
import numpy as np
import librosa.display
from augment import SpecAugment
import matplotlib.pyplot as plt 

file = "/content/SpecAugment/common_voice_bn_30614352.mp3" 
# Load the audio file
audio, sr = librosa.load(file)

# Extract Mel Spectrogram Features from the audio file
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=128, fmax=16000)
plt.figure(figsize=(14, 6))
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=16000) # Base 
# Apply SpecAugment
apply = SpecAugment(mel_spectrogram, 'LB')
time_warped = apply.time_warp()
plt.figure(figsize=(14, 6))
librosa.display.specshow(librosa.power_to_db(time_warped[0, :, :, 0].numpy(), ref=np.max), x_axis='time', y_axis='mel', fmax=16000) # Time Warped 

freq_masked = apply.freq_mask()
plt.figure(figsize=(14, 6)) 
librosa.display.specshow(librosa.power_to_db(freq_masked[0, :, :, 0], ref=np.max), x_axis='time', y_axis='mel', fmax=16000) # Freq Masked 
time_masked = apply.time_mask()
plt.figure(figsize=(14, 6))
librosa.display.specshow(librosa.power_to_db(time_masked[0, :, :, 0], ref=np.max), x_axis='time', y_axis='mel', fmax=16000) # Time Masked 



mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max) 
time_masked_db = librosa.power_to_db(time_masked[0, :, :, 0], ref=np.max) 
plt.figure(figsize=(28, 12))
plt.subplot(1, 2, 1) 
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', fmax=16000)
plt.title('a) Original Mel Spectrogram', fontsize=16)
plt.colorbar(format='%+2.0f dB')
plt.subplot(1, 2, 2) 
librosa.display.specshow(time_masked_db, x_axis='time', y_axis='mel', fmax=16000)
plt.title('b) Time Warped, Time & Freq-Masked Mel Spectrogram', fontsize=16)
plt.colorbar(format='%+2.0f dB')
plt.suptitle('Comparison of Mel Spectrograms after SpecAugmentation', fontsize=24)
plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.savefig('mel_spectrograms_comparison.png', dpi=300)
plt.show()
