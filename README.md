# SpecAugment

This repository contains an implementation of SpecAugment, a simple data augmentation method for automatic speech recognition, as proposed by Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le in their Interspeech 2019 paper.

## Implementation

A Python implementation of SpecAugment can be adapted from the original code available on GitHub, specifically modified from the version by pyyush at `https://github.com/pyyush/SpecAugment`.

### Key Features

- **Time Warping**: A spectrogram is warped along the time axis, simulating the effect of slightly faster or slower speaking rates.
- **Frequency Masking**: Random frequency channels are masked (set to zero), mimicking the effect of missing or dampened frequencies.
- **Time Masking**: Similar to frequency masking, but segments of time are masked instead, simulating pauses or missing time segments in the speech.

### Usage
