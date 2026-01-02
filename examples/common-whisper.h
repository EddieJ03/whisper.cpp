#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <rnnoise.h>

// RNNoise denoising functions
// Initialize resamplers for RNNoise (call once before using denoise_audio)
bool init_resamplers();

// Cleanup resamplers (call when done with RNNoise)
void uninit_resamplers();

// Apply RNNoise denoising to audio data (modifies pcmf32 in-place)
// Audio is expected at 16kHz, will be resampled to 48kHz for RNNoise and back
void denoise_audio(DenoiseState* st, std::vector<float>& pcmf32);

// Read WAV audio file and store the PCM data into pcmf32
// fname can be a buffer of WAV data instead of a filename
// The sample rate of the audio must be equal to COMMON_SAMPLE_RATE
// If stereo flag is set and the audio has 2 channels, the pcmf32s will contain 2 channel PCM
bool read_audio_data(
        const std::string & fname,
        std::vector<float> & pcmf32,
        std::vector<std::vector<float>> & pcmf32s,
        bool stereo);

// convert timestamp to string, 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false);

// given a timestamp get the sample
int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate);

// write text to file, and call system("command voice_id file")
bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id);
