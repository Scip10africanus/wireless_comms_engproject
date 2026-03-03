================================================================================
                        CHANNEL MODEL DOCUMENTATION
================================================================================

This folder contains channel simulation models for testing the Two-Layer
Residual Coding Image Transmission System.


FILE DESCRIPTIONS
-----------------

channel_model.m
    Main channel simulation function supporting multiple channel types.

    Usage:
        [rxSignal, channelInfo] = channel_model(txSignal, params)


SUPPORTED CHANNEL TYPES
-----------------------

1. AWGN (Additive White Gaussian Noise)
   ------------------------------------
   - Simplest channel model
   - Adds Gaussian noise based on specified SNR
   - No fading or multipath effects
   - Good for baseline performance testing

   params.channelType = 'AWGN';
   params.SNR_dB = 20;  % Signal-to-noise ratio


2. RAYLEIGH (Flat Rayleigh Fading)
   --------------------------------
   - Models non-line-of-sight (NLOS) scenarios
   - Signal amplitude follows Rayleigh distribution
   - Simulates urban environments with many reflections
   - Adds time-varying fading + AWGN

   params.channelType = 'Rayleigh';
   params.SNR_dB = 20;
   params.dopplerHz = 10;      % Maximum Doppler frequency
   params.sampleRate = 1e6;    % Sample rate for Doppler calculation


3. MULTIPATH (Frequency-Selective Multipath)
   ------------------------------------------
   - Models multiple signal paths with different delays
   - Creates frequency-selective fading
   - Each path has its own delay and gain
   - Common in indoor/urban environments

   params.channelType = 'Multipath';
   params.SNR_dB = 20;
   params.pathDelays = [0, 5, 10];    % Delays in samples
   params.pathGains = [1, 0.5, 0.3];  % Linear gains


4. RICIAN (Rician Fading)
   -----------------------
   - Models line-of-sight (LOS) + scattered components
   - K-factor determines LOS strength
   - K = 0: Pure Rayleigh
   - K = inf: Pure AWGN (no fading)
   - Typical for suburban/rural with clear LOS

   params.channelType = 'Rician';
   params.SNR_dB = 20;
   params.ricianK = 4;    % Rician K-factor (LOS power / scatter power)


5. FREQUENCY_SELECTIVE
   --------------------
   - Advanced frequency-selective fading model
   - Per-subcarrier fading
   - Useful for OFDM system testing


PARAMETERS
----------
Common parameters for all channel types:

  params.channelType  - String: 'AWGN', 'Rayleigh', 'Multipath', 'Rician'
  params.SNR_dB       - Signal-to-noise ratio in decibels (default: 20)
  params.dopplerHz    - Maximum Doppler frequency in Hz (default: 10)
  params.sampleRate   - Sample rate in Hz (default: 1e6)

Multipath-specific:
  params.pathDelays   - Vector of path delays in samples
  params.pathGains    - Vector of path gains (linear scale)

Rician-specific:
  params.ricianK      - Rician K-factor (default: 4)


OUTPUT STRUCTURE
----------------
channelInfo contains:

  .type         - Channel type used
  .SNR_dB       - Configured SNR
  .txPower      - Transmitted signal power
  .rxPower      - Received signal power
  .noisePower   - Noise power added
  .measuredSNR  - Actual measured SNR after channel
  .fadingCoeff  - Fading coefficients (for fading channels)


EXAMPLE USAGE
-------------
% Simple AWGN test
params.channelType = 'AWGN';
params.SNR_dB = 15;
[rxSignal, info] = channel_model(txSignal, params);

% Rayleigh fading for urban scenario
params.channelType = 'Rayleigh';
params.SNR_dB = 20;
params.dopplerHz = 50;  % Higher Doppler = faster fading
[rxSignal, info] = channel_model(txSignal, params);

% Multipath channel
params.channelType = 'Multipath';
params.pathDelays = [0, 3, 7, 12];
params.pathGains = [1, 0.6, 0.4, 0.2];
params.SNR_dB = 18;
[rxSignal, info] = channel_model(txSignal, params);


SNR GUIDELINES
--------------
Typical SNR values for testing:

  SNR (dB) | Channel Quality | Expected Result
  ---------|-----------------|------------------
   30+     | Excellent       | Near-perfect image
   20-30   | Good            | Minor artifacts
   15-20   | Moderate        | Visible degradation
   10-15   | Poor            | Enhancement may fail
   <10     | Very poor       | Base layer only


RELATIONSHIP TO TWO-LAYER SYSTEM
--------------------------------
The channel model affects each layer differently due to UEP:

- Base layer (QPSK): More robust, survives lower SNR
- Enhancement layer (64-QAM): More sensitive, needs higher SNR

At moderate SNR (15-20 dB):
- Base layer: Usually OK
- Enhancement layer: May have errors -> graceful degradation

At low SNR (<15 dB):
- Both layers struggle
- Adaptive TX controller reduces modulation order


CONTACT
-------
For questions about channel models, contact the Communications Team.

================================================================================
