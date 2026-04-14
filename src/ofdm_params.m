function p = ofdm_params()
% OFDM_PARAMS  Centralized OFDM system configuration
%   Mirrors the structure of the reference OFDM.m, adapted for
%   per-packet transmission in the image-over-OFDM project.
 
    p = struct();
 
    % --- OFDM structure (same as reference OFDM.m) ---
    p.N_fft          = 64;                 % FFT size
    p.N_cp           = 16;                 % Cyclic prefix length
    p.N_symbol       = p.N_fft + p.N_cp;   % Full OFDM symbol length (80 samples)
 
    % --- Pilot configuration ---
    p.pilot_interval = 6;                                  % same as reference
    p.pilot_pos      = 1 : p.pilot_interval : p.N_fft;     % pilot subcarrier indices
    p.N_pilot        = numel(p.pilot_pos);
    p.pilot_value    = 3 + 3j;                             % same as reference
 
    % Data subcarrier positions = everything not taken by a pilot
    all_idx          = 1 : p.N_fft;
    p.data_pos       = setdiff(all_idx, p.pilot_pos);      % data subcarrier indices
    p.N_data_per_sym = numel(p.data_pos);                  % 64 - 11 = 53
 
    % --- Channel coding ---
    p.useConvCode    = true;                 % (2,1,7) conv code, rate 1/2
    p.convConstraint = 7;
    p.convPoly       = [133 171];
    p.tblen          = 6 * p.convConstraint; % Viterbi traceback depth = 42
 
    % --- Spreading (kept as a hook, disabled by default) ---
    % Set spread_factor > 1 to enable. Reference OFDM.m uses 52 (=N_sc),
    % which is too expensive for image packets, so we default to 1 (off).
    p.spread_factor  = 1;
 
    % --- Modulation lookup (mcs index -> constellation) ---
    %  mcs 0: BPSK   (1 bit/sym)
    %  mcs 1: QPSK   (2 bits/sym)
    %  mcs 2: 16-QAM (4 bits/sym)
    %  mcs 3: 64-QAM (6 bits/sym)
    p.mcs_table = struct( ...
        'mod',          {'BPSK', 'QPSK', '16QAM', '64QAM'}, ...
        'bits_per_sym', {1, 2, 4, 6}, ...
        'M',            {2, 4, 16, 64});
 
    % --- Sample rate (used by layer_mux etc.) ---
    p.fs = 20e6;
end