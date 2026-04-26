function [txWave, txInfo] = two_layer_tx_AE(img, policy, frameId)
    % TWO_LAYER_TX_AE  AI-driven transmitter based on Python autoencoder
    % Enhanced: automatic image preprocessing, more robust metadata, Python call conflict fixes.
    % Key fix: pass uint8 directly to avoid normalization pitfalls and maximize PSNR.
    
        txInfo = struct();
        txInfo.frameId = frameId;
        txInfo.Tx_mode = 1; % Force dual-layer UEP mode on
    
        %% 1. Image preprocessing (safety checks)
        % Ensure grayscale
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % Enforce 64x64 as required by the AI model
        if size(img, 1) ~= 64 || size(img, 2) ~= 64
            img = imresize(img, [64 64]);
        end
        
        % Ensure uint8 at preprocessing stage
        if ~isa(img, 'uint8')
            img = uint8(img);
        end
    
        %% 2. Call Python AI for encoding, layering, and quantization
        try
            % Critical: keep uint8 end-to-end; do not cast to double or divide by 255.
            % Reason: double 0~255 can saturate Tanh (e.g. watercolor-like output, PSNR ~8.6);
            % 0~1 float with Python-side clipping can collapse to zeros (black screen, PSNR ~5.7).
            % uint8 lets PyTorch ToTensor handle the correct mapping.
            img_flat = img(:).'; 
            
            % MATLAB array -> Python numpy (dtype uint8), reshape to 64x64
            py_img = py.numpy.array(img_flat).reshape(int32([64, 64]));
            
            % import_module for an explicit module handle
            ae_mod = py.importlib.import_module('ae_inference_api');
            
            % Inference API
            py_out = ae_mod.encode_for_matlab(py_img);
            
            % Unpack tuple (base_bytes, enh_bytes)
            base_py = py_out{1};
            enh_py  = py_out{2};
            
            % Python int8 numpy -> MATLAB uint8 byte streams
            baseBytes = typecast(int8(double(base_py)), 'uint8');
            enhBytes  = typecast(int8(double(enh_py)), 'uint8');
            
        catch ME
            error('TX_AI_ERROR: Python AE API call failed. Message: %s', ME.message);
        end
    
        % Record payload lengths for RX demux
        txInfo.basePktLen = numel(baseBytes);
        txInfo.enhPktLen  = numel(enhBytes);
    
        %% 3. Modulate AI byte streams (UEP preserved)
        
        % Base layer: high-robustness modulation (e.g. QPSK)
        [waveBase, cfgBase] = modem(baseBytes, double(policy.mcs_base));
        txInfo.lenBaseWave = length(waveBase);
        txInfo.cfgBase = cfgBase;
        txInfo.mcs_base = policy.mcs_base;
    
        % Enhancement layer: high throughput but fragile (e.g. 64-QAM)
        [waveEnh, cfgEnh] = modem(enhBytes, double(policy.mcs_enh));
        txInfo.lenEnhWave = length(waveEnh);
        txInfo.cfgEnh = cfgEnh;
        txInfo.mcs_enh = policy.mcs_enh;
    
        %% 4. Waveform multiplexing
        % Sample rate from WLAN modem config
        fs = wlanSampleRate(cfgBase);
        txInfo.fs = fs; 
        
        txWave = layer_mux(waveBase, waveEnh, policy.mux_gap_us, fs);
        txInfo.mux_gap_us = policy.mux_gap_us;
    
    end