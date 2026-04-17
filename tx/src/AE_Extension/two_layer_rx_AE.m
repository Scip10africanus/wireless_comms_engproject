function [rxImage, rxInfo] = two_layer_rx_AE(rxWave, txInfo, policy)
    % TWO_LAYER_RX_AE AI-driven Receiver (Robust Version)
    % Key improvement: intelligent de-normalization and padding to resist sync truncation in demodem.
    
        rxInfo = struct();
        
        % 1. Determine sample rate
        fs = 20e6;
        if isfield(txInfo, 'fs'), fs = txInfo.fs; end
        
        %% 2. Demultiplex (split waveform)
        try
            [waveBase, waveEnh] = layer_demux_bylen(rxWave, txInfo.lenBaseWave, txInfo.lenEnhWave, policy.mux_gap_us, fs);
            
            % Pad to avoid "too short" errors from demodem sync truncation
            pad_len = 500; 
            waveBase = [waveBase(:); complex(zeros(pad_len, 1))];
            waveEnh  = [waveEnh(:); complex(zeros(pad_len, 1))];
        catch ME
            error('RX_DEMUX_FAIL: Waveform demux failed. Reason: %s', ME.message);
        end
    
        %% 3. Demodulate base layer (core path)
        try
            baseBytes_uint8 = demodem(waveBase, txInfo.cfgBase);
            
            if isempty(baseBytes_uint8)
                error('Demodulator returned empty array.');
            end
            
            % Force length alignment
            if numel(baseBytes_uint8) > txInfo.basePktLen
                baseBytes_uint8 = baseBytes_uint8(1:txInfo.basePktLen);
            elseif numel(baseBytes_uint8) < txInfo.basePktLen
                error('Base payload too short: expected %d, got %d', txInfo.basePktLen, numel(baseBytes_uint8));
            end
            rxInfo.baseStatus = 'Success';
        catch ME
            error('RX_BASE_CRITICAL: %s', ME.message);
        end
    
        %% 4. Demodulate enhancement layer (noise/failure allowed)
        try
            enhBytes_uint8 = demodem(waveEnh, txInfo.cfgEnh);
            
            if ~isempty(enhBytes_uint8) && numel(enhBytes_uint8) >= txInfo.enhPktLen
                enhBytes_uint8 = enhBytes_uint8(1:txInfo.enhPktLen);
                rxInfo.enhStatus = 'Success';
            else
                % Graceful degradation: if enh layer is bad/incomplete, fill with random noise to trigger AI inpainting
                shortfall = txInfo.enhPktLen;
                enhBytes_uint8 = uint8(randi([0, 255], shortfall, 1));
                rxInfo.enhStatus = 'Corrupted_Or_Short';
            end
        catch ME
            enhBytes_uint8 = uint8(randi([0, 255], txInfo.enhPktLen, 1));
            rxInfo.enhStatus = 'Demod_Crash_Random_Filled';
        end
    
        %% 5. Python API integration and image reconstruction
        try
            base_int8 = typecast(baseBytes_uint8(:), 'int8');
            enh_int8  = typecast(enhBytes_uint8(:), 'int8');
            
            py_base = py.numpy.array(base_int8(:).');
            py_enh  = py.numpy.array(enh_int8(:).');
            
            ae_mod = py.importlib.import_module('ae_inference_api');
            py_out = ae_mod.decode_from_matlab(py_base, py_enh);
            
            rxImage = double(py_out);
            
            % Key: intelligent de-normalization
            % Model output is often a 0~1 float matrix after sigmoid.
            % If max <= 1 and not all zeros, scale by 255 for standard image display;
            % otherwise values are too small and the image looks black.
            if max(rxImage(:)) <= 1.01 && max(rxImage(:)) > 0
                rxImage = rxImage * 255.0;
            end
            
            % Clamp to prevent overflow
            rxImage = max(0, min(255, rxImage));
            
        catch ME
            error('RX_PYTHON_FAIL: Python decoder failed. Details: %s', ME.message);
        end
    end