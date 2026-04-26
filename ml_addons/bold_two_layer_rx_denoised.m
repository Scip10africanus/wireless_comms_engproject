function [DCTcoe, rxinfo] = bold_two_layer_rx_denoised(rxWave, txInfo, snr_dB)
% BOLD_TWO_LAYER_RX_DENOISED  RX pipeline with ML symbol denoiser
%
% Drop-in replacement for two_layer_rx_testing_dct that routes base and
% enhancement layer decoding through bold_demodem_denoised instead of the
% standard demodem, applying the trained symbol denoiser.
%
% Inputs:
%   rxWave  - received time-domain waveform
%   txInfo  - TX metadata struct (identical to two_layer_rx_testing_dct)
%   snr_dB  - current estimated SNR in dB (for denoiser conditioning)
%
% Bold (Unubold) - EEME30003 Major Project

    if nargin < 3, snr_dB = 20; end

    rxinfo = struct();
    rxinfo.txInfo = txInfo;
    rxinfo.baseFailReason = '';
    rxinfo.enhFailReason  = '';
    rxinfo.denoiserActive = false;

    Qb_list = [0.02 0.05 0.1 0.2];
    Qe_list = [0.02 0.05 0.1 0.2];

    K         = double(txInfo.K);
    N         = double(txInfo.N);
    numBlocks = double(txInfo.numBlocks);

    if isfield(txInfo,'Qb_idx'), Qb = Qb_list(txInfo.Qb_idx + 1); else, Qb = Qb_list(1); end
    if isfield(txInfo,'Qe_idx'), Qe = Qe_list(txInfo.Qe_idx + 1); else, Qe = Qe_list(1); end

    if isfield(txInfo,'cfgBase') && ~isempty(txInfo.cfgBase)
        fs = wlanSampleRate(txInfo.cfgBase);
    elseif isfield(txInfo,'cfgEnh') && ~isempty(txInfo.cfgEnh)
        fs = wlanSampleRate(txInfo.cfgEnh);
    else
        error('bold_two_layer_rx_denoised: both cfgBase and cfgEnh are empty');
    end

    lenB = 0; lenE = 0;
    if isfield(txInfo,'lenBaseWave'), lenB = double(txInfo.lenBaseWave); end
    if isfield(txInfo,'lenEnhWave'),  lenE = double(txInfo.lenEnhWave);  end

    rxBase = rxWave;
    rxEnh  = rxWave;
    if txInfo.Tx_mode ~= 1
        gap_us = 0; if isfield(txInfo,'mux_gap_us'), gap_us = double(txInfo.mux_gap_us); end
        [rxBase, rxEnh] = layer_demux_bylen(rxWave, lenB, lenE, gap_us, fs);
    end

    baseBytes = uint8([]);
    enhBytes  = uint8([]);

    if txInfo.Tx_mode == 1
        if ~isfield(txInfo,'layer_type')
            error('bold_two_layer_rx_denoised: split mode requires txInfo.layer_type');
        end

        if txInfo.layer_type == 1
            if isempty(txInfo.cfgBase)
                error('bold_two_layer_rx_denoised: expected base packet, cfgBase missing');
            end
            % --- Use denoised demodem for base layer ---
            [baseBytes, ok, reason, ~] = bold_demodem_denoised(rxBase, txInfo.cfgBase, snr_dB);
            rxinfo.blockOK        = ok;
            rxinfo.baseFailReason = reason;
            rxinfo.baseBytes      = baseBytes;
            rxinfo.denoiserActive = true;

        elseif txInfo.layer_type == 2
            if isempty(txInfo.cfgEnh)
                error('bold_two_layer_rx_denoised: expected enh packet, cfgEnh missing');
            end
            % --- Use denoised demodem for enhancement layer ---
            [enhBytes, ok, reason, ~] = bold_demodem_denoised(rxEnh, txInfo.cfgEnh, snr_dB);
            rxinfo.blockOK       = ok;
            rxinfo.enhFailReason = reason;
            rxinfo.enhBytesRaw   = enhBytes;
            rxinfo.denoiserActive = true;

            % CRC check (identical to two_layer_rx_testing_dct)
            [enhBytes, crcOK, crcReason] = verify_and_strip_crc16(enhBytes, txInfo);
            rxinfo.enhCrcOK = crcOK;
            if ~crcOK
                rxinfo.blockOK       = false;
                rxinfo.enhFailReason = crcReason;
            end
            rxinfo.enhBytes = enhBytes;
        end

    else
        % Combined mode
        if isempty(txInfo.cfgBase) || lenB == 0
            error('bold_two_layer_rx_denoised: combined mode expected cfgBase and lenBaseWave');
        end
        [pkt, ok, reason, ~] = bold_demodem_denoised(rxBase, txInfo.cfgBase, snr_dB);
        rxinfo.blockOK            = ok;
        rxinfo.combinedFailReason = reason;
        rxinfo.combinedBytesRaw   = pkt;
        rxinfo.denoiserActive     = true;

        [pkt, crcOK, crcReason] = verify_and_strip_crc16(pkt, txInfo);
        rxinfo.combinedCrcOK = crcOK;
        if ok && ~crcOK
            rxinfo.blockOK            = false;
            rxinfo.combinedFailReason = crcReason;
        end
        rxinfo.combinedBytes = pkt;

        need = (2*K + (N*N - K)) * numBlocks;
        if ~ok
            baseBytes = uint8([]); enhBytes = uint8([]);
        elseif numel(pkt) < need
            rxinfo.blockOK            = false;
            rxinfo.combinedFailReason = 'pkt_too_short';
            baseBytes = uint8([]); enhBytes = uint8([]);
        else
            baseBytes = pkt(1 : 2*K*numBlocks);
            enhBytes  = pkt(2*K*numBlocks + 1 : 2*K*numBlocks + (N*N - K)*numBlocks);
        end
    end

    %% Bytes -> quantized coefficients (identical to two_layer_rx_testing_dct)
    base_q = int16([]);
    enh_q  = int8([]);

    if ~isempty(baseBytes)
        assert(mod(numel(baseBytes),2)==0, 'RX: baseBytes not even: %d', numel(baseBytes));
        base_q = typecast(uint8(baseBytes(:)), 'int16');
        assert(numel(base_q)==K*numBlocks, 'RX: base_q=%d expected=%d', numel(base_q), K*numBlocks);
    end
    if ~isempty(enhBytes)
        enh_q = typecast(uint8(enhBytes(:)), 'int8');
        assert(numel(enh_q)==(N*N-K)*numBlocks, 'RX: enh_q=%d expected=%d', numel(enh_q), (N*N-K)*numBlocks);
    end

    [baseHatAll, enhHatAll] = dequant(double(base_q), double(enh_q), Qb, Qe);

    zigzagIdx = Zigzag_index(N); zigzagIdx = zigzagIdx(:);
    idxB = zigzagIdx(1:K);
    idxE = zigzagIdx(K+1:end);

    DCTcoe = zeros(N, N, numBlocks);
    limit  = 3000;
    maxAbsPerBlock = zeros(1, numBlocks);

    for i = 1:numBlocks
        block_coe = zeros(N*N, 1);
        if ~isempty(baseHatAll)
            b_idx = (i-1)*K + 1;
            block_coe(idxB) = baseHatAll(b_idx : b_idx+K-1);
        end
        if ~isempty(enhHatAll)
            e_idx = (i-1)*(N*N-K) + 1;
            block_coe(idxE) = enhHatAll(e_idx : e_idx+(N*N-K)-1);
        end
        DCTcoe(:,:,i)      = reshape(block_coe, N, N);
        maxAbsPerBlock(i)  = max(abs(block_coe));
    end

    rxinfo.sanityLimit      = limit;
    rxinfo.sanityFailBlocks = find(maxAbsPerBlock > limit);
    rxinfo.sanityOK         = isempty(rxinfo.sanityFailBlocks);
    rxinfo.sanityWarning    = ~rxinfo.sanityOK;
end

function [payload, ok, reason] = verify_and_strip_crc16(bytes, txInfo)
    payload = uint8(bytes(:));
    ok = true; reason = 'ok';
    useCrc16 = isfield(txInfo,'use_crc16') && logical(txInfo.use_crc16);
    if ~useCrc16, return; end
    if numel(payload) < 2
        payload = uint8([]); ok = false; reason = 'crc_too_short'; return;
    end
    recvCrc  = typecast(uint8(payload(end-1:end)), 'uint16');
    payload  = payload(1:end-2);
    calcCrc  = crc16_ccitt(payload);
    if uint16(calcCrc) ~= uint16(recvCrc)
        ok = false; reason = 'crc_mismatch';
    end
end

function crc = crc16_ccitt(data)
    data = uint8(data(:));
    crc  = uint16(65535);
    poly = uint16(hex2dec('1021'));
    for i = 1:numel(data)
        crc = bitxor(crc, bitshift(uint16(data(i)), 8));
        for b = 1:8
            if bitand(crc, uint16(32768)) ~= 0
                crc = bitxor(bitshift(crc,1), poly);
            else
                crc = bitshift(crc,1);
            end
        end
    end
end
