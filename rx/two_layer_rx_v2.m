function [DCTcoe, rxinfo] = two_layer_rx_v2(rxWave, txInfo)
% TWO_LAYER_RX_V2  Receive and reconstruct DCT blocks from a WLAN waveform.
%
%   [DCTcoe, rxinfo] = two_layer_rx_v2(rxWave, txInfo)
%
%   Returns:
%     DCTcoe  - N x N x numBlocks reconstructed DCT coefficient array
%               (zeros on base failure)
%     rxinfo  - struct with fields:
%                 .txInfo          copy of txInfo metadata
%                 .baseOK          true if base demodem CRC passed
%                 .enhOK           true if enh  demodem CRC passed
%                 .blockOK         true if base layer was received successfully
%                 .baseBytes       raw recovered base bytes (uint8), empty on failure
%                 .enhBytes        raw recovered enh  bytes (uint8), empty on failure
%                 .baseFailReason  string description of base failure ('' on success)
%                 .enhFailReason   string description of enh  failure ('' on success)
%                 .sanityOK        true if byte-count / typecast sanity checks passed
%                 .baseFailCount   0 or 1 — whether base layer failed this block
%                 .enhFailCount    0 or 1 — whether enh  layer failed this block

    % ── initialise rxinfo ────────────────────────────────────────────────────
    rxinfo = struct();
    rxinfo.txInfo         = txInfo;
    rxinfo.baseOK         = false;
    rxinfo.enhOK          = false;
    rxinfo.blockOK        = false;
    rxinfo.baseBytes      = uint8([]);
    rxinfo.enhBytes       = uint8([]);
    rxinfo.baseFailReason = '';
    rxinfo.enhFailReason  = '';
    rxinfo.sanityOK       = false;
    rxinfo.baseFailCount  = 0;
    rxinfo.enhFailCount   = 0;

    % Must match two_layer_tx_DCT quant lists
    Qb_list = [0.02 0.05 0.1 0.2];
    Qe_list = [0.02 0.05 0.1 0.2];

    K = double(txInfo.K);
    N = double(txInfo.N);
    numBlocks = 1;
    if isfield(txInfo,'numBlocks'), numBlocks = double(txInfo.numBlocks); end

    DCTcoe = zeros(N, N, numBlocks);   % default: return zeros on failure

    if isfield(txInfo,'Qb_idx'), Qb = Qb_list(txInfo.Qb_idx + 1); else, Qb = Qb_list(1); end
    if isfield(txInfo,'Qe_idx'), Qe = Qe_list(txInfo.Qe_idx + 1); else, Qe = Qe_list(1); end

    % ── sample rate ──────────────────────────────────────────────────────────
    if isfield(txInfo,'cfgBase') && ~isempty(txInfo.cfgBase)
        fs = wlanSampleRate(txInfo.cfgBase);
    elseif isfield(txInfo,'cfgEnh') && ~isempty(txInfo.cfgEnh)
        fs = wlanSampleRate(txInfo.cfgEnh);
    else
        rxinfo.baseFailReason = 'cfgBase and cfgEnh both empty';
        rxinfo.baseFailCount  = 1;
        return;
    end

    % ── layer demux ──────────────────────────────────────────────────────────
    % In split mode each transmitted waveform carries only one layer (no gap
    % is prepended), so skip demux and use rxWave directly for whichever
    % layer this packet carries.  Demux is only needed for combined mode.
    lenB = 0; lenE = 0;
    if isfield(txInfo,'lenBaseWave'), lenB = double(txInfo.lenBaseWave); end
    if isfield(txInfo,'lenEnhWave'),  lenE = double(txInfo.lenEnhWave);  end

    rxBase = rxWave;
    rxEnh  = rxWave;
    if txInfo.Tx_mode ~= 1
        gap_us = 0;
        if isfield(txInfo,'mux_gap_us'), gap_us = double(txInfo.mux_gap_us); end
        [rxBase, rxEnh] = layer_demux_bylen(rxWave, lenB, lenE, gap_us, fs);
    end

    % ── demodem ──────────────────────────────────────────────────────────────
    baseBytes = uint8([]);
    enhBytes  = uint8([]);

    if txInfo.Tx_mode == 1
        % split mode — each call carries only one layer
        if ~isfield(txInfo,'layer_type')
            rxinfo.baseFailReason = 'split mode: txInfo.layer_type missing';
            rxinfo.baseFailCount  = 1;
            return;
        end

        if txInfo.layer_type == 1
            % base layer packet
            if ~isfield(txInfo,'cfgBase') || isempty(txInfo.cfgBase) || lenB == 0
                rxinfo.baseFailReason = 'split base: cfgBase/lenBaseWave missing';
                rxinfo.baseFailCount  = 1;
                return;
            end
            [baseBytes, baseOK, baseReason] = demodem(rxBase, txInfo.cfgBase);
            rxinfo.baseOK         = baseOK;
            rxinfo.baseFailReason = baseReason;
            if ~baseOK
                rxinfo.baseFailCount = 1;
                return;   % base failed — return zeros
            end

        elseif txInfo.layer_type == 2
            % enhancement layer packet
            if ~isfield(txInfo,'cfgEnh') || isempty(txInfo.cfgEnh) || lenE == 0
                rxinfo.enhFailReason = 'split enh: cfgEnh/lenEnhWave missing';
                rxinfo.enhFailCount  = 1;
                return;
            end
            [enhBytes, enhOK, enhReason] = demodem(rxEnh, txInfo.cfgEnh);
            rxinfo.enhOK         = enhOK;
            rxinfo.enhFailReason = enhReason;
            if ~enhOK
                rxinfo.enhFailCount = 1;
                rxinfo.baseOK   = true;   % base was handled in the prior call
                rxinfo.blockOK  = true;
                rxinfo.sanityOK = true;
                return;
            end
            % strip CRC16 appended by TX to enh packet
            [enhBytes, crcOK, crcReason] = verify_and_strip_crc16(enhBytes, txInfo);
            rxinfo.enhCrcOK = crcOK;
            if ~crcOK
                rxinfo.enhFailReason = crcReason;
                rxinfo.enhFailCount  = 1;
                rxinfo.baseOK   = true;
                rxinfo.blockOK  = true;
                rxinfo.sanityOK = true;
                return;
            end
        end

    else
        % combined mode — base and enh packed into one packet
        if ~isfield(txInfo,'cfgBase') || isempty(txInfo.cfgBase) || lenB == 0
            rxinfo.baseFailReason = 'combined: cfgBase/lenBaseWave missing';
            rxinfo.baseFailCount  = 1;
            return;
        end
        [pkt, baseOK, baseReason] = demodem(rxBase, txInfo.cfgBase);
        rxinfo.baseOK         = baseOK;
        rxinfo.baseFailReason = baseReason;
        if ~baseOK
            rxinfo.baseFailCount = 1;
            return;
        end
        % strip CRC16 appended by TX to combined packet
        [pkt, crcOK, crcReason] = verify_and_strip_crc16(pkt, txInfo);
        rxinfo.combinedCrcOK = crcOK;
        if ~crcOK
            rxinfo.baseFailReason = crcReason;
            rxinfo.baseFailCount  = 1;
            return;
        end

        need = (2*K + (N*N - K)) * numBlocks;
        if numel(pkt) < need
            rxinfo.baseFailReason = sprintf('combined pkt too short: %d < %d', numel(pkt), need);
            rxinfo.baseFailCount  = 1;
            rxinfo.sanityOK       = false;
            return;
        end

        baseBytes = pkt(1 : 2*K*numBlocks);
        enhBytes  = pkt(2*K*numBlocks + 1 : 2*K*numBlocks + (N*N - K)*numBlocks);
    end

    % ── store raw bytes ───────────────────────────────────────────────────────
    rxinfo.baseBytes = baseBytes;
    rxinfo.enhBytes  = enhBytes;

    % ── sanity checks before typecast ────────────────────────────────────────
    base_q = int16([]);
    enh_q  = int8([]);

    if ~isempty(baseBytes)
        if mod(numel(baseBytes), 2) ~= 0
            rxinfo.sanityOK       = false;
            rxinfo.baseFailReason = sprintf('baseBytes not even: %d', numel(baseBytes));
            rxinfo.baseFailCount  = 1;
            return;
        end
        base_q = typecast(uint8(baseBytes(:)), 'int16');
        if numel(base_q) ~= K * numBlocks
            rxinfo.sanityOK       = false;
            rxinfo.baseFailReason = sprintf('base_q length %d ~= K*numBlocks=%d', numel(base_q), K*numBlocks);
            rxinfo.baseFailCount  = 1;
            return;
        end
    end

    if ~isempty(enhBytes)
        enh_q = typecast(uint8(enhBytes(:)), 'int8');
        if numel(enh_q) ~= (N*N - K) * numBlocks
            rxinfo.enhFailReason = sprintf('enh_q length %d ~= %d', numel(enh_q), (N*N-K)*numBlocks);
            rxinfo.enhFailCount  = 1;
            enhBytes = uint8([]);
            enh_q = int8([]);
        end
    end

    rxinfo.sanityOK = true;

    % ── dequantise ───────────────────────────────────────────────────────────
    [baseHatAll, enhHatAll] = dequant(double(base_q), double(enh_q), Qb, Qe);

    % ── reconstruct DCT coefficient blocks via zigzag ─────────────────────────
    zigzagIdx = Zigzag_index(N);
    zigzagIdx = zigzagIdx(:);
    idxB = zigzagIdx(1:K);
    idxE = zigzagIdx(K+1:end);

    DCTcoe = zeros(N, N, numBlocks);
    for i = 1:numBlocks
        block_coe = zeros(N*N, 1);
        if ~isempty(baseHatAll)
            b_off = (i-1)*K + 1;
            block_coe(idxB) = baseHatAll(b_off : b_off + K - 1);
        end
        if ~isempty(enhHatAll)
            e_off = (i-1)*(N*N-K) + 1;
            block_coe(idxE) = enhHatAll(e_off : e_off + (N*N-K) - 1);
        end
        DCTcoe(:,:,i) = reshape(block_coe, N, N);
    end

    % ── finalise rxinfo ───────────────────────────────────────────────────────
    rxinfo.baseOK  = true;
    rxinfo.blockOK = true;
    if isempty(enhBytes)
        rxinfo.enhOK       = false;
        rxinfo.enhFailCount = 1;
    else
        rxinfo.enhOK = true;
    end
end

function [payload, ok, reason] = verify_and_strip_crc16(bytes, txInfo)
    payload = uint8(bytes(:));
    ok = true;
    reason = 'ok';
    useCrc16 = isfield(txInfo,'use_crc16') && logical(txInfo.use_crc16);
    if ~useCrc16
        return;
    end
    if numel(payload) < 2
        payload = uint8([]);
        ok = false;
        reason = 'crc_too_short';
        return;
    end
    recvCrcBytes = payload(end-1:end);
    recvCrc = typecast(uint8(recvCrcBytes(:)), 'uint16');
    payload = payload(1:end-2);
    calcCrc = crc16_ccitt(payload);
    if uint16(calcCrc) ~= uint16(recvCrc)
        ok = false;
        reason = 'crc_mismatch';
    end
end

function crc = crc16_ccitt(data)
    data = uint8(data(:));
    crc = uint16(65535);
    poly = uint16(hex2dec('1021'));
    for i = 1:numel(data)
        crc = bitxor(crc, bitshift(uint16(data(i)), 8));
        for b = 1:8
            if bitand(crc, uint16(32768)) ~= 0
                crc = bitxor(bitshift(crc, 1), poly);
            else
                crc = bitshift(crc, 1);
            end
        end
    end
end
