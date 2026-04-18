function [DCTcoe, rxinfo] = two_layer_rx_v2(rxWave, txInfo)
% TWO_LAYER_RX_V2  Receive and reconstruct one DCT block from a WLAN waveform.
%
%   [DCTcoe, rxinfo] = two_layer_rx_v2(rxWave, txInfo)
%
%   Returns:
%     DCTcoe  - N x N reconstructed DCT coefficient block (zeros on base failure)
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

    Qb_list = [0.01 0.02 0.05 0.1];
    Qe_list = [0.01 0.02 0.05 0.1];

    K = double(txInfo.K);
    N = double(txInfo.N);
    DCTcoe = zeros(N, N);   % default: return zeros on failure

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
    lenB = 0; lenE = 0;
    if isfield(txInfo,'lenBaseWave'), lenB = double(txInfo.lenBaseWave); end
    if isfield(txInfo,'lenEnhWave'),  lenE = double(txInfo.lenEnhWave);  end
    gap_us = 0;
    if isfield(txInfo,'mux_gap_us'), gap_us = double(txInfo.mux_gap_us); end

    [rxBase, rxEnh] = layer_demux_bylen(rxWave, lenB, lenE, gap_us, fs);

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
            if isempty(txInfo.cfgBase) || lenB == 0
                rxinfo.baseFailReason = 'split base: cfgBase/lenBaseWave missing';
                rxinfo.baseFailCount  = 1;
                return;
            end
            [baseBytes, baseOK, baseReason] = demodem(rxBase, txInfo.cfgBase);
            rxinfo.baseOK         = baseOK;
            rxinfo.baseFailReason = baseReason;
            if ~baseOK
                rxinfo.baseFailCount = 1;
                return;   % base failed — return zeros, don't touch enh
            end

        elseif txInfo.layer_type == 2
            % enhancement layer packet — base already received in prior call
            if isempty(txInfo.cfgEnh) || lenE == 0
                rxinfo.enhFailReason = 'split enh: cfgEnh/lenEnhWave missing';
                rxinfo.enhFailCount  = 1;
                return;
            end
            [enhBytes, enhOK, enhReason] = demodem(rxEnh, txInfo.cfgEnh);
            rxinfo.enhOK         = enhOK;
            rxinfo.enhFailReason = enhReason;
            if ~enhOK
                rxinfo.enhFailCount = 1;
                % enh failure is non-fatal — DCTcoe stays zeros for this half
                rxinfo.baseOK   = true;   % base was handled in the earlier call
                rxinfo.blockOK  = true;
                rxinfo.sanityOK = true;
                rxinfo.enhBytes = uint8([]);
                return;
            end
        end

    else
        % combined mode — base and enh packed into one packet
        if isempty(txInfo.cfgBase) || lenB == 0
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

        % sanity: packet must be long enough for both layers
        need = 2*K + (N*N - K);
        if numel(pkt) < need
            rxinfo.baseFailReason = sprintf('combined pkt too short: %d < %d', numel(pkt), need);
            rxinfo.baseFailCount  = 1;
            rxinfo.sanityOK       = false;
            return;
        end

        baseBytes = pkt(1 : 2*K);
        enhBytes  = pkt(2*K+1 : 2*K + (N*N - K));
    end

    % ── store raw bytes in rxinfo ─────────────────────────────────────────────
    rxinfo.baseBytes = baseBytes;
    rxinfo.enhBytes  = enhBytes;

    % ── sanity checks before typecast ────────────────────────────────────────
    if ~isempty(baseBytes)
        if mod(numel(baseBytes), 2) ~= 0
            rxinfo.sanityOK       = false;
            rxinfo.baseFailReason = sprintf('baseBytes not even: %d', numel(baseBytes));
            rxinfo.baseFailCount  = 1;
            return;
        end
        base_q = typecast(uint8(baseBytes(:)), 'int16');
        if numel(base_q) ~= K
            rxinfo.sanityOK       = false;
            rxinfo.baseFailReason = sprintf('base_q length %d ~= K=%d', numel(base_q), K);
            rxinfo.baseFailCount  = 1;
            return;
        end
    else
        base_q = int16([]);
    end

    if ~isempty(enhBytes)
        enh_q = typecast(uint8(enhBytes(:)), 'int8');
        if numel(enh_q) ~= (N*N - K)
            rxinfo.enhFailReason = sprintf('enh_q length %d ~= %d', numel(enh_q), N*N-K);
            rxinfo.enhFailCount  = 1;
            enhBytes = uint8([]);   % discard bad enh, keep base
            enh_q = int8([]);
        end
    else
        enh_q = int8([]);
    end

    rxinfo.sanityOK = true;

    % ── dequantise ───────────────────────────────────────────────────────────
    [baseHat, enhHat] = dequant(double(base_q), double(enh_q), Qb, Qe);
    baseHat = baseHat(:);
    enhHat  = enhHat(:);

    % ── reconstruct DCT coefficient block via zigzag ─────────────────────────
    zigzagIdx = Zigzag_index(N);
    zigzagIdx = zigzagIdx(:);
    idxB = zigzagIdx(1:K);
    idxE = zigzagIdx(K+1:end);

    coe = zeros(N*N, 1);
    if ~isempty(baseHat), coe(idxB) = baseHat; end
    if ~isempty(enhHat),  coe(idxE) = enhHat;  end

    DCTcoe = reshape(coe, N, N);

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