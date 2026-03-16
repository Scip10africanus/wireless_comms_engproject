function [DCTcoe] = two_layer_rx_testing(rxWave, txInfo)
    Qb_list = [0.01 0.02 0.05 0.1];
    Qe_list = [0.01 0.02 0.05 0.1];

    K = double(txInfo.K);
    N = double(txInfo.N);
    %  Qb/Qe 
    if isfield(txInfo,'Qb_idx'), Qb = Qb_list(txInfo.Qb_idx + 1); else, Qb = Qb_list(1); end
    if isfield(txInfo,'Qe_idx'), Qe = Qe_list(txInfo.Qe_idx + 1); else, Qe = Qe_list(1); end

    % choose cfg for fs 
    if isfield(txInfo,'cfgBase') && ~isempty(txInfo.cfgBase)
        fs = wlanSampleRate(txInfo.cfgBase);
    elseif isfield(txInfo,'cfgEnh') && ~isempty(txInfo.cfgEnh)
        fs = wlanSampleRate(txInfo.cfgEnh);
    else
        error("RX: both cfgBase and cfgEnh are empty");
    end

    %demux by known waveform lengths
    lenB = 0; lenE = 0;
    if isfield(txInfo,'lenBaseWave'), lenB = double(txInfo.lenBaseWave); end
    if isfield(txInfo,'lenEnhWave'),  lenE = double(txInfo.lenEnhWave);  end
    gap_us = 0; if isfield(txInfo,'mux_gap_us'), gap_us = double(txInfo.mux_gap_us); end

    [rxBase, rxEnh] = layer_demux_bylen(rxWave, lenB, lenE, gap_us, fs);

    % demodem to bytes
    baseBytes = uint8([]);
    enhBytes  = uint8([]);

    if txInfo.Tx_mode == 1
        % split
        if ~isfield(txInfo,'layer_type')
            error("RX: split mode requires txInfo.layer_type");
        end

        if txInfo.layer_type == 1
            if isempty(txInfo.cfgBase) || lenB==0
                error("RX: expected base packet, but cfgBase/lenBaseWave missing");
            end
            baseBytes = demodem(rxBase, txInfo.cfgBase); % uint8
        elseif txInfo.layer_type == 2
            if isempty(txInfo.cfgEnh) || lenE==0
                error("RX: expected enh packet, but cfgEnh/lenEnhWave missing");
            end
            enhBytes = demodem(rxEnh, txInfo.cfgEnh);    % uint8
        end

    else
        % combined
        if isempty(txInfo.cfgBase) || lenB==0
            error("RX: combined mode expected cfgBase/lenBaseWave");
        end
        pkt = demodem(rxBase, txInfo.cfgBase); % uint8: [baseBytes; enhBytes]
        need = 2*K + (N*N - K);
        assert(numel(pkt) >= need, "RX: combined pkt too short: %d < %d", numel(pkt), need);

        baseBytes = pkt(1:2*K);
        enhBytes  = pkt(2*K+1 : 2*K + (N*N - K));
    end

    % fprintf("[RX] baseBytes=%d enhBytes=%d\n", numel(baseBytes), numel(enhBytes));

    % bytes to quantized coeffs
    base_q = int16([]);  % K coeffs
    enh_q  = int8([]);   % (64-K) coeffs

    if ~isempty(baseBytes)
        assert(mod(numel(baseBytes),2)==0, "RX: baseBytes not even: %d", numel(baseBytes));
        base_q = typecast(uint8(baseBytes(:)), 'int16');
        assert(numel(base_q)==K, "RX: base_q=%d expected=%d", numel(base_q), K);
    end

    if ~isempty(enhBytes)
        enh_q = typecast(uint8(enhBytes(:)), 'int8');
        assert(numel(enh_q)==(N*N - K), "RX: enh_q=%d expected=%d", numel(enh_q), (N*N - K));
    end

    % dequantize
    [baseHat, enhHat] = dequant(double(base_q), double(enh_q), Qb, Qe);
    baseHat = baseHat(:);
    enhHat  = enhHat(:);

    % place into zigzag positions
    zigzagIdx = Zigzag_index(N); zigzagIdx = zigzagIdx(:);

    DCTcoe = zeros(N*N,1);
    idxB = zigzagIdx(1:K);
    idxE = zigzagIdx(K+1:end);

    if ~isempty(baseHat), DCTcoe(idxB) = baseHat; end
    if ~isempty(enhHat),  DCTcoe(idxE) = enhHat;  end

    DCTcoe = reshape(DCTcoe, N, N);
end