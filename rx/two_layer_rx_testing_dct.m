function [DCTcoe, rxinfo] = two_layer_rx_testing_dct(rxWave, txInfo)
    rxinfo = struct();
    rxinfo.txInfo = txInfo;

    rxinfo.baseFailReason = '';
    rxinfo.enhFailReason  = '';

    Qb_list = [0.02 0.05 0.1 0.2];
    Qe_list = [0.02 0.05 0.1 0.2];

    K = double(txInfo.K);
    N = double(txInfo.N);
    numBlocks = double(txInfo.numBlocks);

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
            [baseBytes,ok,reason] = demodem(rxBase, txInfo.cfgBase); % uint8
            rxinfo.blockOK = ok;
            rxinfo.baseFailReason = reason;
        elseif txInfo.layer_type == 2
            if isempty(txInfo.cfgEnh) || lenE==0
                error("RX: expected enh packet, but cfgEnh/lenEnhWave missing");
            end
            [enhBytes,ok,reason] = demodem(rxEnh, txInfo.cfgEnh);    % uint8
            rxinfo.blockOK = ok;
            rxinfo.enhFailReason = reason;
        end

    else
        % combined
        if isempty(txInfo.cfgBase) || lenB==0
            error("RX: combined mode expected cfgBase/lenBaseWave");
        end
        [pkt,ok,reason] = demodem(rxBase, txInfo.cfgBase); % uint8: [baseBytes; enhBytes]
        rxinfo.blockOK = ok;
        rxinfo.combinedFailReason = reason;
        need = (2*K + (N*N - K)) * numBlocks;
        assert(numel(pkt) >= need, "RX: combined pkt too short: %d < %d", numel(pkt), need);

        baseBytes = pkt(1 : 2*K*numBlocks);
        enhBytes  = pkt(2*K*numBlocks + 1 : 2*K*numBlocks + (N*N - K)*numBlocks);
    end

    % bytes to quantized coeffs
    base_q = int16([]);  % K coeffs
    enh_q  = int8([]);   % (64-K) coeffs

    if ~isempty(baseBytes)
        assert(mod(numel(baseBytes),2)==0, "RX: baseBytes not even: %d", numel(baseBytes));
        base_q = typecast(uint8(baseBytes(:)), 'int16');
        assert(numel(base_q)==K*numBlocks, "RX: base_q=%d expected=%d", numel(base_q), K*numBlocks);
    end

    if ~isempty(enhBytes)
        enh_q = typecast(uint8(enhBytes(:)), 'int8');
        assert(numel(enh_q)==(N*N - K)*numBlocks, "RX: enh_q=%d expected=%d", numel(enh_q), (N*N - K)*numBlocks);
    end

    % dequantize
    [baseHatAll, enhHatAll] = dequant(double(base_q), double(enh_q), Qb, Qe);
    
    zigzagIdx = Zigzag_index(N); zigzagIdx = zigzagIdx(:);
    idxB = zigzagIdx(1:K);
    idxE = zigzagIdx(K+1:end);

    DCTcoe = zeros(N, N, numBlocks);

    for i = 1:numBlocks
        block_coe = zeros(N*N,1);
        
        if ~isempty(baseHatAll)
            b_idx = (i-1)*K + 1;
            block_coe(idxB) = baseHatAll(b_idx : b_idx + K - 1);
        end
        if ~isempty(enhHatAll)
            e_idx = (i-1)*(N*N-K) + 1;
            block_coe(idxE) = enhHatAll(e_idx : e_idx + (N*N-K) - 1);
        end
        
        DCTcoe(:,:,i) = reshape(block_coe, N, N);
    end
end