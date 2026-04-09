function [res_hat, rxinfo] = two_layer_rx_testing_dwt(rxWaves, txInfo, rx_Qb, rx_Qe)
    rxinfo = struct();
    rxinfo.txInfo = txInfo;

    rxinfo.numBasePkts = txInfo.numBaseBlocks;
    rxinfo.numEnhPkts  = txInfo.numEnhBlocks;

    rxinfo.numBaseOk = 0;
    rxinfo.numEnhOk  = 0;

    rxinfo.baseFailReasons = {};
    rxinfo.enhFailReasons  = {};

    baseBlocks = cell(txInfo.numBaseBlocks, 1);

    if txInfo.numBaseBlocks > 0
        for i = 1:txInfo.numBaseBlocks
            [rxBaseBytes, ok, reason] = demodem(rxWaves.base{i}, txInfo.cfgBase);

            if ~ok
                rxinfo.baseFailReasons{end+1} = sprintf('base block %d: %s', i, reason);
                baseBlocks{i} = zeros(txInfo.blk_size, txInfo.blk_size);
                continue;
            end

            try
                rxBaseInt16 = typecast(rxBaseBytes, 'int16');
                [blk_hat, ~] = dequant(rxBaseInt16, [], rx_Qb, rx_Qe);
                baseBlocks{i} = reshape(blk_hat, txInfo.blk_size, txInfo.blk_size);
                rxinfo.numBaseOk = rxinfo.numBaseOk + 1;
            catch ME
                rxinfo.baseFailReasons{end+1} = sprintf('base block %d: base_unpack_fail:%s', ...
                    i, ME.identifier);
                baseBlocks{i} = zeros(txInfo.blk_size, txInfo.blk_size);
            end
        end

        base_hat_coeffs = untile_coeffs(baseBlocks, txInfo.basePadInfo, txInfo.blk_size);
    else
        base_hat_coeffs = zeros(txInfo.basePadInfo);
    end


    res_hat = struct();
    res_hat.finalLL = base_hat_coeffs;
    res_hat.subbands = struct('LH', {}, 'HL', {}, 'HH', {}, 'inputSize', {});

    for k = 1:txInfo.level
        res_hat.subbands(k).LH = [];
        res_hat.subbands(k).HL = [];
        res_hat.subbands(k).HH = [];
        res_hat.subbands(k).inputSize = [];
    end


    enhIdx = 1;

    for i = 1:numel(txInfo.enhSubbandMeta)
        meta = txInfo.enhSubbandMeta(i);
        sbBlocks = cell(meta.numBlocks, 1);

        for b = 1:meta.numBlocks
            waveEnh = rxWaves.enh{enhIdx};
            [rxEnhBytes, ok, reason] = demodem(waveEnh, txInfo.cfgEnh);

            if ~ok
                rxinfo.enhFailReasons{end+1} = sprintf( ...
                    'enh block %d (%s%d #%d): %s', ...
                    enhIdx, meta.type, meta.level, b, reason);
                sbBlocks{b} = zeros(txInfo.blk_size, txInfo.blk_size);
                enhIdx = enhIdx + 1;
                continue;
            end

            try
                rxEnhInt8 = typecast(rxEnhBytes, 'int8');
                [~, blk_hat] = dequant([], rxEnhInt8, rx_Qb, rx_Qe);
                sbBlocks{b} = reshape(blk_hat, txInfo.blk_size, txInfo.blk_size);
                rxinfo.numEnhOk = rxinfo.numEnhOk + 1;
            catch ME
                rxinfo.enhFailReasons{end+1} = sprintf( ...
                    'enh block %d (%s%d #%d): enh_unpack_fail:%s', ...
                    enhIdx, meta.type, meta.level, b, ME.identifier);
                sbBlocks{b} = zeros(txInfo.blk_size, txInfo.blk_size);
            end

            enhIdx = enhIdx + 1;
        end

        sbMat = untile_coeffs(sbBlocks, meta.origSize, txInfo.blk_size);
        res_hat.subbands(meta.level).(meta.type) = sbMat;

        if isempty(res_hat.subbands(meta.level).inputSize)
            res_hat.subbands(meta.level).inputSize = meta.origSize * 2;
        end
    end

    rxinfo.basePER = 1 - rxinfo.numBaseOk / max(rxinfo.numBasePkts, 1);
    rxinfo.enhPER  = 1 - rxinfo.numEnhOk  / max(rxinfo.numEnhPkts, 1);

end