function [txWaves, txInfo] = two_layer_tx_wavelet(img, policy, frameId)
level = 1; % DWT level 
BLK_SIZE = 32; % Coefficient block size 

Qb =  1; % using fixed Quant step first 
Qe =  1;

txInfo = struct();
txWaves = struct('base', [], 'enh', {{}});


info = struct();
[res, info] = DWT(img, info, level);

% Tiled to ensure packets < 4095 bytes in order to fit PSDU length in 802.11Non-HT
[baseBlocks, basePadInfo] = tile_coeffs(res.finalLL, BLK_SIZE);
baseWaves = {};
baseCfgs = {};

for i = 1:numel(baseBlocks)
    blk = baseBlocks{i};
    % Quantize
    % fprintf('BASE block %d before quant : q min=%g, max=%g\n', i, min(blk(:)), max(blk(:)));

    [blk_q, ~, ~, ~] = quant(blk(:), [], Qb, Qe);
    % fprintf('BASE block %d after quant : q min=%g, max=%g\n', i, min(blk_q(:)), max(blk_q(:)));
    blkBytes = typecast(int16(blk_q), 'uint8');
    
    % Modulate
    [w, c] = modem(blkBytes, double(policy.mcs_base));
    baseWaves{end+1} = w;
    baseCfgs{end+1} = c;
end

txWaves.base = baseWaves;
txInfo.basePadInfo = basePadInfo; % [H, W] of LL2
txInfo.numBaseBlocks = numel(baseBlocks);
if ~isempty(baseCfgs)
    txInfo.cfgBase = baseCfgs{1};
    fs = wlanSampleRate(baseCfgs{1});
else
    fs = 20e6;
end

% Order: LH2, HL2, HH2 -> LH1, HL1, HH1
enhWaves = {};
enhCfgs = {};
subbandMeta = struct('type', {}, 'level', {}, 'origSize', {}, 'numBlocks', {});

subbandTypes = {'LH', 'HL', 'HH'};
for k = level:-1:1
    for s = 1:3
        sbType = subbandTypes{s};
        coeffs = res.subbands(k).(sbType);
        
        if isempty(coeffs)
            continue;
        end
        
        % Tile the subband
        [sbBlocks, sbPadInfo] = tile_coeffs(coeffs, BLK_SIZE);
        
        % Process each block
        for b = 1:numel(sbBlocks)
            blk = sbBlocks{b};
            % Quantize (Enh uses Qe, 8-bit)
            % fprintf('ENH %s%d block %d before quant : q min=%g, max=%g\n', ...
            %     sbType, k, b, min(blk(:)), max(blk(:)));

            [~, blk_q, ~, ~] = quant([], blk(:), Qb, Qe);
            % fprintf('ENH %s%d block %d after quant : q min=%g, max=%g\n', ...
            %     sbType, k, b, min(blk_q(:)), max(blk_q(:)));
            blkBytes = typecast(int8(blk_q), 'uint8');
            
            % Modulate
            [w, c] = modem(blkBytes, double(policy.mcs_enh));
            enhWaves{end+1} = w;
            enhCfgs{end+1} = c;
        end
        
        % Record metadata for this subband
        meta = struct();
        meta.type = sbType;
        meta.level = k;
        meta.origSize = sbPadInfo;
        meta.numBlocks = numel(sbBlocks);
        subbandMeta(end+1) = meta;
    end
end

txWaves.enh = enhWaves;
txInfo.numEnhBlocks = numel(enhWaves);
% fprintf('Total %d BASE blocks.\n', txInfo.numBaseBlocks);
% fprintf('Total %d ENH blocks.\n', txInfo.numEnhBlocks);
% Store Enh metadata
if ~isempty(enhCfgs)
    txInfo.cfgEnh = enhCfgs{1}; 
    fs = wlanSampleRate(enhCfgs{1});
end
txInfo.enhSubbandMeta = subbandMeta; % Detailed reconstruction info

% Global Metadata
txInfo.level = level;
txInfo.blk_size = BLK_SIZE;
txInfo.mux_gap_us = policy.mux_gap_us;
txInfo.frameId = frameId;
txInfo.policy = policy;
txInfo.DWTInfo = info;
txInfo.fs = fs;
txInfo.Tx_mode = strcmp(policy.tx_mode, 'split'); 
end
