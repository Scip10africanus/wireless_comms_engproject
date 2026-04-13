function feat = dwt_extract_features(res, txInfo, opts)
    if nargin < 3 || ~isstruct(opts), opts = struct(); end
    if ~isfield(opts,'eps'), opts.eps = 1e-12; end
    if ~isfield(opts,'includeDebug'), opts.includeDebug = false; end
    if ~isfield(opts,'featureVersion'), opts.featureVersion = uint32(1); end
    if ~isfield(opts,'wavelet'), opts.wavelet = ""; end
    if ~isfield(opts,'level'), opts.level = NaN; end

    e = double(opts.eps);

    E_LL = energy2(res.finalLL);
    E_HH = 0;
    E_total = E_LL;

    if isfield(res,'subbands') && ~isempty(res.subbands)
        for k = 1:numel(res.subbands)
            sb = res.subbands(k);
            if isfield(sb,'LH') && ~isempty(sb.LH)
                E_total = E_total + energy2(sb.LH);
            end
            if isfield(sb,'HL') && ~isempty(sb.HL)
                E_total = E_total + energy2(sb.HL);
            end
            if isfield(sb,'HH') && ~isempty(sb.HH)
                eh = energy2(sb.HH);
                E_HH = E_HH + eh;
                E_total = E_total + eh;
            end
        end
    end

    E_LL_ratio = E_LL / (E_total + e);
    E_HH_ratio = E_HH / (E_total + e);

    base_payload_bytes = NaN;
    enh_payload_bytes = NaN;
    base_share = NaN;
    if nargin >= 2 && isstruct(txInfo) ...
            && isfield(txInfo,'numBaseBlocks') && isfield(txInfo,'numEnhBlocks') && isfield(txInfo,'blk_size')
        blk = double(txInfo.blk_size);
        nB = double(txInfo.numBaseBlocks);
        nE = double(txInfo.numEnhBlocks);
        base_payload_bytes = nB * (blk*blk) * 2;
        enh_payload_bytes  = nE * (blk*blk) * 1;
        base_share = base_payload_bytes / (base_payload_bytes + enh_payload_bytes + e);
    end

    feat = struct();
    feat.feature_version = uint32(opts.featureVersion);
    feat.wavelet = string(opts.wavelet);
    feat.level = double(opts.level);
    feat.E_LL_ratio = E_LL_ratio;
    feat.E_HH_ratio = E_HH_ratio;
    feat.base_share = base_share;
    feat.base_share_is_model = true;

    if logical(opts.includeDebug)
        feat.E_LL = E_LL;
        feat.E_HH = E_HH;
        feat.E_total = E_total;
        feat.base_payload_bytes = base_payload_bytes;
        feat.enh_payload_bytes = enh_payload_bytes;
    end
end

function e = energy2(x)
    x = double(x);
    if isempty(x)
        e = 0;
    else
        e = sum(x(:).^2);
    end
end
