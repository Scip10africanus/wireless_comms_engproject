function [DCTcoe] = two_layer_rx_v2(rxWave, txInfo, settings)
% TWO_LAYER_RX_V2  Receiver for new TX pipeline
% rxWave    - received waveform (complex column)
% txInfo    - metadata struct from TX (lenBaseWave, cfgBase, policy, etc.)
% settings  - struct with .originalImage, .policyCsvPath, .useOctaveMode, etc.


    if nargin < 3, settings = struct(); end
    if ~isfield(settings, 'policyCsvPath'),      settings.policyCsvPath = ''; end
    if ~isfield(settings, 'verbose'),            settings.verbose = true; end
    if ~isfield(settings, 'useOctaveMode'),      settings.useOctaveMode = true; end
    if ~isfield(settings, 'useGaOptimizer'),     settings.useGaOptimizer = false; end
    if ~isfield(settings, 'gaSettings'),         settings.gaSettings = struct(); end
    if ~isfield(settings, 'useChannelPredictor'),settings.useChannelPredictor = false; end
    if ~isfield(settings, 'predictorParams'),    settings.predictorParams = struct(); end
    if ~isfield(settings, 'targetSsim'),         settings.targetSsim = 0.9; end
    if ~isfield(settings, 'targetPsnr'),         settings.targetPsnr = 30; end

    policy = txInfo.policy;
    K = double(policy.K);
    N = 8;

    Qb_list = [0.01 0.02 0.05 0.1];
    Qe_list = [0.01 0.02 0.05 0.1];
    
    if settings.verbose
        % fprintf('[RX] Policy v%d: K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d \n', ...
        %     policy.version, K, policy.Qb_idx, policy.Qe_idx, ...
        %     policy.mcs_base, policy.mcs_enh);
    end

    %% 1 - Demux base and enhancement waveforms
    fs = wlanSampleRate(txInfo.cfgBase);
    [rxBase, rxEnh] = layer_demux_bylen(rxWave, ...
        txInfo.lenBaseWave, txInfo.lenEnhWave, double(policy.mux_gap_us), fs);
    % fprintf("[DBG] rxWave=%d\n", numel(rxWave));
    % fprintf("[DBG] lenBaseWave=%d lenEnhWave=%d\n", txInfo.lenBaseWave, txInfo.lenEnhWave);
    % fprintf("[DBG] cfgBase.PSDULength=%d", txInfo.cfgBase.PSDULength);
    % fprintf("[DBG] rxBase=%d rxEnh=%d\n", numel(rxBase), numel(rxEnh));
    %% 2 - Demodulate
    basePktBytes = demodem(rxBase, txInfo.cfgBase);

    enhPktBytes = uint8([]);
    if ~isempty(rxEnh) && ~isempty(txInfo.cfgEnh)
        enhPktBytes = demodem(rxEnh, txInfo.cfgEnh);
    end
    %% 3 - Parse packets -> extract coefficients
    baseParsed = bitStream_parsing(basePktBytes);

    enhParsed = [];
    if ~isempty(enhPktBytes)
        enhParsed = bitStream_parsing(enhPktBytes);
    end


    %% 4 - Dequantize 
    zigzagIdx = Zigzag_index(N);

    % dequant 
    [baseHat, enhHat] = dequant(double(baseParsed.base_coeff_int16), double(enhParsed.enh_coeff_int8), Qb_list(baseParsed.params.Qb_idx + 1), Qe_list(enhParsed.params.Qe_idx + 1));
    baseHat = baseHat(:);
    enhHat  = enhHat(:);
    DCTcoe(zigzagIdx) = [baseHat; enhHat];
    DCTcoe = reshape(DCTcoe, 8, 8);
end

