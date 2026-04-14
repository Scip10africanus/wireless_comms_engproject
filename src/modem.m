% function [waveTx, cfg] = modem(pkt_bytes, mcs)
%   bytes = uint8(pkt_bytes(:));  % column bytes
%   bits  = reshape(de2bi(bytes, 8, 'left-msb').', [], 1);  % column bits (0/1)
%   bits  = int8(bits);  % wlanWaveformGenerator expects int8 or double
% 
%    cfg = wlanNonHTConfig;
%    cfg.ChannelBandwidth = 'CBW20';
%    cfg.MCS = double(mcs);
%    cfg.PSDULength = double(numel(bytes));  % BYTES, must be 1..4095
% 
%    waveTx = wlanWaveformGenerator(bits, cfg, 'IdleTime', 0, 'NumPackets', 1);
% end
function [waveTx, cfg] = modem(pkt_bytes, mcs)
% MODEM  Custom OFDM transmitter following the pipeline of reference OFDM.m
%
%   Pipeline (mirrors OFDM.m in order):
%     bytes -> bits -> [conv code] -> M-PSK/QAM modulate
%           -> [spread] -> insert pilots into freq grid
%           -> IFFT per OFDM symbol -> add CP -> serialize
%
%   Output:
%     waveTx : complex column vector (time-domain OFDM samples)
%     cfg    : struct containing everything demodem needs to reverse
 
    p = ofdm_params();
    bytes = uint8(pkt_bytes(:));
 
    % --- Select MCS ---
    mcs = max(0, min(mcs, length(p.mcs_table) - 1));
    bits_per_sym = p.mcs_table(mcs + 1).bits_per_sym;
    M            = p.mcs_table(mcs + 1).M;
    modName      = p.mcs_table(mcs + 1).mod;
 
    % ==================== 1. Bits ====================
    srcBits = reshape(de2bi(bytes, 8, 'left-msb').', [], 1);
    srcBits = double(srcBits);
    nSrcBits = numel(srcBits);
 
    % ==================== 2. Convolutional Encoding ====================
    %   Same (2,1,7) encoder as reference OFDM.m, rate 1/2
    if p.useConvCode
        trellis = poly2trellis(p.convConstraint, p.convPoly);
        codedBits = convenc(srcBits, trellis);
    else
        codedBits = srcBits;
    end
    nCodedBits = numel(codedBits);
 
    % ==================== 3. Modulation ====================
    % Pad bits to fill complete modulation symbols
    padBits = mod(bits_per_sym - mod(nCodedBits, bits_per_sym), bits_per_sym);
    codedBitsP = [codedBits; zeros(padBits, 1)];
 
    symInts = bi2de(reshape(codedBitsP, bits_per_sym, []).', 'left-msb');
 
    switch modName
        case 'BPSK'
            modSyms = pskmod(symInts, 2, 0);
        case 'QPSK'
            modSyms = pskmod(symInts, 4, pi/4, 'gray');
        case '16QAM'
            modSyms = qammod(symInts, 16, 'gray', 'UnitAveragePower', true);
        case '64QAM'
            modSyms = qammod(symInts, 64, 'gray', 'UnitAveragePower', true);
    end
 
    % ==================== 4. Spreading (optional, off by default) ====================
    %   Reference OFDM.m spreads each symbol across an m-sequence of length N_sc.
    %   Kept here as a hook; with spread_factor = 1 this is a no-op.
    if p.spread_factor > 1
        sf = p.spread_factor;
        code = 2 * (randi([0 1], sf, 1)) - 1;   % simple +/-1 sequence
        modSyms = reshape(modSyms, 1, []);
        spreadMat = code * modSyms;              % sf x nSyms
        modSyms = spreadMat(:);                  % serialized
    end
    nModSyms = numel(modSyms);
 
    % ==================== 5. Pack into OFDM symbols ====================
    nDataPerOFDM = p.N_data_per_sym;
    nOFDMSyms = ceil(nModSyms / nDataPerOFDM);
 
    padSyms = nOFDMSyms * nDataPerOFDM - nModSyms;
    modSyms = [modSyms; zeros(padSyms, 1)];
 
    dataMat = reshape(modSyms, nDataPerOFDM, nOFDMSyms);
 
    % ==================== 6. Insert pilots into frequency grid ====================
    freqGrid = zeros(p.N_fft, nOFDMSyms);
    freqGrid(p.pilot_pos, :) = p.pilot_value;       % constant pilot value
    freqGrid(p.data_pos,  :) = dataMat;
 
    % ==================== 7. IFFT + CP ====================
    timeGrid   = ifft(freqGrid, p.N_fft);           % per-column IFFT
    timeWithCP = [timeGrid(end - p.N_cp + 1 : end, :); timeGrid];  % prepend CP
 
    waveTx = timeWithCP(:);                          % serialize to column vector
 
    % ==================== 8. cfg for demodem ====================
    cfg = struct();
    cfg.mcs          = mcs;
    cfg.bits_per_sym = bits_per_sym;
    cfg.M            = M;
    cfg.modName      = modName;
    cfg.nOFDMSyms    = nOFDMSyms;
    cfg.nSrcBits     = nSrcBits;       % original bit count (pre-FEC)
    cfg.nCodedBits   = nCodedBits;     % coded bit count (post-FEC)
    cfg.nModSyms     = nModSyms;       % symbol count (post-spread)
    cfg.nDataBytes   = numel(bytes);   % original byte count
    cfg.padBits      = padBits;
    cfg.padSyms      = padSyms;
    cfg.params       = p;              % full params snapshot
end