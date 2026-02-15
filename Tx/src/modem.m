function [waveTx, cfg] = modem(pkt_bytes, mcs)
% WIFI_NONHT_MOD 802.11a/g Non-HT modulation for a PSDU=uint8 bytes
% pkt_bytes: uint8 row/col
% mcs: 0..7
    bytes = uint8(pkt_bytes(:));          % column bytes
    bits  = reshape(de2bi(bytes, 8, 'left-msb').', [], 1);  % column bits (0/1), MSB first
    bits  = int8(bits);                   % WLAN toolbox accepts int8

    cfg = wlanNonHTConfig;
    info = wlanNonHTOFDMInfo('NonHT-Data', cfg);
    disp(info);
    cfg.ChannelBandwidth = 'CBW20';
    cfg.MCS = double(mcs);
    cfg.PSDULength = double(numel(bytes));
    waveTx = wlanWaveformGenerator(bits, cfg, 'IdleTime', 0, 'NumPackets', 1);
end
