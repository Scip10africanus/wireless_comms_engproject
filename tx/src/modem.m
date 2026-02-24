function [waveTx, cfg] = modem(pkt_bytes, mcs)
    bytes = uint8(pkt_bytes(:));  % column bytes
    bits  = reshape(de2bi(bytes, 8, 'left-msb').', [], 1);  % column bits (0/1)
    bits  = int8(bits);  % wlanWaveformGenerator expects int8 or double

    cfg = wlanNonHTConfig;
    cfg.ChannelBandwidth = 'CBW20';
    cfg.MCS = double(mcs);
    cfg.PSDULength = double(numel(bytes));  % BYTES, must be 1..4095

    waveTx = wlanWaveformGenerator(bits, cfg, 'IdleTime', 0, 'NumPackets', 1);
end