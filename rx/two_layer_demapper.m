function [baseBits, enhBits, demapInfo] = two_layer_demapper(rxSignal, ofdmInfo)
% two_layer_demapper - thin wrapper around ofdm_demod, just reformats the output

    [baseBits, enhBits, headerParams, demodInfo] = ofdm_demod(rxSignal, ofdmInfo);

    demapInfo = struct();
    demapInfo.baseEVM = demodInfo.baseEVM;
    demapInfo.enhEVM = demodInfo.enhEVM;
    demapInfo.headerValid = demodInfo.headerValid;
    demapInfo.headerParams = headerParams;
    demapInfo.nOFDMSymbols = demodInfo.nOFDMSymbols;
end
