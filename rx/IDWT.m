function recon = IDWT(res, level)
    currentLL = res.finalLL;

    for k = level:-1:1
        LH = res.subbands(k).LH;
        HL = res.subbands(k).HL;
        HH = res.subbands(k).HH;
        inputSize = res.subbands(k).inputSize;

        % inverse one DWT level
        reconLevel = idwt2(currentLL, LH, HL, HH, 'haar', inputSize, 'mode', 'sym');

        % pass reconstructed result outward as next level LL
        currentLL = reconLevel;
    end

    recon = currentLL;
end