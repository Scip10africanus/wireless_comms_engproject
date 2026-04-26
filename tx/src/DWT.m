function [res, info] = DWT(img, info, level)
    img = double(img);
    subbands = struct('LL', {}, 'LH', {}, 'HL', {}, 'HH', {}, 'inputSize', {}); % store subbands at each level
    currentLL = img;
    for k = 1:level
        [LL, LH, HL, HH] = dwt2(currentLL, 'haar', 'mode', 'sym'); % using symmetric padding
        
        subbands(k).LL = LL;
        subbands(k).LH = LH;
        subbands(k).HL = HL;
        subbands(k).HH = HH;
        subbands(k).inputSize = size(currentLL);

        currentLL = LL;
    end

    res.finalLL = currentLL;
    res.subbands = subbands;
    
    % store information about wavelet transform
    info.wavelet = 'haar'; 
    info.level = level;
    info.input_size = size(img);
    info.final_LL_size = size(currentLL);
end