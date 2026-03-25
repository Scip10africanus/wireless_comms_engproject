function [finalImage, fusionInfo] = layer_fusion(baseImage, residual, mode)
% layer_fusion - add the DCT residual back onto the wavelet base image
% mode 'full': output = base + residual
% mode 'base_only': just use base (when enhancement was dropped)

    if nargin < 3, mode = 'full'; end

    fusionInfo = struct();
    fusionInfo.mode = mode;

    baseDouble = double(baseImage);

    switch mode
        case 'full'
            if isempty(residual)
                fusionInfo.mode = 'base_only';
                fusionInfo.warning = 'No residual provided, using base only';
                combined = baseDouble;
            else
                % Crop to common size
                [h1, w1] = size(baseDouble);
                [h2, w2] = size(residual);
                h = min(h1, h2);
                w = min(w1, w2);
                combined = baseDouble(1:h, 1:w) + residual(1:h, 1:w);
            end

        case 'base_only'
            combined = baseDouble;
            fusionInfo.warning = 'Enhancement layer dropped (graceful degradation)';

        otherwise
            combined = baseDouble;
            fusionInfo.warning = sprintf('Unknown mode: %s, using base only', mode);
    end

    finalImage = uint8(max(0, min(255, round(combined))));
    fusionInfo.outputSize = size(finalImage);
end
