function [finalImage, fusionInfo] = layer_fusion(baseImage, residual, params)
% LAYER_FUSION - Combine base layer and enhancement residual
%
% Final reconstruction: I_hat = I_base + R_hat
%
% Handles various degradation scenarios:
%   - Both layers OK: Full quality reconstruction
%   - Base OK, Enhancement lost/corrupted: Blurry but usable image
%   - Base lost: Unusable (enhancement alone is meaningless)
%
% =========================================================================
% TODO: Tune fusion parameters based on actual system performance:
%   - Residual scaling factor
%   - Clipping strategy
%   - Graceful degradation weights
% =========================================================================
%
% Inputs:
%   baseImage  - Decoded base layer (double)
%   residual   - Decoded enhancement residual (double, can be negative)
%   params     - Structure with:
%       .baseValid       - true if base layer decoded successfully
%       .enhancementValid- true if enhancement decoded successfully
%       .residualScale   - Scale factor for residual (default: 1.0)
%       .clipOutput      - Clip to [0, 255] (default: true)
%       .outputType      - 'uint8' or 'double' (default: 'uint8')
%       .baseWeight      - Weight for base in degraded mode (default: 1.0)
%       .enhanceWeight   - Weight for enhancement (default: 1.0)
%
% Outputs:
%   finalImage - Reconstructed image
%   fusionInfo - Structure with fusion statistics and quality indicators

    %% Default parameters
    if ~isfield(params, 'baseValid')
        params.baseValid = true;
    end
    if ~isfield(params, 'enhancementValid')
        params.enhancementValid = true;
    end
    if ~isfield(params, 'residualScale')
        params.residualScale = 1.0;
    end
    if ~isfield(params, 'clipOutput')
        params.clipOutput = true;
    end
    if ~isfield(params, 'outputType')
        params.outputType = 'uint8';
    end
    if ~isfield(params, 'baseWeight')
        params.baseWeight = 1.0;
    end
    if ~isfield(params, 'enhanceWeight')
        params.enhanceWeight = 1.0;
    end

    %% Initialize fusion info
    fusionInfo = struct();
    fusionInfo.baseValid = params.baseValid;
    fusionInfo.enhancementValid = params.enhancementValid;
    fusionInfo.fusionMode = 'unknown';

    %% Determine fusion mode based on layer validity
    if params.baseValid && params.enhancementValid
        % Normal mode: Full reconstruction
        fusionInfo.fusionMode = 'full';
        finalImage = fuse_full(baseImage, residual, params);

    elseif params.baseValid && ~params.enhancementValid
        % Graceful degradation: Base only (blurry but usable)
        fusionInfo.fusionMode = 'base_only';
        finalImage = fuse_base_only(baseImage, params);
        fusionInfo.warning = 'Enhancement layer lost - using base only';

    elseif ~params.baseValid && params.enhancementValid
        % Critical failure: Enhancement without base is unusable
        fusionInfo.fusionMode = 'failed';
        fusionInfo.error = 'Base layer lost - cannot reconstruct';
        % Return enhancement as-is (will be garbage but allows analysis)
        finalImage = residual;
        fusionInfo.warning = 'BASE LAYER CRITICAL - image unusable';

    else
        % Complete failure
        fusionInfo.fusionMode = 'failed';
        fusionInfo.error = 'Both layers lost';
        finalImage = zeros(size(baseImage));
    end

    %% Clip and convert output
    if params.clipOutput
        finalImage = max(0, min(255, finalImage));
    end

    if strcmp(params.outputType, 'uint8')
        finalImage = uint8(round(finalImage));
    end

    %% Calculate fusion statistics
    fusionInfo = calculate_fusion_stats(fusionInfo, baseImage, residual, finalImage, params);
end

%% Fusion Functions

function img = fuse_full(baseImage, residual, params)
% FUSE_FULL - Normal fusion: base + scaled residual

    % Scale residual
    scaledResidual = residual * params.residualScale;

    % Apply weights
    img = params.baseWeight * baseImage + params.enhanceWeight * scaledResidual;
end

function img = fuse_base_only(baseImage, params)
% FUSE_BASE_ONLY - Graceful degradation using only base layer
%
% Optionally apply slight sharpening to partially compensate for
% missing enhancement layer

    img = params.baseWeight * baseImage;

    % Optional: Apply mild unsharp mask to recover some edges
    if isfield(params, 'compensateEnhancement') && params.compensateEnhancement
        img = compensate_missing_enhancement(img);
    end
end

function img = compensate_missing_enhancement(img)
% COMPENSATE_MISSING_ENHANCEMENT - Mild sharpening for base-only mode
%
% This is a simple attempt to recover some edge detail when
% enhancement layer is lost. Results will be inferior to full fusion.

    % Simple unsharp mask
    blurred = imgaussfilt_simple(img, 1.0);
    img = img + 0.3 * (img - blurred);  % Mild sharpening
end

function blurred = imgaussfilt_simple(img, sigma)
% IMGAUSSFILT_SIMPLE - Simple Gaussian blur

    kernelSize = ceil(sigma * 3) * 2 + 1;
    [X, Y] = meshgrid(-(kernelSize-1)/2:(kernelSize-1)/2);
    kernel = exp(-(X.^2 + Y.^2) / (2*sigma^2));
    kernel = kernel / sum(kernel(:));

    if ndims(img) == 3
        blurred = zeros(size(img));
        for ch = 1:size(img, 3)
            blurred(:,:,ch) = conv2(img(:,:,ch), kernel, 'same');
        end
    else
        blurred = conv2(img, kernel, 'same');
    end
end

%% Statistics Functions

function info = calculate_fusion_stats(info, baseImage, residual, finalImage, params)
% CALCULATE_FUSION_STATS - Compute fusion quality indicators

    % Base layer contribution
    info.baseMean = mean(baseImage(:));
    info.baseStd = std(baseImage(:));
    info.baseEnergy = sum(baseImage(:).^2);

    % Residual contribution
    if params.enhancementValid
        info.residualMean = mean(residual(:));
        info.residualStd = std(residual(:));
        info.residualEnergy = sum(residual(:).^2);

        % Energy ratio (how much detail the residual adds)
        totalEnergy = info.baseEnergy + info.residualEnergy;
        info.residualEnergyRatio = info.residualEnergy / (totalEnergy + eps);
    else
        info.residualMean = NaN;
        info.residualStd = NaN;
        info.residualEnergy = NaN;
        info.residualEnergyRatio = 0;
    end

    % Final image stats
    info.finalMean = mean(double(finalImage(:)));
    info.finalStd = std(double(finalImage(:)));
    info.finalMin = min(double(finalImage(:)));
    info.finalMax = max(double(finalImage(:)));

    % Clipping indicator (how much was clipped)
    if params.clipOutput
        preClip = params.baseWeight * baseImage;
        if params.enhancementValid
            preClip = preClip + params.enhanceWeight * residual * params.residualScale;
        end
        clippedLow = sum(preClip(:) < 0);
        clippedHigh = sum(preClip(:) > 255);
        info.pixelsClipped = clippedLow + clippedHigh;
        info.clipRatio = info.pixelsClipped / numel(preClip);
    end

    % Quality tier estimation
    info.qualityTier = estimate_quality_tier(info, params);
end

function tier = estimate_quality_tier(info, params)
% ESTIMATE_QUALITY_TIER - Estimate output quality level
%
% Returns: 'full', 'good', 'degraded', 'poor', 'failed'

    if strcmp(info.fusionMode, 'failed')
        tier = 'failed';
    elseif strcmp(info.fusionMode, 'base_only')
        tier = 'degraded';
    elseif info.clipRatio > 0.1
        tier = 'poor';  % Too much clipping
    elseif info.residualEnergyRatio < 0.01
        tier = 'good';  % Very little enhancement needed (already good)
    else
        tier = 'full';
    end
end

%% Diagnostic Functions

function visualize_fusion(baseImage, residual, finalImage, info)
% VISUALIZE_FUSION - Display fusion components and result
%
% Call separately for debugging:
%   visualize_fusion(baseImage, residual, finalImage, fusionInfo)

    figure('Position', [100, 100, 1200, 400], 'Name', 'Layer Fusion');

    subplot(1, 4, 1);
    imshow(uint8(baseImage), []);
    title(sprintf('Base Layer\n(Mean: %.1f)', info.baseMean));

    subplot(1, 4, 2);
    % Residual visualization (centered at 128 for visibility)
    residualVis = residual + 128;
    imshow(uint8(residualVis), []);
    title(sprintf('Residual (+128)\n(Std: %.1f)', info.residualStd));

    subplot(1, 4, 3);
    imshow(finalImage, []);
    title(sprintf('Final Image\n(Mode: %s)', info.fusionMode));

    subplot(1, 4, 4);
    % Show absolute residual magnitude
    imshow(uint8(abs(residual) * 2), []);
    title(sprintf('|Residual| x2\n(Energy ratio: %.1f%%)', info.residualEnergyRatio * 100));

    sgtitle(sprintf('Fusion Quality Tier: %s', info.qualityTier));
end
