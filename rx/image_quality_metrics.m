function quality = image_quality_metrics(original, reconstructed)
% image_quality_metrics - compute PSNR, SSIM and MSE between two images
% uses global (non-windowed) SSIM so it works in Octave without the image toolbox

    quality = struct('hasMetrics', false, 'ssim', NaN, 'psnr', NaN, 'mse', NaN);

    if isempty(original) || isempty(reconstructed)
        return;
    end

    orig = double(original);
    rec = double(reconstructed);

    % Crop to common size
    [h1, w1] = size(orig);
    [h2, w2] = size(rec);
    h = min(h1, h2);
    w = min(w1, w2);
    orig = orig(1:h, 1:w);
    rec = rec(1:h, 1:w);

    % MSE
    quality.mse = mean((orig(:) - rec(:)).^2);

    % PSNR
    quality.psnr = 10 * log10(255^2 / (quality.mse + eps));

    % SSIM (global, not windowed — Octave-compatible)
    C1 = (0.01 * 255)^2;
    C2 = (0.03 * 255)^2;
    mu1 = mean(orig(:));
    mu2 = mean(rec(:));
    s1 = var(orig(:));
    s2 = var(rec(:));
    s12 = mean((orig(:) - mu1) .* (rec(:) - mu2));
    quality.ssim = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1^2 + mu2^2 + C1) * (s1 + s2 + C2));

    quality.hasMetrics = true;
end
