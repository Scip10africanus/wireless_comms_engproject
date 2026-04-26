function val = bold_psnr(ref, test)
% BOLD_PSNR  Toolbox-free PSNR for double [0,1] or uint8 images
    ref  = double(ref);
    test = double(test);
    % Auto-detect range
    if max(ref(:)) > 1.5
        peak = 255;
    else
        peak = 1.0;
    end
    mse_val = mean((ref(:) - test(:)).^2);
    if mse_val < 1e-12
        val = 100;
    else
        val = 10 * log10(peak^2 / mse_val);
    end
end
