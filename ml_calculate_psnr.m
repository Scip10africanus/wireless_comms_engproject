function psnr_val = ml_calculate_psnr(ref, test)
% ML_CALCULATE_PSNR - Simple PSNR calculation for ML training data
    mse = mean((double(ref(:)) - double(test(:))).^2);
    if mse == 0
        psnr_val = 100;
    else
        psnr_val = 10*log10(255^2/mse);
    end
end
