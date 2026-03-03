clear; clc; close all
addpath(fullfile(pwd,'src'));
N = 8;
K = 21;
img = imread("images\Testing examples.jpg");
img = im2double(img);
if size(img,3)==3
    img = rgb2gray(img); 
end
idx = Zigzag_index(N);


Qb_list = [0.01 0.02 0.05 0.1];
Qe_list = [0.01 0.02 0.05 0.1];

results = [];

for Qb = Qb_list
    for Qe = Qe_list
        [D, info] = DCT_Block(img, N);

        [H,W] = size(D);
        D_hat = zeros(size(D));

        sum_base_abs = 0; sum_enh_abs = 0;
        max_baseq = 0; max_enhanceq = 0;
        cnt_base = 0; cnt_enh = 0;
        enh_zero_cnt = 0;

    for r = 1:N:H
        for c = 1:N:W
            Db = D(r:r+N-1, c:c+N-1);

            vec = Db(idx);
            base = vec(1:K);
            enh  = vec(K+1:end);

            [base_q, enh_q, enh_cnt, clip_cnt] = quant(base, enh, Qb, Qe); 
            [base_hat, enh_hat] = dequant(base_q, enh_q, Qb, Qe);

            ab = abs(double(base_q));
            ae = abs(double(enh_q));

            sum_base_abs = sum_base_abs + sum(ab);
            sum_enh_abs  = sum_enh_abs  + sum(ae);
    
            max_baseq     = max(max_baseq, max(ab));
            max_enhanceq  = max(max_enhanceq, max(ae));

            cnt_base = cnt_base + numel(base_q);
            cnt_enh  = cnt_enh  + numel(enh_q);

            enh_zero_cnt = enh_zero_cnt + sum(enh_q==0);

            vec_hat = [base_hat(:); enh_hat(:)];
            Db_hat = zeros(N,N);
            Db_hat(idx) = vec_hat;
            D_hat(r:r+N-1, c:c+N-1) = Db_hat;
        end
    end

    mean_baseq    = sum_base_abs / cnt_base;
    mean_enhanceq = sum_enh_abs  / cnt_enh;
    zero_ratio_enh = enh_zero_cnt / cnt_enh;

    rec = IDCT(D_hat, info);

    mse = mean((rec(:)-img(:)).^2);
    psnr_val = psnr(rec, img);
    ssim_val = ssim(rec, img);
    max_abs_err = max(abs(rec(:)-img(:)));

    results = [results;Qb Qe mean_baseq mean_enhanceq max_baseq max_enhanceq zero_ratio_enh psnr_val ssim_val max_abs_err];

        fprintf("Qb=%-2g Qe=%-2g | PSNR=%6.2f dB | SSIM=%7.4f\n", ...
            Qb, Qe, psnr_val, ssim_val);
    end
end

[~,ord] = sort(results(:,3), 'descend');
sorted = results(ord,:);

disp("Top 10 by PSNR: [Qb Qe mean_baseq mean_enhanceq max_baseq max_enhanceq zero_ratio_enh psnr_val ssim_val max_abs_err]");
disp(sorted(1:min(10,size(sorted,1)), :));


targets = Qe_list;
profiles = zeros(numel(targets), size(results,2));

for i = 1:numel(targets)
    tqe = targets(i);
    candidates = results(abs(results(:,2)-tqe) < 1e-12, :);
    [~,j] = max(candidates(:,8));  % PSNR largest
    profiles(i,:) = candidates(j,:);
end

disp("Suggested 4 profiles (by best PSNR at target Qe):");
disp(" [Qb Qe mean_baseq mean_enhanceq max_baseq max_enhanceq zero_ratio_enh psnr_val ssim_val max_abs_err]");
disp(profiles);

