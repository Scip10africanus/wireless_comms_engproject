clear; clc; close all;  % clear old variables, console, and close figures
addpath(fullfile(pwd, 'src'));  % protocol Tx/Rx functions live here
addpath(pwd);                  % make sure current folder (your DCT/quant funcs) is visible too

N = 8;
K = 21;

Qb_list = [0.01 0.02 0.05 0.1];         
Qe_list = [0.01 0.02 0.05 0.1];

qb_i = 1;   % MATLAB index: 1..length(Qb_list)
qe_i = 1;

Qb = Qb_list(qb_i);
Qe = Qe_list(qe_i);

img = imread(fullfile("images", "Testing examples.jpg"));
img = im2double(img);

if size(img,3) == 3
    grayimg = rgb2gray(img);
else
    grayimg = img;
end

patch = grayimg(1:N, 1:N);  % 8x8 block

[dct_block, info] = DCT_Block(patch, N);

idx = Zigzag_index(N);               % your function
vec = dct_block(idx);                % zigzag vector (64x1)
[base, enh] = Layer_Split(K, N, dct_block);

[base_q, enh_q, enh_cnt, clip_cnt] = quant(base, enh, Qb, Qe);

fprintf("Enhance coeff count: %d, clipped: %d\n", enh_cnt, clip_cnt);

params = struct();
params.Qb_idx = uint8(qb_i - 1);              % example index (you can map idx->actual Qb table later)
params.Qe_idx = uint8(qe_i - 1);
params.K = uint8(K);
params.N = uint8(N);
params.codec_id = uint8(0);     % 0=DCT 
params.packet_mode = uint8(0);  % 0 => combined
params.crc_flag = uint8(1);     % enable CRC
params.endian_flag = uint8(0);  % 0 little-endian for base payload int16
params.layer_type = uint8(0);   % ignored in combined mode
params.reserved_flag = uint8(0);
params.frame_id = uint8(1);
params.block_id = uint16(1);

base_q = int16(base_q);
enh_q  = int8(enh_q);

pkt = bitStream_packet(params, base_q, enh_q);
out = bitStream_parsing(pkt);

assert(isequal(int16(out.base_coeff_int16(:)), base_q(:)), "Base payload mismatch after Rx");
assert(isequal(int8(out.enh_coeff_int8(:)),  enh_q(:)),  "Enh payload mismatch after Rx");

qb_i_rx = double(out.params.Qb_idx) + 1;
qe_i_rx = double(out.params.Qe_idx) + 1;

[base_hat, enh_hat] = dequant(out.base_coeff_int16, out.enh_coeff_int8, Qb_list(qb_i_rx), Qe_list(qe_i_rx));

dct_recon = inv_layer_merge(base_hat, enh_hat, idx, N);

patch_rec = IDCT(dct_recon, info);

mse = mean((patch(:) - patch_rec(:)).^2);
psnr_val = 10*log10(1/mse);

fprintf("Patch MSE = %.6g, PSNR = %.2f dB\n", mse, psnr_val);
ssim_val = ssim(patch_rec, patch);
psnr_val = psnr(patch_rec, patch);
fprintf("Identity test | PSNR=%.2f dB | SSIM=%.4f\n", psnr_val, ssim_val);
figure;
subplot(1,3,1); imshow(patch, []); title("Original patch");
subplot(1,3,2); imshow(patch_rec, []); title("Reconstructed patch");
subplot(1,3,3); imagesc(patch - patch_rec); axis image; colorbar; title("Error map");

function block = inv_layer_merge(base_hat, enh_hat, zigzag_idx, N)
    v = [base_hat(:); enh_hat(:)];
    assert(numel(v) == N*N, "Merged coeff length must be N*N");

    block = zeros(N,N);
    block(zigzag_idx) = v;
end
