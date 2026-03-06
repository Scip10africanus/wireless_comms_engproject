clear; clc; close all; % clear old variable console and close figure

currentDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(currentDir);
addpath(fullfile(rootDir, 'src'));
addpath(fullfile(rootDir, 'test_functions'));

N = 8;
K = 21;
imgPath = fullfile(rootDir, 'images', 'Testing examples.jpg');
img = imread(imgPath);
img = im2double(img);
if size(img,3) == 3 % transform into gray image
    grayimg = rgb2gray(img);
end
patch = grayimg(1:8, 1:8);   % 8×8 block
[DCT,info] = DCT_Block(patch,N);
[base,enhance] = Layer_Split(K,N,DCT);

disp("Base length / Enh length:");
disp([numel(base), numel(enhance)]);

figure; imshow(patch, []); title("8x8 patch");
figure; imagesc(DCT); axis image; colorbar; title("DCT coefficients (D)");

enh0 = zeros(size(enhance));
vec_baseonly = [base(:); enh0(:)];

D_base = zeros(N,N);
idx = Zigzag_index(N);
D_base(idx) = vec_baseonly;

rec_baseonly = IDCT(D_base,info);

figure; imshow(patch, []); title("Original 8x8 block");
figure; imshow(rec_baseonly,   []); title("Reconstructed (base-only, enh=0)");

base0 = zeros(size(base));
vec_enhanceonly = [base0(:); enhance(:)];

D_enhance = zeros(N,N);
idx = Zigzag_index(N);
D_enhance(idx) = vec_enhanceonly;

rec_baseonly = IDCT(D_enhance,info);

figure; imshow(rec_baseonly,   []); title("Reconstructed (enhance-only, base=0)");


vec = [base(:); enhance(:)];
D_full = zeros(N,N);
D_full(idx) = vec;
rec_whole = IDCT(D_full,info);
figure; imshow(rec_whole, []); title("Reconstructed (whole, base + enhance)");
psnr_val = psnr(rec_whole, patch);
ssim_val = ssim(rec_whole, patch);
fprintf("Identity test | PSNR=%.2f dB | SSIM=%.4f\n", psnr_val, ssim_val);
figure; imshow(abs(patch - rec_whole), []); title("Abs error map");
