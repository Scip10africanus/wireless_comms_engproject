clear; clc; close all; % clear old variable console and close figure
% Set up paths relative to this file
currentDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(currentDir);
addpath(fullfile(rootDir, 'src'));
addpath(fullfile(rootDir, 'test_functions'));

block_length = 8;
imgPath = fullfile(rootDir, 'images', 'Testing examples.jpg');
img = imread(imgPath);
img = im2double(img);
if size(img,3) == 3 % transform into gray image
    grayimg = rgb2gray(img);
end
[res, info] = DCT_Block(grayimg, 8); % using block struct for DCT-II function to processing data
recon = IDCT(res, info);
psnr_val = psnr(recon, grayimg);
ssim_val = ssim(recon, grayimg);
fprintf("Identity test | PSNR=%.2f dB | SSIM=%.4f\n", psnr_val, ssim_val);
figure; imshow(grayimg, []);  title("Original gray image");
figure; imshow(recon, []); title("Reconstructed gray image");
figure; imshow(abs(grayimg - recon), []); title("Abs error map");
