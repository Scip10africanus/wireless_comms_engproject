clear; clc; close all; % clear old variable console and close figure

currentDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(currentDir);
addpath(fullfile(rootDir, 'src'));
addpath(fullfile(rootDir, 'test_functions'));

Qb_list = [0.01 0.02 0.05 0.1];
Qe_list = [0.01 0.02 0.05 0.1];
controller = PolicyController(fullfile(rootDir, "policy.csv"));
frame_id = 0;
policy = controller.tick(frame_id);
phy    = controller.makePhy(); 

assert(phy.tx_mode == "split", "For two-layer test, please set tx_mode=split in policy.");

N = 8;
K = double(policy.K);

Qb = Qb_list(double(policy.Qb_idx)+1);
Qe = Qe_list(double(policy.Qe_idx)+1);

fprintf("Using: K=%d, Qb_idx=%d(Qb=%g), Qe_idx=%d(Qe=%g), MCSb=%d, MCSe=%d, gap=%gus\n", ...
    K, policy.Qb_idx, Qb, policy.Qe_idx, Qe, phy.mcs_base, phy.mcs_enh, phy.mux_gap_us);


imgPath = fullfile(rootDir, 'images', 'Testing examples.jpg');
img = imread(imgPath);
img = im2double(img);

if size(img,3) == 3
    grayimg = rgb2gray(img);
else
    grayimg = img;
end

patch = grayimg(1:N, 1:N);  % 8x8 block
[dct_patch, info] = DCT_Block(patch, N);

[base, enh] = Layer_Split(K, N, dct_patch);

[base_q, enh_q, enh_cnt, clip_cnt] = quant(base, enh, Qb, Qe);
fprintf("Enh coeff count: %d, clipped: %d\n", enh_cnt, clip_cnt);

block_id = uint16(0);

paramsB = struct();
paramsB.Qb_idx = uint8(policy.Qb_idx);
paramsB.Qe_idx = uint8(policy.Qe_idx);
paramsB.K      = uint8(policy.K);
paramsB.N      = uint8(N);
paramsB.codec_id = uint8(policy.codec_id);
paramsB.crc_flag = uint8(policy.crc_flag);
paramsB.endian_flag = uint8(policy.endian_flag);
paramsB.reserved_flag = uint8(policy.reserved_flag);
paramsB.packet_mode = uint8(1);       
paramsB.layer_type  = uint8(1); 
paramsB.frame_id = uint8(frame_id);
paramsB.block_id = uint16(block_id);


paramsE = struct();
paramsE.Qb_idx = uint8(policy.Qb_idx);
paramsE.Qe_idx = uint8(policy.Qe_idx);
paramsE.K      = uint8(policy.K);
paramsE.N      = uint8(N);
paramsE.codec_id = uint8(policy.codec_id);
paramsE.crc_flag = uint8(policy.crc_flag);
paramsE.endian_flag = uint8(policy.endian_flag);
paramsE.reserved_flag = uint8(policy.reserved_flag);
paramsE.packet_mode = uint8(1);
paramsE.layer_type  = uint8(3); 
paramsE.frame_id = uint8(frame_id);
paramsE.block_id = uint16(block_id);

pkt_base = bitStream_packet(paramsB, base_q, int8([]));
pkt_enh  = bitStream_packet(paramsE, int16([]), enh_q);

[waveB, cfgB] = modem(pkt_base, phy.mcs_base);
[waveE, cfgE] = modem(pkt_enh,  phy.mcs_enh);

fs = wlanSampleRate(cfgB);

txWave = layer_mux(waveB, waveE, phy.mux_gap_us, fs);
rxWave = txWave;


[rxB, rxE] = layer_demux_bylen(rxWave, numel(waveB), numel(waveE), phy.mux_gap_us, fs);

rx_pkt_base = demodem(rxB, cfgB);
rx_pkt_enh  = demodem(rxE, cfgE);

assert_bytes_equal("BASE", pkt_base, rx_pkt_base);
assert_bytes_equal("ENH ", pkt_enh,  rx_pkt_enh);

outB = bitStream_parsing(rx_pkt_base);
outE = bitStream_parsing(rx_pkt_enh);

assert(isequal(outB.base_coeff_int16(:).', base_q(:).'), "Base coeff mismatch after parsing.");
assert(isequal(outE.enh_coeff_int8(:).',  enh_q(:).'),  "Enh coeff mismatch after parsing.");
disp("Parsing + coeff round-trip OK.");

[base_hat, enh_hat] = dequant(outB.base_coeff_int16, outE.enh_coeff_int8, Qb, Qe);

idx = Zigzag_index(N);
vec_hat = zeros(N*N,1);
vec_hat(1:K) = base_hat(:);
vec_hat(K+1:end) = enh_hat(:);

blk_hat = zeros(N,N);
blk_hat(idx) = vec_hat; % inverse of Layer_Split

dct_hat = blk_hat;

patch_hat = IDCT(dct_hat, info);


mse = mean((patch_hat(:) - patch(:)).^2);
ssim_val = ssim(patch_hat, patch);
psnr_val = psnr(patch_hat, patch);
fprintf("MSE=%.6f, PSNR=%.3f dB , SSIM = %.6f \n", mse, psnr_val,ssim_val);

figure;
subplot(1,2,1); imshow(patch, []); title("Original patch");
subplot(1,2,2); imshow(min(max(patch_hat,0),255)); title("Reconstructed");


function assert_bytes_equal(tag, a, b)
    assert(numel(a)==numel(b), "%s length mismatch: %d vs %d", tag, numel(a), numel(b));
    k = find(a~=b, 1, 'first');
    assert(isempty(k), "%s first mismatch at %d (tx=%d rx=%d)", tag, k, a(k), b(k));
    fprintf("%s OK (%d bytes)\n", tag, numel(a));
end