function results = bold_train_all(dataPath, opts)
% BOLD_TRAIN_ALL  Train all 3 ML models from generated data
%
% CHANGES v2 (Bold - Apr 2026):
%   - Quality predictor expanded: 12 -> 15 inputs (+ Zibo's DWT features)
%   - Loads and merges Zibo's DWT dataset (data.samples MATLAB table)
%     if dwt_training_data.mat exists in common/ folder
%   - Added z-score feature normalisation for inputs (mean/std saved
%     to quality_predictor_norm.mat alongside model weights)
%   - Architecture: 12->64->32->16->2  =>  15->64->32->16->2
%
% Bold (Unubold) - EEME30003 Major Project
%
% Usage:
%   results = bold_train_all();
%   results = bold_train_all('bold_training_data.mat');
%
% Produces:
%   quality_predictor.mat        (15-input MLP)
%   quality_predictor_norm.mat   (feat_mean, feat_std, psnr_mean, psnr_std)
%   channel_lstm.mat
%   symbol_denoiser_mod{4,16,64}.mat

    if nargin < 1 || isempty(dataPath)
        dataPath = fullfile(fileparts(mfilename('fullpath')), 'bold_training_data.mat');
    end
    if nargin < 2, opts = struct(); end
    if ~isfield(opts,'qualityEpochs'),  opts.qualityEpochs  = 120; end
    if ~isfield(opts,'lstmEpochs'),     opts.lstmEpochs     = 80;  end
    if ~isfield(opts,'denoiserEpochs'), opts.denoiserEpochs = 60;  end
    if ~isfield(opts,'verbose'),        opts.verbose         = true; end

    NFEAT    = 16;
    modelDir = fileparts(mfilename('fullpath'));
    rootDir  = fullfile(modelDir, '..');

    fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
    fprintf('║       Bold ML Training Pipeline v2 - 15 features              ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

    %% Load main training data
    if ~exist(dataPath, 'file')
        fprintf('Training data not found. Generating now...\n');
        bold_generate_training_data(struct('savePath', dataPath));
    end
    fprintf('Loading DCT training data: %s\n', dataPath);
    data = load(dataPath);
    results = struct();

    X_q_raw = data.quality_features;   % [12 x N_dct]  OR  [15 x N_dct]
    Y_q     = data.quality_targets;    % [2  x N_dct]  (SSIM, PSNR)
    N_dct   = size(X_q_raw, 2);
    nFeatRaw = size(X_q_raw, 1);

    % bold_quality_predictor in 'extract_features' mode now returns 15 features.
    % Older saved data may still have 12. Detect and handle both formats.
    if nFeatRaw == NFEAT
        X_q_dct = X_q_raw;
        fprintf('  Feature format: v3 (%d features — DCT+DWT+noiseVar)\n', NFEAT);
    elseif nFeatRaw == 15
        % v2 data (DCT+DWT, no noiseVar) — append log10(0.01) prior for feat 16
        noiseVar_prior = log10(0.01) * ones(1, N_dct);  % -2: 20 dB SNR prior
        X_q_dct = [X_q_raw; noiseVar_prior];
        fprintf('  Feature format: v2 (15 feat) — appending noiseVar prior to reach %d.\n', NFEAT);
    elseif nFeatRaw == 12
        % v1 data (DCT only) — append DWT priors + noiseVar prior
        X_q_dct = [X_q_raw; zeros(3, N_dct); log10(0.01)*ones(1, N_dct)];
        X_q_dct(13,:) = 0.50;
        X_q_dct(14,:) = 0.10;
        X_q_dct(15,:) = 0.50;
        fprintf('  Feature format: v1 (12 feat) — padding to %d with priors.\n', NFEAT);
    else
        error('Unexpected feature dimension: %d (expected 12, 15, or %d).', nFeatRaw, NFEAT);
    end

    %% Attempt to load Zibo's DWT dataset and merge
    %  Zibo's dataset has data.samples as a MATLAB table with columns:
    %  E_LL_ratio, E_HH_ratio, base_share_model, ssim, psnr, mse, ...
    dwtDataPath = fullfile(rootDir, 'common', 'dwt_training_data.mat');

    N_dwt_extra = 0;
    X_dwt_extra = [];
    Y_dwt_extra = [];

    if exist(dwtDataPath, 'file')
        fprintf('Loading DWT dataset from: %s\n', dwtDataPath);
        try
            dwt_raw = load(dwtDataPath);
            if isfield(dwt_raw,'data') && isstruct(dwt_raw.data) && isfield(dwt_raw.data,'samples')
                T = dwt_raw.data.samples;  % MATLAB table
            elseif isfield(dwt_raw,'samples')
                T = dwt_raw.samples;
            else
                T = [];
            end

            if ~isempty(T) && istable(T)
                vars = T.Properties.VariableNames;
                has = @(v) any(strcmp(vars, v));

                if has('E_LL_ratio') && has('E_HH_ratio') && has('ssim') && has('psnr')
                    N_dwt_extra = height(T);
                    X_dwt_extra = zeros(NFEAT, N_dwt_extra);

                    % Columns 1-12: DCT features not in Zibo's table — use priors
                    X_dwt_extra(1,:)  = 2.0;   % log10 energy prior (moderate)
                    X_dwt_extra(2,:)  = 0.5;   % base energy ratio prior
                    X_dwt_extra(3,:)  = 0.5;   % enh energy ratio prior
                    X_dwt_extra(4,:)  = 1.0;   % base CV prior
                    X_dwt_extra(5,:)  = 1.0;   % enh CV prior
                    X_dwt_extra(6,:)  = 0.0;   % DC mean prior
                    X_dwt_extra(7,:)  = 0.1;   % DC std prior
                    % Columns 8-12: fill from policy context if available
                    if has('snr_dB')
                        % rough mcs_enh proxy from SNR
                        snr_vals = table2array(T(:,'snr_dB'));
                        mcs_proxy = min(6, max(0, floor((snr_vals - 8) / 4)));
                        X_dwt_extra(11,:) = 0 / 7;                     % mcs_base default
                        X_dwt_extra(12,:) = mcs_proxy.' / 7;
                    end
                    if has('Qb_idx'), X_dwt_extra(9,:)  = table2array(T(:,'Qb_idx')).' / 3; end
                    if has('Qe_idx'), X_dwt_extra(10,:) = table2array(T(:,'Qe_idx')).' / 3; end

                    % Columns 13-15: DWT features from Zibo (ground truth)
                    X_dwt_extra(13,:) = table2array(T(:,'E_LL_ratio')).';
                    X_dwt_extra(14,:) = table2array(T(:,'E_HH_ratio')).';
                    if has('base_share_model')
                        X_dwt_extra(15,:) = table2array(T(:,'base_share_model')).';
                    else
                        X_dwt_extra(15,:) = 0.5;
                    end

                    % Labels
                    Y_dwt_extra = zeros(2, N_dwt_extra);
                    Y_dwt_extra(1,:) = table2array(T(:,'ssim')).';
                    Y_dwt_extra(2,:) = table2array(T(:,'psnr')).';

                    fprintf('  DWT dataset: %d samples loaded and merged.\n', N_dwt_extra);
                else
                    fprintf('  DWT table missing required columns — skipping DWT merge.\n');
                end
            end
        catch ME
            fprintf('  Warning: could not load DWT data (%s). Continuing without it.\n', ME.message);
        end
    else
        fprintf('DWT dataset not found at %s — using priors for DWT features.\n', dwtDataPath);
    end

    %% Assemble full feature matrix [15 x (N_dct + N_dwt)]
    % X_q_dct is already [15 x N_dct] (either native or padded above)
    if ~isempty(X_dwt_extra)
        X_q = [X_q_dct, X_dwt_extra];        % [15 x (N_dct + N_dwt)]
        Y_q = [Y_q, Y_dwt_extra];
    else
        X_q = X_q_dct;
    end

    N_total = size(X_q, 2);
    fprintf('  Total training samples (DCT+DWT): %d\n\n', N_total);

    %% ═══════════════════════════════════════════════════════════════
    %  MODEL 1: MLP Quality Predictor  (15 inputs)
    %  ═══════════════════════════════════════════════════════════════
    fprintf('┌─────────────────────────────────────────────────────┐\n');
    fprintf('│  Training Model 1: MLP Quality Predictor (15-feat)  │\n');
    fprintf('└─────────────────────────────────────────────────────┘\n');

    %% Feature normalisation (z-score per feature across training set)
    feat_mean = mean(X_q, 2);                   % [15 x 1]
    feat_std  = std(X_q, 0, 2) + 1e-8;          % [15 x 1]
    X_q_norm  = (X_q - feat_mean) ./ feat_std;  % [15 x N]

    %% PSNR normalisation (SSIM is already [0,1])
    psnr_mean = mean(Y_q(2,:));
    psnr_std  = std(Y_q(2,:))  + 1e-8;
    Y_q_norm  = Y_q;
    Y_q_norm(2,:) = (Y_q(2,:) - psnr_mean) / psnr_std;

    %% Train / val split (80/20, stratified by SSIM)
    perm   = randperm(N_total);
    nTrain = round(0.8 * N_total);
    trainIdx = perm(1:nTrain);
    valIdx   = perm(nTrain+1:end);

    X_train = X_q_norm(:, trainIdx);  Y_train = Y_q_norm(:, trainIdx);
    X_val   = X_q_norm(:, valIdx);    Y_val   = Y_q_norm(:, valIdx);

    fprintf('  Architecture: 15 -> 64 -> 32 -> 16 -> 2\n');
    fprintf('  Training: %d  |  Validation: %d\n', nTrain, N_total-nTrain);
    fprintf('  Epochs: %d\n\n', opts.qualityEpochs);

    qNet = SimpleNet([NFEAT 64 32 16 2], 5e-4, 0.9, 'linear');

    bestValLoss = inf;
    bestParams  = [];
    trainLossHist = zeros(opts.qualityEpochs, 1);
    valLossHist   = zeros(opts.qualityEpochs, 1);

    %% Learning rate step decay: halve LR every 40 epochs
    lrBase = 5e-4;
    for ep = 1:opts.qualityEpochs
        qNet.lr = lrBase * 0.5^floor(ep/40);
        tl = qNet.train_epoch(X_train, Y_train, 64);
        trainLossHist(ep) = tl;

        y_vp    = qNet.predict(X_val);
        valLoss = 0.5 * mean(sum((y_vp - Y_val).^2, 1));
        valLossHist(ep) = valLoss;

        if valLoss < bestValLoss
            bestValLoss = valLoss;
            bestParams  = struct('W', {qNet.W}, 'b', {qNet.b});
        end

        if opts.verbose && (mod(ep,20)==0 || ep==1)
            y_vp_d  = y_vp;
            y_vp_d(2,:) = y_vp(2,:)*psnr_std + psnr_mean;
            y_val_d = Y_val;
            y_val_d(2,:) = Y_val(2,:)*psnr_std + psnr_mean;
            ssim_mae = mean(abs(y_vp_d(1,:) - y_val_d(1,:)));
            psnr_mae = mean(abs(y_vp_d(2,:) - y_val_d(2,:)));
            fprintf('  Ep %3d | Train: %.5f | Val: %.5f | SSIM MAE: %.4f | PSNR MAE: %.2f dB | LR: %.2e\n', ...
                ep, tl, valLoss, ssim_mae, psnr_mae, qNet.lr);
        end
    end

    if ~isempty(bestParams)
        qNet.W = bestParams.W;
        qNet.b = bestParams.b;
    end

    qSavePath = fullfile(modelDir, 'quality_predictor.mat');
    qNet.save_model(qSavePath);

    %% Save ALL normalisation params to a single file
    normParams = struct(...
        'feat_mean', feat_mean,   ...
        'feat_std',  feat_std,    ...
        'psnr_mean', psnr_mean,   ...
        'psnr_std',  psnr_std,    ...
        'nFeat',     NFEAT,       ...
        'nTotalSamples', N_total, ...
        'nDCTSamples',   N_dct,   ...
        'nDWTSamples',   N_dwt_extra ...
    );
    save(fullfile(modelDir, 'quality_predictor_norm.mat'), '-struct', 'normParams');

    results.quality.trainLoss   = trainLossHist;
    results.quality.valLoss     = valLossHist;
    results.quality.bestValLoss = bestValLoss;
    fprintf('  Quality predictor saved. Best val loss: %.5f\n', bestValLoss);

    %% ═══════════════════════════════════════════════════════════════
    %  MODEL 2: LSTM Channel Predictor  (unchanged from v1)
    %  ═══════════════════════════════════════════════════════════════
    fprintf('\n┌─────────────────────────────────────────────┐\n');
    fprintf('│  Training Model 2: LSTM Channel Predictor   │\n');
    fprintf('└─────────────────────────────────────────────┘\n');

    snr_seqs = data.snr_sequences;
    snr_mean_lstm = 20;  snr_std_lstm = 10;
    seqLen = 10;

    X_lstm_list = {};
    Y_lstm_list = {};
    for si = 1:length(snr_seqs)
        seq      = snr_seqs{si};
        seq_norm = (seq - snr_mean_lstm) / snr_std_lstm;
        for t = seqLen+1:length(seq_norm)
            X_lstm_list{end+1} = seq_norm(t-seqLen:t-1);
            Y_lstm_list{end+1} = seq_norm(t);
        end
    end

    nLSTM = length(X_lstm_list);
    fprintf('  Architecture: LSTM(1->32) -> Dense(1)\n');
    fprintf('  Sequence length: %d | Pairs: %d | Epochs: %d\n\n', seqLen, nLSTM, opts.lstmEpochs);

    lstmNet = SimpleLSTM(1, 32, 1, 5e-4, 0.9);
    lstmTrainLoss = zeros(opts.lstmEpochs, 1);

    for ep = 1:opts.lstmEpochs
        perm_l     = randperm(nLSTM);
        epoch_loss = 0;
        nBatch     = min(nLSTM, 200);
        for i = 1:nBatch
            idx = perm_l(i);
            l   = lstmNet.train_step(X_lstm_list{idx}, Y_lstm_list{idx});
            epoch_loss = epoch_loss + l;
        end
        lstmTrainLoss(ep) = epoch_loss / nBatch;

        if opts.verbose && (mod(ep,20)==0 || ep==1)
            test_err = 0;
            nTest = min(50, nLSTM);
            for i = 1:nTest
                pred = lstmNet.predict_seq(X_lstm_list{i});
                test_err = test_err + abs(pred - Y_lstm_list{i});
            end
            mae_snr = test_err / nTest * snr_std_lstm;
            fprintf('  Ep %3d | Train: %.5f | MAE: %.2f dB\n', ep, lstmTrainLoss(ep), mae_snr);
        end
    end

    lstmNet.save_model(fullfile(modelDir, 'channel_lstm.mat'));
    results.lstm.trainLoss = lstmTrainLoss;
    fprintf('  LSTM channel predictor saved.\n');

    %% ═══════════════════════════════════════════════════════════════
    %  MODEL 3: MLP Symbol Denoiser  (unchanged from v1)
    %  ═══════════════════════════════════════════════════════════════
    fprintf('\n┌─────────────────────────────────────────────┐\n');
    fprintf('│  Training Model 3: MLP Symbol Denoiser      │\n');
    fprintf('└─────────────────────────────────────────────┘\n');

    modOrders = [4 16 64];
    results.denoiser = struct();

    for mi = 1:length(modOrders)
        M = modOrders(mi);
        fn = sprintf('mod%d', M);
        if ~isfield(data.symbol_data, fn)
            fprintf('  Skipping M=%d (no data)\n', M);
            continue;
        end
        sd   = data.symbol_data.(fn);
        nSym = length(sd.noisy);

        X_d = zeros(4, nSym);
        X_d(1,:) = real(sd.noisy).';
        X_d(2,:) = imag(sd.noisy).';
        X_d(3,:) = (sd.snr / 40).';
        X_d(4,:) = log2(M) / 6;

        Y_d = zeros(2, nSym);
        Y_d(1,:) = real(sd.clean).';
        Y_d(2,:) = imag(sd.clean).';

        perm_d = randperm(nSym);
        nT = round(0.8 * nSym);
        Xt = X_d(:, perm_d(1:nT));    Yt = Y_d(:, perm_d(1:nT));
        Xv = X_d(:, perm_d(nT+1:end)); Yv = Y_d(:, perm_d(nT+1:end));

        dNet = SimpleNet([4 64 32 2], 1e-3, 0.9, 'linear');
        fprintf('\n  M=%d-QAM: %d symbols | 4->64->32->2\n', M, nSym);

        dTrainLoss = zeros(opts.denoiserEpochs, 1);
        for ep = 1:opts.denoiserEpochs
            tl = dNet.train_epoch(Xt, Yt, 128);
            dTrainLoss(ep) = tl;
            if opts.verbose && (mod(ep,20)==0 || ep==1)
                yp     = dNet.predict(Xv);
                iq_err = mean(sqrt(sum((yp-Yv).^2, 1)));
                fprintf('    Ep %3d | Train: %.6f | Val IQ: %.5f\n', ep, tl, iq_err);
            end
        end

        dNet.save_model(fullfile(modelDir, sprintf('symbol_denoiser_mod%d.mat', M)));
        results.denoiser.(fn) = dTrainLoss;
        fprintf('    Saved denoiser for %d-QAM\n', M);
    end

    %% ═══════════════════════════════════════════════════════════════
    %  Plot
    %% ═══════════════════════════════════════════════════════════════
    try, bold_plot_training(results); catch, end

    fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
    fprintf('║  All 3 Models Trained (Quality: 15 feat, PSNR norm, LR decay) ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');
end

function bold_plot_training(r)
    figure('Position',[100 100 1000 600],'Name','Bold ML Training v2');

    subplot(2,2,1);
    plot(r.quality.trainLoss,'b-','LineWidth',1.5); hold on;
    plot(r.quality.valLoss,  'r-','LineWidth',1.5);
    legend('Train','Val'); xlabel('Epoch'); ylabel('Loss');
    title('Quality Predictor (MLP, 15 feat)'); grid on;

    subplot(2,2,2);
    plot(r.lstm.trainLoss,'b-','LineWidth',1.5);
    xlabel('Epoch'); ylabel('Loss');
    title('Channel Predictor (LSTM)'); grid on;

    if isfield(r.denoiser,'mod4')
        subplot(2,2,3);
        plot(r.denoiser.mod4,'b-','LineWidth',1.5);
        xlabel('Epoch'); ylabel('Loss');
        title('Symbol Denoiser (QPSK)'); grid on;
    end

    if isfield(r.denoiser,'mod64')
        subplot(2,2,4);
        plot(r.denoiser.mod64,'r-','LineWidth',1.5);
        xlabel('Epoch'); ylabel('Loss');
        title('Symbol Denoiser (64-QAM)'); grid on;
    end

    sgtitle('Bold ML Training Pipeline v2 - All Models');
    try saveas(gcf,'bold_training_results_v2.png'); end
end
