function results = bold_train_all(dataPath, opts)
% BOLD_TRAIN_ALL  Train all 3 ML models from generated data
% Trains: (1) MLP quality predictor, (2) LSTM channel predictor,
%         (3) MLP symbol denoiser (per modulation order)
%
% Bold (Unubold) - EEME30003 Major Project
%
% Usage:
%   results = bold_train_all();                          % uses default data
%   results = bold_train_all('bold_training_data.mat');   % specify data file
%
% Produces saved model files:
%   quality_predictor.mat
%   channel_lstm.mat
%   symbol_denoiser_mod4.mat
%   symbol_denoiser_mod16.mat
%   symbol_denoiser_mod64.mat

    if nargin < 1 || isempty(dataPath)
        dataPath = fullfile(fileparts(mfilename('fullpath')), 'bold_training_data.mat');
    end
    if nargin < 2, opts = struct(); end
    if ~isfield(opts, 'qualityEpochs'),  opts.qualityEpochs = 100; end
    if ~isfield(opts, 'lstmEpochs'),     opts.lstmEpochs = 80; end
    if ~isfield(opts, 'denoiserEpochs'), opts.denoiserEpochs = 60; end
    if ~isfield(opts, 'verbose'),        opts.verbose = true; end

    modelDir = fileparts(mfilename('fullpath'));

    fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
    fprintf('║           Bold ML Training Pipeline - 3 Models                ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

    % Check for data
    if ~exist(dataPath, 'file')
        fprintf('Training data not found. Generating now...\n');
        bold_generate_training_data(struct('savePath', dataPath));
    end

    fprintf('Loading training data from: %s\n', dataPath);
    data = load(dataPath);
    results = struct();

    %% ═══════════════════════════════════════════════════════════════
    %  MODEL 1: MLP Quality Predictor
    %  ═══════════════════════════════════════════════════════════════
    fprintf('\n┌─────────────────────────────────────────────┐\n');
    fprintf('│  Training Model 1: MLP Quality Predictor    │\n');
    fprintf('└─────────────────────────────────────────────┘\n');

    X_q = data.quality_features;  % [12 x N]
    Y_q = data.quality_targets;   % [2 x N]  (SSIM, PSNR)

    % Normalise PSNR to similar scale as SSIM for balanced training
    psnr_mean = mean(Y_q(2,:));
    psnr_std  = std(Y_q(2,:)) + 1e-8;
    Y_q_norm = Y_q;
    Y_q_norm(2,:) = (Y_q(2,:) - psnr_mean) / psnr_std;

    % Train/val split (80/20)
    N_q = size(X_q, 2);
    perm = randperm(N_q);
    nTrain = round(0.8 * N_q);
    trainIdx = perm(1:nTrain);
    valIdx   = perm(nTrain+1:end);

    X_train = X_q(:, trainIdx);  Y_train = Y_q_norm(:, trainIdx);
    X_val   = X_q(:, valIdx);    Y_val   = Y_q_norm(:, valIdx);

    % Architecture: 12 -> 64 -> 32 -> 16 -> 2
    qNet = SimpleNet([12 64 32 16 2], 5e-4, 0.9, 'linear');

    fprintf('  Architecture: 12 → 64 → 32 → 16 → 2\n');
    fprintf('  Training samples: %d  |  Validation: %d\n', nTrain, N_q - nTrain);
    fprintf('  Epochs: %d\n\n', opts.qualityEpochs);

    bestValLoss = inf;
    bestParams = [];
    trainLossHist = zeros(opts.qualityEpochs, 1);
    valLossHist   = zeros(opts.qualityEpochs, 1);

    for ep = 1:opts.qualityEpochs
        tl = qNet.train_epoch(X_train, Y_train, 64);
        trainLossHist(ep) = tl;

        % Validation
        y_vp = qNet.predict(X_val);
        valLoss = 0.5 * mean(sum((y_vp - Y_val).^2, 1));
        valLossHist(ep) = valLoss;

        if valLoss < bestValLoss
            bestValLoss = valLoss;
            bestParams = struct('W', {qNet.W}, 'b', {qNet.b});
        end

        if opts.verbose && (mod(ep, 20)==0 || ep==1)
            % Compute actual SSIM/PSNR errors on validation
            y_vp_denorm = y_vp;
            y_vp_denorm(2,:) = y_vp(2,:) * psnr_std + psnr_mean;
            y_val_denorm = Y_val;
            y_val_denorm(2,:) = Y_val(2,:) * psnr_std + psnr_mean;

            ssim_mae = mean(abs(y_vp_denorm(1,:) - y_val_denorm(1,:)));
            psnr_mae = mean(abs(y_vp_denorm(2,:) - y_val_denorm(2,:)));
            fprintf('  Ep %3d | Train: %.5f | Val: %.5f | SSIM MAE: %.4f | PSNR MAE: %.2f dB\n', ...
                ep, tl, valLoss, ssim_mae, psnr_mae);
        end
    end

    % Restore best weights
    if ~isempty(bestParams)
        qNet.W = bestParams.W;
        qNet.b = bestParams.b;
    end

    qSavePath = fullfile(modelDir, 'quality_predictor.mat');
    qNet.save_model(qSavePath);

    % Save normalisation params alongside
    normParams = struct('psnr_mean', psnr_mean, 'psnr_std', psnr_std);
    save(fullfile(modelDir, 'quality_predictor_norm.mat'), '-struct', 'normParams');

    results.quality.trainLoss = trainLossHist;
    results.quality.valLoss   = valLossHist;
    results.quality.bestValLoss = bestValLoss;
    fprintf('  ✓ Quality predictor saved. Best val loss: %.5f\n', bestValLoss);

    %% ═══════════════════════════════════════════════════════════════
    %  MODEL 2: LSTM Channel Predictor
    %  ═══════════════════════════════════════════════════════════════
    fprintf('\n┌─────────────────────────────────────────────┐\n');
    fprintf('│  Training Model 2: LSTM Channel Predictor   │\n');
    fprintf('└─────────────────────────────────────────────┘\n');

    snr_seqs = data.snr_sequences;
    snr_mean = 20;  snr_std = 10;

    % Create training pairs: (input sequence, target next value)
    seqLen = 10;  % use 10 past values to predict next
    X_lstm_list = {};
    Y_lstm_list = {};

    for si = 1:length(snr_seqs)
        seq = snr_seqs{si};
        seq_norm = (seq - snr_mean) / snr_std;
        for t = seqLen+1:length(seq_norm)
            X_lstm_list{end+1} = seq_norm(t-seqLen:t-1);
            Y_lstm_list{end+1} = seq_norm(t);
        end
    end

    nLSTM = length(X_lstm_list);
    fprintf('  Architecture: LSTM(1 → 32) → Dense(1)\n');
    fprintf('  Sequence length: %d | Training pairs: %d\n', seqLen, nLSTM);
    fprintf('  Epochs: %d\n\n', opts.lstmEpochs);

    lstmNet = SimpleLSTM(1, 32, 1, 5e-4, 0.9);

    lstmTrainLoss = zeros(opts.lstmEpochs, 1);

    for ep = 1:opts.lstmEpochs
        perm = randperm(nLSTM);
        epoch_loss = 0;
        nBatch = min(nLSTM, 200);  % subsample for speed

        for i = 1:nBatch
            idx = perm(i);
            X_s = X_lstm_list{idx};  % [1 x seqLen]
            Y_s = Y_lstm_list{idx};  % scalar

            l = lstmNet.train_step(X_s, Y_s);
            epoch_loss = epoch_loss + l;
        end

        lstmTrainLoss(ep) = epoch_loss / nBatch;

        if opts.verbose && (mod(ep, 20)==0 || ep==1)
            % Evaluate on a few samples
            test_err = 0;
            nTest = min(50, nLSTM);
            for i = 1:nTest
                pred = lstmNet.predict_seq(X_lstm_list{i});
                test_err = test_err + abs(pred - Y_lstm_list{i});
            end
            mae_snr = test_err / nTest * snr_std;  % denormalised
            fprintf('  Ep %3d | Train Loss: %.5f | Test MAE: %.2f dB\n', ...
                ep, lstmTrainLoss(ep), mae_snr);
        end
    end

    lstmSavePath = fullfile(modelDir, 'channel_lstm.mat');
    lstmNet.save_model(lstmSavePath);
    results.lstm.trainLoss = lstmTrainLoss;
    fprintf('  ✓ LSTM channel predictor saved.\n');

    %% ═══════════════════════════════════════════════════════════════
    %  MODEL 3: MLP Symbol Denoiser (per modulation order)
    %  ═══════════════════════════════════════════════════════════════
    fprintf('\n┌─────────────────────────────────────────────┐\n');
    fprintf('│  Training Model 3: MLP Symbol Denoiser      │\n');
    fprintf('└─────────────────────────────────────────────┘\n');

    modOrders = [4 16 64];
    results.denoiser = struct();

    for mi = 1:length(modOrders)
        M = modOrders(mi);
        fieldName = sprintf('mod%d', M);

        if ~isfield(data.symbol_data, fieldName)
            fprintf('  Skipping M=%d (no data)\n', M);
            continue;
        end

        sd = data.symbol_data.(fieldName);
        nSym = length(sd.noisy);

        % Build input/output matrices
        X_d = zeros(4, nSym);
        X_d(1,:) = real(sd.noisy).';
        X_d(2,:) = imag(sd.noisy).';
        X_d(3,:) = (sd.snr / 40).';
        X_d(4,:) = (log2(M) / 6);

        Y_d = zeros(2, nSym);
        Y_d(1,:) = real(sd.clean).';
        Y_d(2,:) = imag(sd.clean).';

        % Train/val split
        perm = randperm(nSym);
        nT = round(0.8 * nSym);
        Xt = X_d(:, perm(1:nT));  Yt = Y_d(:, perm(1:nT));
        Xv = X_d(:, perm(nT+1:end)); Yv = Y_d(:, perm(nT+1:end));

        % Architecture: 4 -> 64 -> 32 -> 2
        dNet = SimpleNet([4 64 32 2], 1e-3, 0.9, 'linear');

        fprintf('\n  M=%d-QAM: %d symbols | Arch: 4→64→32→2\n', M, nSym);

        dTrainLoss = zeros(opts.denoiserEpochs, 1);
        for ep = 1:opts.denoiserEpochs
            tl = dNet.train_epoch(Xt, Yt, 128);
            dTrainLoss(ep) = tl;

            if opts.verbose && (mod(ep, 20)==0 || ep==1)
                yp = dNet.predict(Xv);
                iq_err = mean(sqrt(sum((yp - Yv).^2, 1)));
                fprintf('    Ep %3d | Train: %.6f | Val IQ error: %.5f\n', ep, tl, iq_err);
            end
        end

        dSavePath = fullfile(modelDir, sprintf('symbol_denoiser_mod%d.mat', M));
        dNet.save_model(dSavePath);
        results.denoiser.(fieldName) = dTrainLoss;
        fprintf('    ✓ Saved denoiser for %d-QAM\n', M);
    end

    %% Plot training curves
    fprintf('\n  Plotting training curves...\n');
    try
        plot_training_results(results);
    catch ME
        fprintf('  Warning: could not plot (%s)\n', ME.message);
    end

    fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
    fprintf('║   All 3 Models Trained Successfully                           ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');
end

%% Plotting
function plot_training_results(results)
    figure('Position', [100 100 1000 600], 'Name', 'Bold ML Training Results');

    subplot(2,2,1);
    plot(results.quality.trainLoss, 'b-', 'LineWidth', 1.5); hold on;
    plot(results.quality.valLoss, 'r-', 'LineWidth', 1.5);
    legend('Train','Val'); xlabel('Epoch'); ylabel('Loss');
    title('Quality Predictor (MLP)'); grid on;

    subplot(2,2,2);
    plot(results.lstm.trainLoss, 'b-', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Loss');
    title('Channel Predictor (LSTM)'); grid on;

    if isfield(results.denoiser, 'mod4')
        subplot(2,2,3);
        plot(results.denoiser.mod4, 'b-', 'LineWidth', 1.5);
        xlabel('Epoch'); ylabel('Loss');
        title('Symbol Denoiser (QPSK)'); grid on;
    end

    if isfield(results.denoiser, 'mod64')
        subplot(2,2,4);
        plot(results.denoiser.mod64, 'r-', 'LineWidth', 1.5);
        xlabel('Epoch'); ylabel('Loss');
        title('Symbol Denoiser (64-QAM)'); grid on;
    end

    sgtitle('Bold ML Training Pipeline - All Models');
    try saveas(gcf, 'bold_training_results.png'); end
end
