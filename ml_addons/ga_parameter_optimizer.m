function [optParams, gaMetrics] = ga_parameter_optimizer(chCond, curParams, gaSettings)
% GA_PARAMETER_OPTIMIZER  GA to find best [K, Qb_idx, Qe_idx]
% Evolves layer-split and quantization params for current channel conditions
% using tournament selection, crossover, and mutation.
% MCS is not in the search space; rules + ceiling handle MCS in
% rx_policy_update.m. The GA focuses on parameters where the analytic
% model is sufficient and rules are coarse.
%
% Recalibrated for the improved baseline (quant-index fix, proactive MCS
% push, rx_v2 receiver).  K search space capped at 32 to match the policy
% clamp.  Fitness centres K around 20 with a stability penalty to prevent
% harmful frame-to-frame oscillation.

    if nargin < 3, gaSettings = struct(); end
    if ~isfield(gaSettings, 'popSize'),      gaSettings.popSize = 40; end
    if ~isfield(gaSettings, 'generations'),  gaSettings.generations = 15; end
    if ~isfield(gaSettings, 'mutationRate'), gaSettings.mutationRate = 0.15; end
    if ~isfield(gaSettings, 'eliteCount'),   gaSettings.eliteCount = 4; end
    if ~isfield(gaSettings, 'verbose'),      gaSettings.verbose = false; end

    if ~isfield(chCond, 'snr_dB'), chCond.snr_dB = 20; end
    if ~isfield(curParams, 'K'),       curParams.K = 20; end
    if ~isfield(curParams, 'Qb_idx'),  curParams.Qb_idx = 0; end
    if ~isfield(curParams, 'Qe_idx'),  curParams.Qe_idx = 0; end
    if ~isfield(curParams, 'mcs_base'),curParams.mcs_base = 2; end
    if ~isfield(curParams, 'mcs_enh'), curParams.mcs_enh = 6; end

    %% chromosome: [K, Qb_idx, Qe_idx]   (MCS not in search space)
    nGenes = 3;
    gMin = [5  0 0];
    gMax = [32 3 3];   % K capped at 32 to match rx_policy_update clamp

    popSize = gaSettings.popSize;
    nGen = gaSettings.generations;

    %% init population (seed with current params + tighter perturbations)
    pop = zeros(popSize, nGenes);
    curChr = clamp([double(curParams.K) double(curParams.Qb_idx) double(curParams.Qe_idx)], gMin, gMax);
    pop(1,:) = curChr;
    for i = 2:popSize
        perturb = round(randn(1,nGenes) .* [3 1 1]);
        pop(i,:) = clamp(curChr + perturb, gMin, gMax);
    end

    snr = chCond.snr_dB;
    currentK = double(curParams.K);

    prevQuality = struct('ssim', NaN, 'psnr', NaN, 'enhFailRatio', NaN);
    if isfield(chCond, 'prevSSIM') && ~isnan(chCond.prevSSIM)
        prevQuality.ssim = chCond.prevSSIM;
    end
    if isfield(chCond, 'prevPsnr') && ~isnan(chCond.prevPsnr)
        prevQuality.psnr = chCond.prevPsnr;
    end
    if isfield(chCond, 'prevEnhFailRatio') && ~isnan(chCond.prevEnhFailRatio)
        prevQuality.enhFailRatio = chCond.prevEnhFailRatio;
    end

    fitHist = zeros(nGen,1);

    %% evolve
    for gen = 1:nGen
        fit = zeros(popSize,1);
        for i = 1:popSize
            fit(i) = calc_fitness(pop(i,:), snr, prevQuality, currentK);
        end

        [bestFit, bestIdx] = max(fit);
        fitHist(gen) = bestFit;

        if gaSettings.verbose
            fprintf('  GA gen %d: fitness=%.3f K=%d Qb=%d Qe=%d\n', ...
                gen, bestFit, pop(bestIdx,1), pop(bestIdx,2), pop(bestIdx,3));
        end

        pop = evolve(pop, fit, gaSettings, gMin, gMax);
    end

    %% extract best
    fit = zeros(popSize,1);
    for i = 1:popSize
        fit(i) = calc_fitness(pop(i,:), snr, prevQuality, currentK);
    end
    [~, bestIdx] = max(fit);
    best = pop(bestIdx,:);

    optParams = struct();
    optParams.K        = uint8(best(1));
    optParams.Qb_idx   = uint8(best(2));
    optParams.Qe_idx   = uint8(best(3));
    optParams.mcs_base = uint8(curParams.mcs_base);   % pass-through, not evolved
    optParams.mcs_enh  = uint8(curParams.mcs_enh);    % pass-through, not evolved

    gaMetrics = struct();
    gaMetrics.fitnessHistory = fitHist;
    gaMetrics.finalFitness = fitHist(end);
    gaMetrics.generations = nGen;
    gaMetrics.bestChromosome = best;
end


%% ---- fitness (regime-adaptive, recalibrated for improved baseline) ----

function f = calc_fitness(chr, snr_dB, prevQuality, currentK)
% Fitness for [K, Qb_idx, Qe_idx].
% Recalibrated for the improved pipeline (quant-fix, proactive MCS push,
% rx_v2).  Key changes vs. prior version:
%   - kRatio normalised to 32 (policy cap) instead of 64
%   - Quadratic K-centering penalty around K=20 (proven sweet spot)
%   - Stability penalty to prevent frame-to-frame K oscillation
%   - Regime thresholds raised (baseline SSIM now ~0.97, not ~0.88)
%   - Crisis regime less aggressive to avoid harmful K spikes
    if nargin < 3, prevQuality = struct('ssim', NaN, 'psnr', NaN, 'enhFailRatio', NaN); end
    if nargin < 4, currentK = 20; end
    K = chr(1); QbIdx = chr(2); QeIdx = chr(3);
    kRatio = K / 32;
    qSum = QbIdx + QeIdx;

    %% --- sub-scores ---
    qDistBase = (QbIdx + 1) / 4;
    qDistEnh  = (QeIdx + 1) / 4;
    qScore = 1.0 - 0.5*(kRatio * qDistBase + (1 - kRatio) * qDistEnh);

    compScore = (QbIdx + QeIdx) / 6;
    compScore = compScore + 0.25 * (1 - kRatio);

    RICIAN_PENALTY_dB = 4.0;
    effSnr = snr_dB - RICIAN_PENALTY_dB;
    if effSnr < 10
        baseReliability = 0.5;
    elseif effSnr < 20
        baseReliability = 0.5 + 0.5 * (effSnr - 10) / 10;
    else
        baseReliability = 1.0;
    end
    protScore = 0.7 * kRatio * baseReliability + 0.3 * (1 - kRatio) * baseReliability;

    %% --- K-centering penalty (quadratic, centred on K=20) ---
    kPen = 0.015 * ((K - 20)^2) / 10;

    %% --- stability penalty (discourages large K jumps between frames) ---
    stabilityPen = 0.013 * abs(K - currentK);

    %% --- regime detection & weight selection ---
    prevSSIM = NaN;
    if isfield(prevQuality, 'ssim'), prevSSIM = prevQuality.ssim; end

    [wQ, wC, wP] = regime_weights(snr_dB, prevSSIM);

    f = wQ * qScore + wC * compScore + wP * protScore - kPen - stabilityPen;

    %% --- measured quality correction (thresholds raised for new baseline) ---
    if ~isnan(prevSSIM)
        if prevSSIM < 0.95
            f = f - 1.2 * (0.95 - prevSSIM);
        end
        f = f + 0.4 * (prevSSIM - 0.93);
        if prevSSIM < 0.85
            f = f + 0.2 * (K - 20) / 20;
        end
    end

    if isfield(prevQuality, 'enhFailRatio') && ~isnan(prevQuality.enhFailRatio)
        if prevQuality.enhFailRatio > 0.5
            f = f - 0.3 * (1 - kRatio);
        end
    end

    %% --- PSNR / quant guard (aligns GA with rx_policy_update PSNR gates) ---
    % Without this, high wC can make coarse Qb/Qe win even when SSIM is NaN
    % (no telemetry) or when there is no PSNR headroom for coarser quant.
    prevPsnr = NaN;
    if isfield(prevQuality, 'psnr'), prevPsnr = prevQuality.psnr; end
    if ~isnan(prevPsnr) && prevPsnr < 40
        f = f - 0.5 * qSum;
    end
    if isnan(prevSSIM) && isnan(prevPsnr)
        f = f - 0.45 * qSum;
    end
    if ~isnan(prevSSIM) && prevSSIM >= 0.96 && qSum > 0
        % Softer than 0.15 so GA can still nudge Q/K in easy channels (closer to
        % legacy report spread); strong pipeline SSIM ~0.97 otherwise locks GA to rules.
        f = f - 0.07 * qSum * (prevSSIM - 0.95) / 0.05;
    end
end


function [wQ, wC, wP] = regime_weights(snr_dB, prevSSIM)
% REGIME_WEIGHTS  Recalibrated for improved baseline (SSIM ~0.97).
%
% Regimes (SNR thresholds with SSIM overrides):
%   Crisis      SNR < 12  OR  prevSSIM < 0.85   -> [2.5, 0.0, 1.0]
%   Stressed    12 <= SNR < 20                   -> [2.2, 0.3, 0.8]
%   Comfortable 20 <= SNR < 32                 -> [2.15, 0.38, 0.42]
%   Surplus     SNR >= 32 AND prevSSIM >= 0.97 -> [1.65, 0.42, 0.32]
%
% Notes:
%   - wC kept modest so compScore cannot dominate qScore (coarse Q).
%   - rx_feedback passes prevSSIM/prevPsnr into chCond; without them the GA
%     used to favour coarse quant indices and tank PSNR.
%
% Blending: within +-2 dB of each boundary, weights are linearly
% interpolated between adjacent regimes for continuity.

    W_CRISIS      = [2.5  0.0  1.0];
    W_STRESSED    = [2.2  0.3  0.8];
    % Keep wC modest: compScore can otherwise dominate qScore and pick coarse Q.
    W_COMFORTABLE = [2.15  0.38  0.42];
    W_SURPLUS     = [1.65  0.42  0.32];

    hasPrev = ~isnan(prevSSIM);

    if hasPrev && prevSSIM < 0.85
        wQ = W_CRISIS(1); wC = W_CRISIS(2); wP = W_CRISIS(3);
        return;
    end
    if hasPrev && prevSSIM < 0.93 && snr_dB >= 20
        alpha = (prevSSIM - 0.85) / 0.08;   % 0 at 0.85, 1 at 0.93
        alpha = max(0, min(1, alpha));
        w = (1 - alpha) * W_CRISIS + alpha * W_STRESSED;
        wQ = w(1); wC = w(2); wP = w(3);
        return;
    end

    surplusOK = hasPrev && prevSSIM >= 0.97;

    BLEND = 2.0;

    if snr_dB < (12 - BLEND)
        w = W_CRISIS;
    elseif snr_dB < (12 + BLEND)
        t = (snr_dB - (12 - BLEND)) / (2 * BLEND);
        w = (1 - t) * W_CRISIS + t * W_STRESSED;
    elseif snr_dB < (20 - BLEND)
        w = W_STRESSED;
    elseif snr_dB < (20 + BLEND)
        t = (snr_dB - (20 - BLEND)) / (2 * BLEND);
        w = (1 - t) * W_STRESSED + t * W_COMFORTABLE;
    elseif snr_dB < (32 - BLEND)
        w = W_COMFORTABLE;
    elseif snr_dB < (32 + BLEND)
        if surplusOK
            t = (snr_dB - (32 - BLEND)) / (2 * BLEND);
            w = (1 - t) * W_COMFORTABLE + t * W_SURPLUS;
        else
            w = W_COMFORTABLE;
        end
    else
        if surplusOK
            w = W_SURPLUS;
        else
            w = W_COMFORTABLE;
        end
    end

    wQ = w(1); wC = w(2); wP = w(3);
end


%% ---- GA operators ----

function newPop = evolve(pop, fit, settings, gMin, gMax)
    [popSize, nGenes] = size(pop);
    newPop = zeros(popSize, nGenes);

    % elitism
    [~, si] = sort(fit, 'descend');
    nElite = settings.eliteCount;
    newPop(1:nElite,:) = pop(si(1:nElite),:);

    % fill rest with crossover + mutation
    for i = (nElite+1):popSize
        c1 = randperm(popSize, min(3,popSize));
        [~,w1] = max(fit(c1));  p1 = pop(c1(w1),:);
        c2 = randperm(popSize, min(3,popSize));
        [~,w2] = max(fit(c2));  p2 = pop(c2(w2),:);

        % uniform crossover
        mask = rand(1,nGenes) > 0.5;
        child = p1;  child(mask) = p2(mask);

        % mutation
        for g = 1:nGenes
            if rand < settings.mutationRate
                child(g) = child(g) + round(randn * (gMax(g)-gMin(g)) * 0.2);
            end
        end
        newPop(i,:) = clamp(child, gMin, gMax);
    end
end

function c = clamp(c, gMin, gMax)
    c = round(max(gMin, min(gMax, c)));
end
