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
%
% DWT mode: set chCond.isDWT = true to activate wavelet-aware fitness.
% In DWT mode:
%   - K centering shifts to 26 (DWT sweet spot — more protection needed)
%   - Fitness accounts for subband energy distribution (LL >> LH/HL >> HH)
%   - Stability penalty is stronger (DWT is already well-optimised by rules)
%   - Regime thresholds raised (DWT baseline SSIM ~0.99, not ~0.97)
%   - compScore is penalised more aggressively to avoid hurting LL quality

    if nargin < 3, gaSettings = struct(); end
    if ~isfield(gaSettings, 'popSize'),      gaSettings.popSize = 40; end
    if ~isfield(gaSettings, 'generations'),  gaSettings.generations = 15; end
    if ~isfield(gaSettings, 'mutationRate'), gaSettings.mutationRate = 0.15; end
    if ~isfield(gaSettings, 'eliteCount'),   gaSettings.eliteCount = 4; end
    if ~isfield(gaSettings, 'verbose'),      gaSettings.verbose = false; end

    % GA uses rand/randn heavily; channel_model_v1 calls awgn *after* restoring
    % rng from the fading block, so awgn noise would otherwise depend on how
    % many draws the GA took — biasing Rules+GA vs Rules-only comparisons.
    savedRng = rng;
    cleanupGaRng = onCleanup(@() rng(savedRng));

    if ~isfield(chCond, 'snr_dB'),  chCond.snr_dB = 20; end
    if ~isfield(chCond, 'isDWT'),   chCond.isDWT = false; end   % <-- DWT flag
    if ~isfield(curParams, 'K'),       curParams.K = 20; end
    if ~isfield(curParams, 'Qb_idx'),  curParams.Qb_idx = 0; end
    if ~isfield(curParams, 'Qe_idx'),  curParams.Qe_idx = 0; end
    if ~isfield(curParams, 'mcs_base'),curParams.mcs_base = 2; end
    if ~isfield(curParams, 'mcs_enh'), curParams.mcs_enh = 6; end

    isDWT = logical(chCond.isDWT);

    %% chromosome: [K, Qb_idx, Qe_idx]   (MCS not in search space)
    % DWT NOTE: two_layer_tx_wavelet uses fixed Qb=Qe=1 and ignores
    % Qb_idx/Qe_idx entirely. Evolving them wastes search capacity and
    % introduces noise. In DWT mode we lock Qb_idx=0, Qe_idx=0 and only
    % evolve K — the one parameter that actually affects DWT quality.
    nGenes = 3;
    if isDWT
        gMin = [5  0 0];
        gMax = [32 0 0];   % Lock Qb/Qe at 0 for DWT — TX ignores them anyway
    else
        gMin = [5  0 0];
        gMax = [32 3 3];   % K capped at 32 to match rx_policy_update clamp
    end

    popSize = gaSettings.popSize;
    nGen = gaSettings.generations;

    %% init population (seed with current params + tighter perturbations)
    % DWT: use smaller perturbations — the rules already do a good job,
    % so we want the GA to make fine adjustments rather than large jumps.
    pop = zeros(popSize, nGenes);
    curChr = clamp([double(curParams.K) double(curParams.Qb_idx) double(curParams.Qe_idx)], gMin, gMax);
    pop(1,:) = curChr;
    perturbScale = [3 1 1];
    if isDWT
        perturbScale = [2 0 0];  % only perturb K in DWT mode
    end
    for i = 2:popSize
        perturb = round(randn(1,nGenes) .* perturbScale);
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
            fit(i) = calc_fitness(pop(i,:), snr, prevQuality, currentK, isDWT);
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
        fit(i) = calc_fitness(pop(i,:), snr, prevQuality, currentK, isDWT);
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
    gaMetrics.isDWT = isDWT;
end


%% ---- fitness router ----

function f = calc_fitness(chr, snr_dB, prevQuality, currentK, isDWT)
    if nargin < 5, isDWT = false; end
    if isDWT
        f = calc_fitness_dwt(chr, snr_dB, prevQuality, currentK);
    else
        f = calc_fitness_dct(chr, snr_dB, prevQuality, currentK);
    end
end


%% ---- DCT fitness (original, recalibrated) ----

function f = calc_fitness_dct(chr, snr_dB, prevQuality, currentK)
% Fitness for [K, Qb_idx, Qe_idx] — DCT pipeline.
% Recalibrated for the improved pipeline (quant-fix, proactive MCS push,
% rx_v2).  Key changes vs. prior version:
%   - kRatio normalised to 32 (policy cap) instead of 64
%   - Quadratic K-centering penalty around K=20 (proven sweet spot)
%   - Stability penalty to prevent frame-to-frame K oscillation
%   - Regime thresholds raised (baseline SSIM now ~0.97, not ~0.88)
%   - Crisis regime less aggressive to avoid harmful K spikes
%
% Mode-aware improvements (latest):
%   - CEILING mode (prevSSIM>=0.97 + SNR>=20): strong stability, GA becomes
%     near-no-op to avoid introducing blend noise in already-optimal cases.
%   - CRISIS mode (SNR<15): K-centering penalty disabled, strong reward for
%     high K so GA can push protection to maximum.
%   - SWING mode (prevSSIM>=0.95 + SNR<15): channel just dropped — max
%     protection bonus to counteract one-frame feedback lag.
%   - RECOVERY mode (prevSSIM<0.5 + SNR<15): previous frame was catastrophic
%     and we're still in crisis — lock to maximum protection.
    if nargin < 3, prevQuality = struct('ssim', NaN, 'psnr', NaN, 'enhFailRatio', NaN); end
    if nargin < 4, currentK = 20; end
    K = chr(1); QbIdx = chr(2); QeIdx = chr(3);
    kRatio = K / 32;
    qSum = QbIdx + QeIdx;

    % Pull feedback early for mode detection
    prevSSIM = NaN;
    if isfield(prevQuality, 'ssim'), prevSSIM = prevQuality.ssim; end
    prevPsnr = NaN;
    if isfield(prevQuality, 'psnr'), prevPsnr = prevQuality.psnr; end

    % === MODE DETECTION ===
    atCeiling    = ~isnan(prevSSIM) && prevSSIM >= 0.97 && snr_dB >= 20;
    inCrisis     = snr_dB < 15;
    hadSwing     = ~isnan(prevSSIM) && prevSSIM >= 0.95 && snr_dB < 15;
    hadCatastrophe = ~isnan(prevSSIM) && prevSSIM < 0.5;

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

    %% --- K-centering penalty (mode-aware) ---
    % In crisis, remove centering penalty — we WANT K high for protection.
    if inCrisis
        kPen = 0;
    else
        kPen = 0.015 * ((K - 20)^2) / 10;
    end

    %% --- stability penalty (mode-aware) ---
    % Ceiling: very strong — lock to currentK to avoid blend noise.
    % Crisis: moderate — let GA move but don't thrash.
    % Normal: light — allow smooth adaptation.
    if atCeiling
        stabilityPen = 0.15 * abs(K - currentK);
    elseif inCrisis
        stabilityPen = 0.03 * abs(K - currentK);
    else
        stabilityPen = 0.02 * abs(K - currentK);
    end

    %% --- regime detection & weight selection ---
    [wQ, wC, wP] = regime_weights_dct(snr_dB, prevSSIM);

    f = wQ * qScore + wC * compScore + wP * protScore - kPen - stabilityPen;

    %% --- MODE BONUSES ---

    % CRISIS: strongly reward high K for maximum base-layer protection
    if inCrisis
        if K >= 25, f = f + 0.20 * (K - 24) / 8; end
        if K >= 28, f = f + 0.15; end
        if K >= 30, f = f + 0.10; end
    end

    % SWING: channel just dropped — push even harder on protection
    if hadSwing
        if K >= 27, f = f + 0.35; end
        if K >= 30, f = f + 0.20; end
        if QbIdx == 0, f = f + 0.10; end    % keep base quant fine
    end

    % CATASTROPHE: last frame failed badly — lock to max protection
    if hadCatastrophe && inCrisis
        if K == 32, f = f + 0.40; end
        if QbIdx == 0, f = f + 0.15; end
    end

    % CEILING: reward staying at current state + finest Qe
    if atCeiling
        if K == currentK, f = f + 0.15; end
        if QeIdx == 0, f = f + 0.08; end
        if QbIdx == 0, f = f + 0.05; end
    end

    %% --- measured quality correction (existing) ---
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

    %% --- PSNR / quant guard ---
    if ~isnan(prevPsnr) && prevPsnr < 40
        f = f - 0.5 * qSum;
    end
    if isnan(prevSSIM) && isnan(prevPsnr)
        f = f - 0.45 * qSum;
    end
    if ~isnan(prevSSIM) && prevSSIM >= 0.96 && qSum > 0
        f = f - 0.15 * qSum * (prevSSIM - 0.95) / 0.05;
    end
end


%% ---- DWT fitness (new) ----

function f = calc_fitness_dwt(chr, snr_dB, prevQuality, currentK)
% Fitness for [K, Qb_idx, Qe_idx] — DWT pipeline.
%
% DWT-specific design rationale:
%   - Subband energy distribution: LL ~60-90% of image energy, LH/HL medium,
%     HH minimal. This is fundamentally different from DCT where energy is
%     more uniformly distributed across zigzag-ordered coefficients.
%   - K in DWT context controls how many coefficients per block go to base
%     layer across all subbands. Higher K = more LL/LH/HL content protected.
%   - Sweet spot for DWT is K=26 (higher than DCT's K=20) because wavelet
%     coefficients in the base layer need to cover more structural content.
%   - Quantisation coarseness hurts DWT more than DCT because wavelet
%     coefficients represent spatial frequency bands rather than individual
%     pixel blocks — errors propagate across larger image regions.
%   - DWT baseline SSIM is already ~0.99 under good conditions, so the
%     correction thresholds are raised accordingly.
%   - Stability penalty is stronger — rules already handle DWT well, so
%     the GA should make conservative fine adjustments only.

    if nargin < 3, prevQuality = struct('ssim', NaN, 'psnr', NaN, 'enhFailRatio', NaN); end
    if nargin < 4, currentK = 26; end

    K = chr(1); QbIdx = chr(2); QeIdx = chr(3);
    kRatio = K / 32;
    qSum = QbIdx + QeIdx;

    %% --- subband-aware protection score ---
    % LL subband: highest energy, must get through cleanly.
    % Protection score weights base layer reliability much more heavily
    % than DCT because LL failure cascades across the whole image.
    RICIAN_PENALTY_dB = 4.0;
    effSnr = snr_dB - RICIAN_PENALTY_dB;
    if effSnr < 10
        baseReliability = 0.45;   % DWT more sensitive to deep fades
    elseif effSnr < 20
        baseReliability = 0.45 + 0.55 * (effSnr - 10) / 10;
    else
        baseReliability = 1.0;
    end

    % LL protection: scales strongly with kRatio — more K = more LL protected
    llProtScore = kRatio * baseReliability;

    % LH/HL protection: medium importance — these are horizontal/vertical edges
    % They sit in the middle of the subband ordering, benefit from higher K
    lhHlScore = 0.5 * kRatio * baseReliability;

    % HH: high frequency diagonal detail — least important, goes to enh layer
    % Penalise if we're wasting base layer capacity on HH content
    hhPenalty = 0.1 * max(0, kRatio - 0.7);   % only penalise if K > ~22

    % Combined protection score — heavily weighted toward LL
    protScore = 0.6 * llProtScore + 0.3 * lhHlScore - hhPenalty;

    %% --- quantisation score ---
    % DWT quantisation errors spread spatially across subband regions,
    % so we penalise coarse quantisation more than DCT does.
    qDistBase = (QbIdx + 1) / 4;
    qDistEnh  = (QeIdx + 1) / 4;
    % Base quant matters more than enhancement for DWT (LL integrity)
    qScore = 1.0 - (0.65 * qDistBase + 0.35 * qDistEnh);

    %% --- compression score ---
    % DWT can tolerate some compression in HH (enhancement) but base
    % (LL) must stay fine. Reward modest compression in enhancement only.
    compScore = 0.5 * (QeIdx / 3);              % reward coarser enh quant
    compScore = compScore + 0.1 * (1 - kRatio); % small reward for less base

    %% --- K-centering penalty ---
    % DWT sweet spot is K=26 (more protection than DCT's K=20)
    % because wavelet subbands need more coefficients to represent structure.
    % Use a softer quadratic to allow exploration around the sweet spot.
    K_SWEET_DWT = 26;
    kPen = 0.012 * ((K - K_SWEET_DWT)^2) / 10;

    %% --- stability penalty (stronger for DWT) ---
    % DWT rules already handle adaptation well. The GA should only make
    % small corrections. Larger penalty than DCT to prevent oscillation.
    % Extra penalty when SNR is low — avoid wild K jumps on bad channel.
    if snr_dB < 15
        stabilityPen = 0.08 * abs(K - currentK);  % very conservative at low SNR
    else
        stabilityPen = 0.05 * abs(K - currentK);  % moderate elsewhere
    end

    %% --- regime weights ---
    prevSSIM = NaN;
    if isfield(prevQuality, 'ssim'), prevSSIM = prevQuality.ssim; end

    [wQ, wC, wP] = regime_weights_dwt(snr_dB, prevSSIM);

    f = wQ * qScore + wC * compScore + wP * protScore - kPen - stabilityPen;

    %% --- measured quality correction ---
    % DWT thresholds raised: baseline is ~0.99, not ~0.97.
    % Only intervene meaningfully if quality has actually dropped.
    if ~isnan(prevSSIM)
        if prevSSIM < 0.97
            % Quality has dropped below DWT baseline — push harder on protection
            f = f - 1.5 * (0.97 - prevSSIM);
        end
        f = f + 0.3 * (prevSSIM - 0.97);
        if prevSSIM < 0.92
            % Significant drop — favour higher K to protect base layer
            f = f + 0.25 * (K - K_SWEET_DWT) / 10;
        end
    end

    if isfield(prevQuality, 'enhFailRatio') && ~isnan(prevQuality.enhFailRatio)
        if prevQuality.enhFailRatio > 0.5
            % Enhancement failing a lot — stop investing in enhancement
            f = f - 0.25 * (1 - kRatio);
        end
    end

    %% --- PSNR / quant guard ---
    % DWT is more sensitive to quantisation so threshold is higher (45 dB)
    prevPsnr = NaN;
    if isfield(prevQuality, 'psnr'), prevPsnr = prevQuality.psnr; end
    if ~isnan(prevPsnr) && prevPsnr < 45
        f = f - 0.6 * qSum;   % stronger penalty than DCT (0.5)
    end
    if isnan(prevSSIM) && isnan(prevPsnr)
        f = f - 0.5 * qSum;   % no telemetry: avoid coarse quant
    end
    if ~isnan(prevSSIM) && prevSSIM >= 0.99 && qSum > 0
        % Already near perfect — slight compression exploration allowed
        f = f - 0.08 * qSum * (prevSSIM - 0.98) / 0.02;
    end
end


%% ---- DCT regime weights (original) ----

function [wQ, wC, wP] = regime_weights_dct(snr_dB, prevSSIM)
% REGIME_WEIGHTS_DCT  Recalibrated for improved DCT baseline (SSIM ~0.97).
    W_CRISIS      = [2.5  0.0  1.0];
    W_STRESSED    = [2.2  0.3  0.8];
    W_COMFORTABLE = [2.15 0.38 0.42];
    W_SURPLUS     = [1.65 0.42 0.32];

    hasPrev = ~isnan(prevSSIM);

    if hasPrev && prevSSIM < 0.85
        wQ = W_CRISIS(1); wC = W_CRISIS(2); wP = W_CRISIS(3);
        return;
    end
    if hasPrev && prevSSIM < 0.93 && snr_dB >= 20
        alpha = (prevSSIM - 0.85) / 0.08;
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


%% ---- DWT regime weights (new) ----

function [wQ, wC, wP] = regime_weights_dwt(snr_dB, prevSSIM)
% REGIME_WEIGHTS_DWT  Calibrated for DWT baseline (SSIM ~0.99).
%
% Key differences from DCT regime weights:
%   - All SSIM thresholds raised by ~0.04 (DWT natural baseline is higher)
%   - wC (compression) capped lower — DWT quant errors spread more widely
%   - wP (protection) higher across all regimes — LL subband must be protected
%   - Crisis triggered earlier (SSIM < 0.92 instead of 0.85) because DWT
%     degradation is more abrupt when subbands fail
%   - Surplus regime requires SSIM >= 0.99 (vs 0.97 for DCT)
%
% Regimes:
%   Crisis      SNR < 12  OR  prevSSIM < 0.92   -> [2.6, 0.0, 1.2]
%   Stressed    12 <= SNR < 20                   -> [2.3, 0.2, 1.0]
%   Comfortable 20 <= SNR < 32                   -> [2.2, 0.3, 0.5]
%   Surplus     SNR >= 32 AND prevSSIM >= 0.99   -> [1.8, 0.35, 0.35]

    W_CRISIS      = [2.6  0.0  1.2];
    W_STRESSED    = [2.3  0.2  1.0];
    W_COMFORTABLE = [2.2  0.3  0.5];
    W_SURPLUS     = [1.8  0.35 0.35];

    hasPrev = ~isnan(prevSSIM);

    % DWT crisis triggers earlier — SSIM < 0.92 is a red flag for DWT
    if hasPrev && prevSSIM < 0.92
        wQ = W_CRISIS(1); wC = W_CRISIS(2); wP = W_CRISIS(3);
        return;
    end
    if hasPrev && prevSSIM < 0.97 && snr_dB >= 20
        alpha = (prevSSIM - 0.92) / 0.05;
        alpha = max(0, min(1, alpha));
        w = (1 - alpha) * W_CRISIS + alpha * W_STRESSED;
        wQ = w(1); wC = w(2); wP = w(3);
        return;
    end

    % Surplus only when SSIM >= 0.99 (higher bar than DCT's 0.97)
    surplusOK = hasPrev && prevSSIM >= 0.99;
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