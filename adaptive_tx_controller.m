function [newParams, adaptInfo] = adaptive_tx_controller(currentParams, feedback)
% ADAPTIVE_TX_CONTROLLER - Adjust TX parameters based on RX feedback
%
% Implements closed-loop adaptation:
%   - Reads RX feedback (quality metrics, layer status)
%   - Adjusts modulation orders, subcarrier allocation
%   - Returns updated params for next transmission
%
% Inputs:
%   currentParams - Current TX parameters
%   feedback      - Feedback from RX (from two_layer_rx_pipeline)
%
% Outputs:
%   newParams - Updated parameters for next frame
%   adaptInfo - What changed and why

    newParams = currentParams;
    adaptInfo = struct();
    adaptInfo.changes = {};
    adaptInfo.reason = {};

    %% Check if feedback is valid
    if isempty(feedback) || ~isstruct(feedback)
        adaptInfo.action = 'none';
        adaptInfo.reason{end+1} = 'No valid feedback';
        return;
    end

    %% Urgent adaptation
    if isfield(feedback, 'urgentAdapt') && feedback.urgentAdapt
        adaptInfo.urgent = true;

        % Aggressive protection increase
        [newParams, change] = decrease_modulation(newParams, 'base', 2);
        if change
            adaptInfo.changes{end+1} = 'base_mod_down_urgent';
            adaptInfo.reason{end+1} = 'Urgent: base layer critical';
        end

        [newParams, change] = decrease_modulation(newParams, 'enhancement', 2);
        if change
            adaptInfo.changes{end+1} = 'enh_mod_down_urgent';
            adaptInfo.reason{end+1} = 'Urgent: quality below threshold';
        end

        adaptInfo.action = 'urgent_protect';
        return;
    end

    adaptInfo.urgent = false;

    %% Base layer adaptation
    if isfield(feedback, 'baseStatus')
        switch feedback.baseStatus
            case 'poor'
                [newParams, change] = decrease_modulation(newParams, 'base', 1);
                if change
                    adaptInfo.changes{end+1} = 'base_mod_down';
                    adaptInfo.reason{end+1} = 'Base layer poor quality';
                end

            case 'good'
                % Only increase if quality targets met
                if isfield(feedback, 'qualityMet') && feedback.qualityMet
                    [newParams, change] = increase_modulation(newParams, 'base', 1);
                    if change
                        adaptInfo.changes{end+1} = 'base_mod_up';
                        adaptInfo.reason{end+1} = 'Base good + targets met';
                    end
                end
        end
    end

    %% Enhancement layer adaptation
    if isfield(feedback, 'enhancementStatus')
        switch feedback.enhancementStatus
            case 'poor'
                [newParams, change] = decrease_modulation(newParams, 'enhancement', 1);
                if change
                    adaptInfo.changes{end+1} = 'enh_mod_down';
                    adaptInfo.reason{end+1} = 'Enhancement layer poor';
                end

            case 'good'
                if isfield(feedback, 'qualityMet') && feedback.qualityMet
                    [newParams, change] = increase_modulation(newParams, 'enhancement', 1);
                    if change
                        adaptInfo.changes{end+1} = 'enh_mod_up';
                        adaptInfo.reason{end+1} = 'Enhancement good + targets met';
                    end
                end
        end
    end

    %% Quality-based fine tuning
    if isfield(feedback, 'ssim') && ~isnan(feedback.ssim)
        if feedback.ssim < 0.8 && strcmp(feedback.enhancementStatus, 'marginal')
            [newParams, change] = decrease_modulation(newParams, 'enhancement', 1);
            if change
                adaptInfo.changes{end+1} = 'enh_mod_down_ssim';
                adaptInfo.reason{end+1} = sprintf('SSIM=%.2f below 0.8', feedback.ssim);
            end
        end
    end

    %% Summarize
    if isempty(adaptInfo.changes)
        adaptInfo.action = 'maintain';
    else
        adaptInfo.action = 'adapted';
    end
end

%% Modulation Adjustment Functions

function [params, changed] = decrease_modulation(params, layer, steps)
% DECREASE_MODULATION - Reduce modulation order for more protection

    modLevels = [2, 4, 16, 64, 256];  % BPSK, QPSK, 16QAM, 64QAM, 256QAM
    changed = false;

    if strcmp(layer, 'base')
        currentMod = params.baseMod;
        currentIdx = find(modLevels == currentMod, 1);
        if isempty(currentIdx), currentIdx = 2; end

        newIdx = max(1, currentIdx - steps);
        if newIdx ~= currentIdx
            params.baseMod = modLevels(newIdx);
            changed = true;
        end

    elseif strcmp(layer, 'enhancement')
        currentMod = params.enhancementMod;
        currentIdx = find(modLevels == currentMod, 1);
        if isempty(currentIdx), currentIdx = 4; end

        newIdx = max(2, currentIdx - steps);  % Enhancement min = QPSK
        if newIdx ~= currentIdx
            params.enhancementMod = modLevels(newIdx);
            changed = true;
        end
    end
end

function [params, changed] = increase_modulation(params, layer, steps)
% INCREASE_MODULATION - Increase modulation order for more throughput

    modLevels = [2, 4, 16, 64, 256];
    changed = false;

    if strcmp(layer, 'base')
        currentMod = params.baseMod;
        currentIdx = find(modLevels == currentMod, 1);
        if isempty(currentIdx), currentIdx = 2; end

        newIdx = min(3, currentIdx + steps);  % Base max = 16QAM
        if newIdx ~= currentIdx
            params.baseMod = modLevels(newIdx);
            changed = true;
        end

    elseif strcmp(layer, 'enhancement')
        currentMod = params.enhancementMod;
        currentIdx = find(modLevels == currentMod, 1);
        if isempty(currentIdx), currentIdx = 4; end

        newIdx = min(5, currentIdx + steps);  % Enhancement max = 256QAM
        if newIdx ~= currentIdx
            params.enhancementMod = modLevels(newIdx);
            changed = true;
        end
    end
end
