function bold_smoke_ml_ga_ablation()
% BOLD_SMOKE_ML_GA_ABLATION  Quick smoke test for ML+GA vs ML (GA off) ablation.
%
% Calls bold_compare_baseline_vs_ml with a short frame count. Produces the same
% outputs as a full compare (bold_comparison.png / .mat), including the optional
% third leg when useGaOptimizer is true (default).
%
% Usage (from repo root or ml_addons):
%   >> bold_smoke_ml_ga_ablation

    bold_compare_baseline_vs_ml(struct( ...
        'nFrames', 32, ...
        'useGaReportVarying', true, ...
        'doPlot', false, ...
        'runMlGaOffAblation', false));
end
