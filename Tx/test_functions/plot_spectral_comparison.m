function plot_spectral_comparison(img_orig, img_recon, basePath, custom_title_suffix)
    % PLOT_SPECTRAL_COMPARISON
    % Generate and save:
    %   1) 2D log power spectrum heatmap comparison
    %   2) 1D radially averaged power spectrum (RAPS) comparison

    if nargin < 4
        custom_title_suffix = '';
    end

    [LP_orig, P_orig]   = compute_log_power_spectrum(img_orig);
    [LP_recon, P_recon] = compute_log_power_spectrum(img_recon);

    % -----------------------------
    % Style settings
    % -----------------------------
    textColor  = [0.15, 0.15, 0.15];
    axisColor  = [0.15, 0.15, 0.15];
    gridColor  = [0.85, 0.85, 0.85];

    origColor  = [0.20, 0.20, 0.22];   % charcoal
    reconColor = [0.35, 0.47, 0.58];   % muted blue-gray

    % =============================
    % 1) 2D Log Power Spectrum
    % =============================
    fig2d = figure( ...
        'Name', ['2D Log Power Spectrum ' custom_title_suffix], ...
        'Position', [100, 100, 1000, 450], ...
        'Color', 'w', ...
        'InvertHardcopy', 'off', ...
        'Visible', 'off');

    % More restrained scientific colormap
    colormap(fig2d, bone());

    % Shared robust color scale
    vmax = max(prctile(LP_orig(:), 99.5), prctile(LP_recon(:), 99.5));
    vmin = min(min(LP_orig(:)), min(LP_recon(:)));

    tl = tiledlayout(fig2d, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile;
    imagesc(ax1, LP_orig);
    axis(ax1, 'image');
    axis(ax1, 'off');
    caxis(ax1, [vmin, vmax]);
    title(ax1, 'Original', 'Color', textColor, 'FontWeight', 'normal');

    ax2 = nexttile;
    imagesc(ax2, LP_recon);
    axis(ax2, 'image');
    axis(ax2, 'off');
    caxis(ax2, [vmin, vmax]);
    title(ax2, 'Reconstructed', 'Color', textColor, 'FontWeight', 'normal');

    cb = colorbar(ax2);
    cb.Color = textColor;

    if ~isempty(custom_title_suffix)
        sgtitle(tl, ['2D Log Power Spectrum Comparison ' custom_title_suffix], ...
            'Color', textColor, ...
            'FontWeight', 'normal');
    else
        sgtitle(tl, '2D Log Power Spectrum Comparison', ...
            'Color', textColor, ...
            'FontWeight', 'normal');
    end

    save_image_figure_multi(fig2d, [basePath '_2D_Spectrum']);
    close(fig2d);

    % =============================
    % 2) 1D RAPS
    % =============================
    [r_orig,  raps_orig]  = radial_average_power_spectrum(P_orig);
    [r_recon, raps_recon] = radial_average_power_spectrum(P_recon);

    fig1d = figure( ...
        'Name', ['1D RAPS ' custom_title_suffix], ...
        'Position', [100, 100, 620, 500], ...
        'Color', 'w', ...
        'InvertHardcopy', 'off', ...
        'Visible', 'off');

    % Log scale for readability
    log_raps_orig  = log10(raps_orig  + 1e-10);
    log_raps_recon = log10(raps_recon + 1e-10);

    ax = axes(fig1d);
    hold(ax, 'on');
    grid(ax, 'on');

    % Use solid lines for both; distinguish by color and slight width difference
    plot(ax, r_orig, log_raps_orig, ...
        '-', ...
        'Color', origColor, ...
        'LineWidth', 2.2, ...
        'DisplayName', 'Original');

    plot(ax, r_recon, log_raps_recon, ...
        '-', ...
        'Color', reconColor, ...
        'LineWidth', 2.0, ...
        'DisplayName', 'Reconstructed');

    xlabel(ax, 'Radial Frequency Bin', 'Color', textColor);
    ylabel(ax, 'Log_{10}(Power)', 'Color', textColor);

    if ~isempty(custom_title_suffix)
        title(ax, ['1D Radially Averaged Power Spectrum ' custom_title_suffix], ...
            'Color', textColor, ...
            'FontWeight', 'normal');
    else
        title(ax, '1D Radially Averaged Power Spectrum', ...
            'Color', textColor, ...
            'FontWeight', 'normal');
    end

    ax.Color     = 'w';
    ax.FontName  = 'Times New Roman';
    ax.FontSize  = 11;
    ax.LineWidth = 1;
    ax.Box       = 'on';
    ax.XColor    = axisColor;
    ax.YColor    = axisColor;
    ax.GridColor = gridColor;
    ax.GridAlpha = 0.6;
    ax.Layer     = 'top';

    lgd = legend(ax, 'Location', 'northeast');
    lgd.Box = 'off';
    lgd.FontSize = 10;
    lgd.TextColor = textColor;
    lgd.Color = 'none';

    save_plot_multi(fig1d, [basePath '_1D_RAPS']);
    close(fig1d);
end