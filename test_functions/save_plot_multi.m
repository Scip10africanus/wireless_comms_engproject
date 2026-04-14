function save_plot_multi(figHandle, basePath)
    % SAVE_PLOT_MULTI Saves a figure as .png, .pdf, and .eps
    set(figHandle, 'Color', 'w');
    exportgraphics(figHandle, [basePath '.png'], 'Resolution', 300);
    exportgraphics(figHandle, [basePath '.pdf'], 'ContentType', 'vector');
    exportgraphics(figHandle, [basePath '.eps'], 'ContentType', 'vector');
end
