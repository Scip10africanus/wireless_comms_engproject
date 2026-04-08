function save_image_figure_multi(figHandle, basePath)
    % SAVE_IMAGE_FIGURE_MULTI Saves a figure as .png and .pdf
    set(figHandle, 'Color', 'w');
    exportgraphics(figHandle, [basePath '.png'], 'Resolution', 300);
    exportgraphics(figHandle, [basePath '.pdf'], 'ContentType', 'image');
end
