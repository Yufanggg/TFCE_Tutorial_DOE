function plot_tfce_results(Obs, Mask, times, e_loc, plot_title)

    sigObs = Obs;
    sigObs(~Mask) = 0;

    figure;

    imagesc(times, 1:size(sigObs,1), sigObs);
    axis xy;
    xlim([-200 800]);

    set(gca, ...
        'YTick', 1:size(sigObs,1), ...
        'YTickLabel', {e_loc.labels}, ...
        'XTick', -200:200:800, ...
        'TickLength', [0 0], ...
        'FontSize', 15, ...
        'FontName', 'Arial');

    xlabel('Time (ms)');
    ylabel('Channel');
    title(plot_title);

    colorbar;

end
