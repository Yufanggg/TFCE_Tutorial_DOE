%% Local functions
function z = zscore_checked(x, variableName)
    x = double(x(:));
    if any(~isfinite(x))
        error('%s contains missing or non-finite values.', variableName);
    end
    sx = std(x);
    if sx == 0
        error('%s has zero variance.', variableName);
    end
    z = (x - mean(x)) ./ sx;
end

function permIdx = restricted_index_within_cells(cellID)
    n = numel(cellID);
    permIdx = (1:n)';
    cells = unique(cellID, 'stable');

    for k = 1:numel(cells)
        idx = find(cellID == cells(k));
        permIdx(idx) = idx(randperm(numel(idx)));
    end
end