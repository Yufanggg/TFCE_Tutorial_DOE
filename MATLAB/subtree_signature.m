
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct label-independent subtree signature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signature = subtree_signature( ...
    blockHierarchy, ...
    rows, ...
    currentLevel, ...
    nLevels)
%SUBTREE_SIGNATURE
% Construct a label-independent description of a nested subtree.
%
% Compatible with older MATLAB versions:
%   - uses cell arrays instead of string arrays;
%   - does not use strings(), string(), or strjoin().
%
% INPUTS
% -------------------------------------------------------------------------
% blockHierarchy
%   nObservation x nLevel numeric hierarchy matrix.
%
% rows
%   Row indices belonging to the current subtree.
%
% currentLevel
%   Hierarchy level currently being examined.
%
% nLevels
%   Total number of hierarchy levels.
%
% OUTPUT
% -------------------------------------------------------------------------
% signature
%   Character vector describing the subtree structure.

    %% ======================================================
    % Leaf level
    %% ======================================================

    if currentLevel > nLevels

        signature = sprintf( ...
            'Leaf[%d]', ...
            numel(rows));

        return;

    end


    %% ======================================================
    % Identify child blocks at the current level
    %% ======================================================

    childIDs = unique( ...
        blockHierarchy(rows, currentLevel), ...
        'stable');

    nChildren = numel(childIDs);

    childSignatures = cell(nChildren, 1);


    %% ======================================================
    % Recursively obtain the signature of each child subtree
    %% ======================================================

    for i = 1:nChildren

        childRows = rows( ...
            blockHierarchy(rows, currentLevel) ...
            == childIDs(i));

        childSignatures{i} = subtree_signature( ...
            blockHierarchy, ...
            childRows, ...
            currentLevel + 1, ...
            nLevels);

    end


    %% ======================================================
    % Sort signatures
    %
    % This makes the final signature independent of the
    % original ordering or labels of the child blocks.
    %% ======================================================

    childSignatures = sort(childSignatures);


    %% ======================================================
    % Combine child signatures into one character vector
    %% ======================================================

    combinedSignature = '';

    for i = 1:nChildren

        if i == 1

            combinedSignature = childSignatures{i};

        else

            combinedSignature = [ ...
                combinedSignature, ...
                ',', ...
                childSignatures{i}]; %#ok<AGROW>

        end

    end


    %% ======================================================
    % Construct signature for the current hierarchy level
    %% ======================================================

    signature = sprintf( ...
        'Level%d{%s}', ...
        currentLevel, ...
        combinedSignature);

end