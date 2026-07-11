% Recursive multi-level block permutation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function permIndex = recursive_block_permutation( ...
    blockHierarchy, ...
    stream)
%RECURSIVE_BLOCK_PERMUTATION
% Recursively permute all levels of a nested block hierarchy.
%
% INPUTS
% -------------------------------------------------------------------------
% blockHierarchy
%   nObservation x nLevel numeric matrix.
%
%   Current design:
%
%       column 1 = Class
%       column 2 = Student nested within Class
%
%   Individual condition observations form the leaf level.
%
% stream
%   RandStream used for reproducible randomization.
%
% OUTPUT
% -------------------------------------------------------------------------
% permIndex
%   nObservation x 1 permutation index.
%
%   permIndex(targetRow) gives the source observation assigned
%   to the target design row.
%
% Apply the permutation using:
%
%   EEGdata_perm = EEGdata(permIndex,:,:);

    if nargin < 2 || isempty(stream)

        stream = RandStream.getGlobalStream;

    end

    if isempty(blockHierarchy)

        error('blockHierarchy cannot be empty.');

    end

    if ~isnumeric(blockHierarchy)

        error('blockHierarchy must be numeric.');

    end

    if any(isnan(blockHierarchy(:)))

        error( ...
            'blockHierarchy contains missing identifiers.');

    end

    nObs = size(blockHierarchy, 1);
    nLevels = size(blockHierarchy, 2);

    allRows = (1:nObs)';

    permIndex = zeros(nObs, 1);

    permIndex = recurse_block_level( ...
        blockHierarchy, ...
        allRows, ...
        allRows, ...
        1, ...
        nLevels, ...
        permIndex, ...
        stream);

    if ~isequal( ...
            sort(permIndex), ...
            (1:nObs)')

        error( ...
            ['The recursive procedure did not generate ', ...
             'a valid permutation index.']);

    end

end
