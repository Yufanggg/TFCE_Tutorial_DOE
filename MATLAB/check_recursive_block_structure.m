%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate recursive hierarchy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function check_recursive_block_structure(blockHierarchy)
%CHECK_RECURSIVE_BLOCK_STRUCTURE
% Verify that sibling blocks under each parent have compatible
% recursive subtree structures.

    nObs = size(blockHierarchy, 1);
    nLevels = size(blockHierarchy, 2);

    allRows = (1:nObs)';

    validate_parent_structure( ...
        blockHierarchy, ...
        allRows, ...
        1, ...
        nLevels);

end
