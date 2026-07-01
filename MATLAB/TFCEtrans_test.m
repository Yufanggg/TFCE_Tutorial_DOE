%% Load R-generated test data
clear; clc;

load('../data/test_tfce_from_R.mat');

% x is time ¡Á channels from R
% MATLAB TFCE function expects channels ¡Á time
tObs = x';

fprintf('tObs size: %d channels ¡Á %d time points\n', size(tObs,1), size(tObs,2));

%% Load channel locations
chanlocs_1020 = readlocs('standard_1005.elc');

chanLabels_32 = {
'Fp1','Fp2', ...
'F7','F3','Fz','F4','F8', ...
'FC5','FC1','FC2','FC6', ...
'T7','C3','Cz','C4','T8', ...
'CP5','CP1','CP2','CP6', ...
'P7','P3','Pz','P4','P8', ...
'PO9','O1','Oz','O2','PO10', ...
'TP9','TP10'
};

allLabels = {chanlocs_1020.labels};
[tf, idx] = ismember(chanLabels_32, allLabels);

if any(~tf)
    error('Missing channels: %s', strjoin(chanLabels_32(~tf), ', '));
end

e_loc = chanlocs_1020(idx);

%% Run MATLAB TFCE
ChN = ept_ChN2(e_loc);
E_H = [0.66 2];

TFCE_Obs = ept_mex_TFCE2D(tObs, ChN, E_H);

%% Plot
figure;
imagesc(TFCE_Obs);
axis xy;
colorbar;

set(gca, ...
    'YTick', 1:length(chanLabels_32), ...
    'YTickLabel', chanLabels_32);

xlabel('Time index');
ylabel('Channel');
title('MATLAB TFCE result from R-generated test data');