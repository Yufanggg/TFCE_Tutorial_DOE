%% ==========================================================
% TFCE analysis for simple between-subject EEG design
%
% Loads only:
%   EEGdata
%   designTable
%
% GroupLabel:
%   Control -1 vs Treatment 1
%
% EEGdata dimensions:
%   Subjects x Channels x Time
%% ==========================================================

clear; clc; close all;
fprintf('\nStarting TFCE analysis...\n');

%% ==========================================================
% Load dataset
%% ==========================================================

load('../data/01_simulated_between_subject_EEG.mat', ...
     'EEGdata', 'designTable');

group = designTable.GroupCode;

times = -200:4:800;

%% ==========================================================
% Load channel locations
%% ==========================================================

chanlocs_1020 = readlocs('standard_1005.elc');

chanLabels_32 = {
'Fp1','Fp2',...
'F7','F3','Fz','F4','F8',...
'FC5','FC1','FC2','FC6',...
'T7','C3','Cz','C4','T8',...
'CP5','CP1','CP2','CP6',...
'P7','P3','Pz','P4','P8',...
'PO9','O1','Oz','O2','PO10',...
'TP9','TP10'
};

allLabels = {chanlocs_1020.labels};
[tf, idx] = ismember(chanLabels_32, allLabels);

if any(~tf)
    error('Missing channels: %s', strjoin(chanLabels_32(~tf), ', '));
end

e_loc = chanlocs_1020(idx);

%% ==========================================================
% Basic checks
%% ==========================================================

[nSub, nChan, nTime] = size(EEGdata);

if length(group) ~= nSub
    error('Length of group must match number of subjects.');
end

if length(times) ~= nTime
    error('Length of times must match number of time points.');
end

if length(e_loc) ~= nChan
    error('Number of channel locations must match number of channels.');
end

controlIdx   = group == -1;
treatmentIdx = group == 1;

if sum(controlIdx) == 0 || sum(treatmentIdx) == 0
    error('Group must contain both -1 = Control and 1 = Treatment.');
end

%% ==========================================================
% Step 1: Observed t-statistic map
%% ==========================================================
fprintf('Step 1: Computing observed t-statistic map...\n');

X = group;
tObs = zeros(nChan, nTime);

for ch = 1:nChan
    for t = 1:nTime

        EEG_local = double(EEGdata(:, ch, t));

        lm_local = fitlm(X, EEG_local);

        % Coefficient 2 = Group effect
        tObs(ch, t) = lm_local.Coefficients.tStat(2);

    end
end

fprintf('Observed t-statistic map completed.\n');
%% ==========================================================
% Step 2: TFCE transform of observed t-map
%% ==========================================================
fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66 2];

TFCE_Obs = ept_mex_TFCE2D(tObs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');
%% ==========================================================
% Step 3: Permutation testing
%% ==========================================================

nPerm = 999;
TFCE_permMax = zeros(nPerm, 1);

fprintf('Step 3: Starting permutation testing: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    permX = X(randperm(nSub));

    permT = zeros(nChan, nTime);

    for ch = 1:nChan

        for t = 1:nTime

            EEG_local = double(EEGdata(:, ch, t));

            lm_perm = fitlm(permX, EEG_local);

            % Coefficient 2 = permuted Group effect
            permT(ch, t) = lm_perm.Coefficients.tStat(2);

        end
    end
    
    fprintf('At the %dth permutation\n', p);

    TFCE_perm = ept_mex_TFCE2D(permT, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end

fprintf('\nPermutation testing completed.\n');

%% ==========================================================
% Step 4: TFCE-corrected significance
%% ==========================================================

fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;
nPerm = length(TFCE_permMax);
maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);
maxTFCEcrit = maxTFCE(round(nPerm*(1-Alpha)));

Mask = abs(TFCE_Obs) >= maxTFCEcrit;

P_Values = zeros(nChan, nTime);

for ch = 1:nChan
    for t = 1:nTime

        P_Values(ch,t) = ...
            (sum(TFCE_permMax >= abs(TFCE_Obs(ch,t))) + 1) / ...
            (nPerm + 1);

    end
end

fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', maxTFCEcrit);
%% ==========================================================
% Step 5: Store results
%% ==========================================================
fprintf('Step 5: Storing results...\n');

Results = struct();

Results.tObs         = tObs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_Null    = TFCE_permMax;
Results.maxTFCEcrit     = maxTFCEcrit;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.alpha        = alpha;
Results.nPerm        = nPerm;

fprintf('Results stored.\n');
%% ==========================================================
% Step 6: Plot TFCE-corrected significant t-values
%% ==========================================================

sigT = tObs;
sigT(~Mask) = 0;

figure;

imagesc(times, 1:nChan, sigT);
axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {e_loc.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('TFCE-corrected Significant Effects');

colorbar;

%% ==========================================================
% Step 7: Plot TFCE values
%% ==========================================================

figure;

imagesc(times, 1:nChan, TFCE_Obs);
axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {e_loc.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed TFCE Map');

colorbar;

%% ==========================================================
% Step 8: Save TFCE results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/01_TFCE_between_subject_results.mat', ...
     'Results', ...
     'tObs', ...
     'TFCE_Obs', ...
     'TFCE_permMax', ...
     'critTFCE', ...
     'P_Values', ...
     'Mask', ...
     'times', ...
     'e_loc');

disp('TFCE analysis completed and results saved.');