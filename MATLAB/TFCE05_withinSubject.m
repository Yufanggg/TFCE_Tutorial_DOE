%% ==========================================================
% TFCE analysis
% Within-subject EEG/ERP design using linear mixed model
%
% Data format:
%   data = Subjects x Conditions x Channels x Time
%
% Conditions:
%   Condition 1 = Control
%   Condition 2 = Treatment
%
% Model:
%   EEG ~ Condition + (1|Subject)
%
% Test:
%   Treatment - Control fixed effect
%
% Permutation:
%   Within-subject condition-label flipping
%% ==========================================================

clear; clc; close all;
fprintf('\nStarting TFCE analysis...\n');
%% ==========================================================
% Load data
%% ==========================================================

load('../data/05_simulated_within_subject_EEG.mat');

%% ==========================================================
% Load channel locations
%% ==========================================================

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

%% ==========================================================
% Basic checks
%% ==========================================================

[nSubj, nCond, nChan, nTime] = size(data);

if nCond ~= 2
    error('Expected data format: Subjects x 2 Conditions x Channels x Time');
end

if length(times) ~= nTime
    error('Length of times does not match number of time points.');
end

if length(e_loc) ~= nChan
    error('Number of channel locations does not match number of channels.');
end

%% ==========================================================
% Long-format design variables for fitlme
%% ==========================================================

% For old MATLAB compatibility, use kron instead of repelem
Subject = kron((1:nSubj)', ones(nCond,1));

% 0 = Control, 1 = Treatment
Condition = repmat([0; 1], nSubj, 1);

Subject = categorical(Subject);

%% ==========================================================
% Step 1: Observed LME t-map
%% ==========================================================
fprintf('Step 1: Computing observed t-statistic map...\n');
t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        controlVals   = squeeze(data(:,1,ch,tpoint));
        treatmentVals = squeeze(data(:,2,ch,tpoint));

        EEG = reshape([controlVals treatmentVals]', [], 1);

        tbl = table(EEG, Condition, Subject, ...
            'VariableNames', {'EEG','Condition','Subject'});

        lme = fitlme(tbl, 'EEG ~ Condition + (1|Subject)');

        % Row 2 = Treatment - Control fixed effect
        t_Obs(ch,tpoint) = lme.Coefficients.tStat(2);

    end
end
fprintf('Observed t-statistic map completed.\n');
%% ==========================================================
% Step 2: Observed TFCE map
%% ==========================================================
fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');
%% ==========================================================
% Step 3: Permutation using within-subject condition flipping
%% ==========================================================

nPerm = 999;
TFCE_permMax = nan(nPerm,1);

fprintf('Step 3: Starting permutation testing: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    perm_t = nan(nChan, nTime);

    % Randomly flip Control/Treatment labels within each subject
    flip = randi(nSubj, 1);

    for ch = 1:nChan
        for tpoint = 1:nTime
            
            permControl   = squeeze(data(:,1,ch,tpoint));
            permTreatment  = squeeze(data(:,2,ch,tpoint));

            for s = 1:nSubj
                if flip(s)
                    tmp = permControl(s);
                    permControl(s)   = permTreatment(s);
                    permTreatment(s) = tmp;
                end
            end

            EEG = reshape([permControl permTreatment]', [], 1);

            tbl = table(EEG, Condition, Subject, ...
                'VariableNames', {'EEG','Condition','Subject'});

            lme = fitlme(tbl, 'EEG ~ Condition + (1|Subject)');

            perm_t(ch,tpoint) = lme.Coefficients.tStat(2);

        end
    end
    
    fprintf('At the %dth permutation\n', p);

    TFCE_perm = ept_mex_TFCE2D(perm_t, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end

fprintf('\nPermutation testing completed.\n');
%% ==========================================================
% Step 4: TFCE correction
%% ==========================================================
fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;

critTFCE = prctile(TFCE_permMax, 100 * (1 - alpha));

Mask = abs(TFCE_Obs) >= critTFCE;

P_Values = nan(nChan, nTime);

for i = 1:numel(TFCE_Obs)

    P_Values(i) = ...
        (sum(TFCE_permMax >= abs(TFCE_Obs(i))) + 1) / ...
        (nPerm + 1);

end

fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', critTFCE);
%% ==========================================================
% Step 5: Store results
%% ==========================================================
fprintf('Step 5: Storing results...\n');

Results = struct();

Results.Obs          = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_Null    = TFCE_permMax;
Results.critTFCE     = critTFCE;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.alpha        = alpha;
Results.nPerm        = nPerm;
Results.model        = 'EEG ~ Condition + (1|Subject)';
Results.test         = 'Treatment - Control fixed effect';

fprintf('Results stored.\n');
%% ==========================================================
% Step 6: Plot significant observed t-values
%% ==========================================================

mT = t_Obs;
mT(~Mask) = 0;

figure;

imagesc(times, 1:nChan, mT);
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
title('TFCE-corrected LME Effect: Treatment - Control');

colorbar;

%% ==========================================================
% Step 7: Plot observed TFCE map
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
title('Observed TFCE Map: LME Treatment Effect');

colorbar;

%% ==========================================================
% Step 8: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/05_TFCE_within_subject_LME_results.mat', ...
     'Results', ...
     't_Obs', ...
     'TFCE_Obs', ...
     'TFCE_permMax', ...
     'critTFCE', ...
     'P_Values', ...
     'Mask', ...
     'times', ...
     'e_loc');

disp('Within-subject LME TFCE analysis completed and saved.');
