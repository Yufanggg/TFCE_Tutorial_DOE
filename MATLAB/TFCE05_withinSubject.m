%% ==========================================================
% TFCE analysis
% Within-subject design using long-format data
%
% EEGdata:
%   Rows x Channels x Time
%   2*nSubj x nChan x nTime
%
% designTable:
%   Subject
%   CondCode    % -1 = Control, 1 = Treatment
%   CondName
%
% Model:
%   EEG ~ Condition + (1|Subject)
%
% Permutation:
%   Rearrange condition labels within each subject
%   EEGdata itself is never permuted
%% ==========================================================

clear; clc; close all;
fprintf('\nStarting TFCE analysis...\n');

%% Load data

load('../data/05_simulated_within_subject_EEG.mat', ...
     'EEGdata', 'designTable');

times = -200:4:800;

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

%% Basic checks

[nRows, nChan, nTime] = size(EEGdata);

Subject  = designTable.Subject;
CondCode = designTable.CondCode;

if height(designTable) ~= nRows
    error('Rows in designTable must match rows in EEGdata.');
end

if length(times) ~= nTime
    error('Length of times does not match number of time points.');
end

if length(e_loc) ~= nChan
    error('Number of channel locations does not match number of channels.');
end

subjects = unique(Subject);
nSubj = length(subjects);

for s = 1:nSubj

    idxSubj = find(Subject == subjects(s));

    if numel(idxSubj) ~= 2
        error('Each subject must have exactly two rows.');
    end

    if ~all(sort(CondCode(idxSubj)) == [-1; 1])
        error('Each subject must have one Control (-1) and one Treatment (1).');
    end
end

%% Observed design variables
SubjectLME = categorical(Subject);

%% Step 1: Observed LME t-map

fprintf('Step 1: Computing observed t-statistic map...\n');

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        EEG = double(squeeze(EEGdata(:, ch, tpoint)));

        tbl = table(EEG, CondCode, SubjectLME, ...
            'VariableNames', {'EEG','Condition','Subject'});

        lme = fitlme(tbl, 'EEG ~ Condition + (1|Subject)');

        t_Obs(ch,tpoint) = lme.Coefficients.tStat(2);
    end
end

fprintf('Observed t-statistic map completed.\n');

%% Step 2: Observed TFCE map

fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');

%% Step 3: Permutation by rearranging labels within subject

nPerm = 999;
TFCE_permMax = nan(nPerm,1);

fprintf('Step 3: Starting within-subject label permutation: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    perm_t = nan(nChan, nTime);

    CondCode_perm = CondCode;

    % Rearrange condition labels independently within each subject
    for s = 1:nSubj

        idxSubj = find(Subject == subjects(s));

        permOrder = randperm(numel(idxSubj));

        CondCode_perm(idxSubj) = CondCode(idxSubj(permOrder));
    end


    for ch = 1:nChan
        for tpoint = 1:nTime

            EEG = double(squeeze(EEGdata(:, ch, tpoint)));

            tbl = table(EEG, CondCode_perm, SubjectLME, ...
                'VariableNames', {'EEG','Condition','Subject'});

            lme = fitlme(tbl, 'EEG ~ Condition + (1|Subject)');

            perm_t(ch,tpoint) = lme.Coefficients.tStat(2);
        end
    end

    TFCE_perm = ept_mex_TFCE2D(perm_t, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

    fprintf('Permutation %d / %d completed\n', p, nPerm);
end

fprintf('\nPermutation testing completed.\n');

%% Step 4: TFCE correction

fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;

maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);

maxTFCEcrit = maxTFCE(round(nPerm*(1-alpha)));


Mask = abs(TFCE_Obs) >= maxTFCEcrit;

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        P_Values(ch, tpoint) = ...
            (sum(TFCE_permMax >= abs(TFCE_Obs(ch, tpoint))) + 1) / ...
            (nPerm + 1);

    end
end

fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', maxTFCEcrit);

%% Step 5: Store results

Results = struct();

Results.tObs          = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_Null    = TFCE_permMax;
Results.maxTFCEcrit  = maxTFCEcrit;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.alpha        = alpha;
Results.nPerm        = nPerm;
Results.model        = 'EEG ~ Condition + (1|Subject)';
Results.test         = 'Treatment - Control fixed effect';
Results.permutation  = 'Within-subject condition-label permutation';

%% Step 8: Save results

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/05_TFCE_within_subject_LME_results.mat', ...
     'Results', ...
     'nChan', ...
     'times', ...
     'e_loc');

disp('Within-subject LME TFCE analysis completed and saved.');

%% Step 6: Plot significant observed t-values
clear all; clc; close all
load('../results/05_TFCE_within_subject_LME_results.mat');

mT = Results.tObs;
mT(~Results.Mask) = 0;

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
title('TFCE-corrected t-Obs: Treatment - Control');
colorbar;

%% Step 7: Plot observed TFCE map

figure;
imagesc(times, 1:nChan, Results.TFCE_Obs);
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

