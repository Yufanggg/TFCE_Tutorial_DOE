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

load('../Data/05_simulated_within_subject_EEG.mat', ...
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

t_Obs = nan(nChan, nTime);

beta_subject = nan(nSubj, 1);
var_beta = nan(nSubj, 1);
beta_standardizeds = nan(nChan, nTime, nSubj);

tPerm = nan(nChan, nTime);
for ch = 1:nChan
    for tpoint = 1:nTime
        EEG = double(squeeze(EEGdata(:, ch, tpoint)));
        for s = 1:nSubj
            idxSubj = Subject == subjects(s);
            lm_subject = fitlm(CondCode(idxSubj), EEG(idxSubj));  
            beta_subject(s) = lm_subject.Coefficients.Estimate(2);
            var_beta(s) = lm_subject.Coefficients.SE(2)^2;
            
            % Method-of-moments estimate under H0
            sigma2_between = max(0, mean(beta_subject.^2 - var_beta));
            % Standardized participant coefficients
            beta_standardized = beta_subject ./sqrt(sigma2_between + var_beta);
            % Observed statistic
            t_Obs(ch, tpoint) = mean(beta_standardized);
            
            % save for the sake of permutation 
            beta_standardizeds(ch, tpoint, :) = beta_standardized;
        end
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
TFCE_permMaxPos = nan(nPerm,1);
TFCE_permMinNeg = nan(nPerm,1);
fprintf('Step 3: Starting permutation: %d permutations...\n', nPerm);

flipSigns = 2 * (rand(nSubj, nPerm) > 0.5) - 1;
parfor p = 1:nPerm
    perm_t = nan(nChan, nTime);
    
    flipSign = flipSigns(:, p);
    for ch = 1:nChan
        for tpoint = 1:nTime
            beta_standardized = beta_standardizeds(ch, tpoint, :);
            perm_t(ch, tpoint) = mean(beta_standardized .* flipSign);       
        end
    end
    
    TFCE_perm = ept_mex_TFCE2D(perm_t, ChN, E_H);
    
    TFCE_permMaxPos(p) = max(TFCE_perm(:));
    TFCE_permMinNeg(p) = max(TFCE_perm(:));

    progressbar(p/nPerm)
end
%% Step 4: TFCE correction

fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;

maxTFCEPos = sort([TFCE_permMaxPos; max(TFCE_Obs(:))]);
minTFCENeg = sort([TFCE_permMinNeg; min(TFCE_Obs(:))]);

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime
        
        pr = (1 + sum(maxTFCEPos >= TFCE_Obs(ch,tp))) / (nPerm + 1);
        pl = (1 + sum(minTFCENeg <= TFCE_Obs(ch,tp))) / (nPerm + 1);        
        P_Values(ch,tp) = min(1, 2 * min(pr, pl));
    end
end

fprintf('TFCE-corrected significance completed.\n');

%% Step 5: Store results

Results = struct();

Results.tObs         = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.maxTFCEPos   = maxTFCEPos;
Results.minTFCENeg   = minTFCENeg;
Results.P_Values     = P_Values;
Results.alpha        = alpha;
Results.nPerm        = nPerm;
Results.model        = 'EEG ~ Condition + (1|Subject)';
Results.test         = 'Treatment - Control fixed effect';
Results.permutation  = 'Within-subject condition-label permutation';

%% Step 8: Save results

if ~exist('../Results', 'dir')
    mkdir('../Results');
end

save('../Results/05_TFCE_within_subject_LME_results_Alternative.mat', ...
     'Results', ...
     'nChan', ...
     'times', ...
     'e_loc');

disp('Within-subject LME TFCE analysis completed and saved.');

%% Step 6: Plot significant observed t-values
clear all; clc; close all
load('../Results/05_TFCE_within_subject_LME_results_Alternative.mat');

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
cb = colorbar;
ylabel(cb, 't-value', ...
    'FontSize', 15, ...
    'FontName', 'Arial');

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
cb = colorbar;
ylabel(cb, 'TFCE-value', ...
    'FontSize', 15, ...
    'FontName', 'Arial');

