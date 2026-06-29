%% ==========================================================
% TFCE analysis
% Between-subject 2 x 2 design without interaction
%
% Model:
%   EEG ~ var1 + var2
%
% Tests:
%   Main effect of var1, adjusted for var2
%   Main effect of var2, adjusted for var1
%
% Factor coding:
%   var1 = -1 / +1
%   var2 = -1 / +1
%
% Permutation:
%   Freedman-Lane
%% ==========================================================

clear; clc; close all;
fprintf('\nStarting TFCE analysis...\n');

%% ==========================================================
% Load simulated data
%% ==========================================================

load('../data/03_simulated_between_subject_2by2_EEG.mat');

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

[nSubj, nChan, nTime] = size(data);

var1 = var1(:);
var2 = var2(:);

if length(var1) ~= nSubj
    error('Length of var1 does not match number of subjects.');
end

if length(var2) ~= nSubj
    error('Length of var2 does not match number of subjects.');
end

if length(times) ~= nTime
    error('Length of times does not match number of time points.');
end

if length(e_loc) ~= nChan
    error('Number of channel locations does not match number of channels.');
end

%% ==========================================================
% Design matrices for fitlm
%% ==========================================================

% Do not include intercept.
% fitlm adds the intercept automatically.
%
% fitlm(X_full, Y) fits:
%   EEG ~ Intercept + var1 + var2
%
% Coefficient rows:
%   1 = Intercept
%   2 = var1
%   3 = var2

X_full = [var1, var2];

coefRow_var1 = 2;
coefRow_var2 = 3;

%% ==========================================================
% Step 1: Observed t-maps
%% ==========================================================
fprintf('Step 1: Computing observed t-statistic map...\n');

t_Obs_var1 = zeros(nChan, nTime);
t_Obs_var2 = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        Y = double(data(:, ch, tp));

        lm_full = fitlm(X_full, Y);

        t_Obs_var1(ch, tp) = lm_full.Coefficients.tStat(coefRow_var1);
        t_Obs_var2(ch, tp) = lm_full.Coefficients.tStat(coefRow_var2);

    end
end

fprintf('Observed t-statistic map completed.\n');
%% ==========================================================
% Step 2: Observed TFCE maps
%% ==========================================================
fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs_var1 = ept_mex_TFCE2D(t_Obs_var1, ChN, E_H);
TFCE_Obs_var2 = ept_mex_TFCE2D(t_Obs_var2, ChN, E_H);

fprintf('Observed TFCE map completed.\n');
%% ==========================================================
% Step 3: Freedman-Lane permutation
%% ==========================================================

nPerms = 999;

TFCE_permMax_var1 = nan(nPerms, 1);
TFCE_permMax_var2 = nan(nPerms, 1);

fprintf('Step 3: Starting permutation testing: %d permutations...\n', nPerms);

% Reduced model for testing var1:
%   EEG ~ var2
X_red_var1 = var2;

% Reduced model for testing var2:
%   EEG ~ var1
X_red_var2 = var1;

parfor p = 1:nPerms

    perm_t_var1 = zeros(nChan, nTime);
    perm_t_var2 = zeros(nChan, nTime);

    perm_idx_var1 = randperm(nSubj);
    perm_idx_var2 = randperm(nSubj);

    for ch = 1:nChan
        for tp = 1:nTime

            Y = double(data(:, ch, tp));

            %% ----------------------------------------------
            % Test var1 adjusted for var2
            %% ----------------------------------------------

            lm_red_var1 = fitlm(X_red_var1, Y);

            Y_hat_red_var1 = lm_red_var1.Fitted;
            resid_red_var1 = lm_red_var1.Residuals.Raw;

            Y_perm_var1 = Y_hat_red_var1 + resid_red_var1(perm_idx_var1);

            lm_perm_var1 = fitlm(X_full, Y_perm_var1);

            perm_t_var1(ch, tp) = ...
                lm_perm_var1.Coefficients.tStat(coefRow_var1);

            %% ----------------------------------------------
            % Test var2 adjusted for var1
            %% ----------------------------------------------

            lm_red_var2 = fitlm(X_red_var2, Y);

            Y_hat_red_var2 = lm_red_var2.Fitted;
            
            resid_red_var2 = lm_red_var2.Residuals.Raw;

            Y_perm_var2 = Y_hat_red_var2 + resid_red_var2(perm_idx_var2);

            lm_perm_var2 = fitlm(X_full, Y_perm_var2);

            perm_t_var2(ch, tp) = ...
                lm_perm_var2.Coefficients.tStat(coefRow_var2);

        end
    end
    
    fprintf('At the %dth permutation\n', p);

    TFCE_perm_var1 = ept_mex_TFCE2D(perm_t_var1, ChN, E_H);
    TFCE_perm_var2 = ept_mex_TFCE2D(perm_t_var2, ChN, E_H);

    TFCE_permMax_var1(p) = max(abs(TFCE_perm_var1(:)));
    TFCE_permMax_var2(p) = max(abs(TFCE_perm_var2(:)));

end

fprintf('\nPermutation testing completed.\n');
%% ==========================================================
% Step 4: TFCE-corrected significance
%% ==========================================================

fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;

critTFCE_var1 = prctile(TFCE_permMax_var1, 100 * (1 - alpha));
critTFCE_var2 = prctile(TFCE_permMax_var2, 100 * (1 - alpha));

Mask_var1 = abs(TFCE_Obs_var1) >= critTFCE_var1;
Mask_var2 = abs(TFCE_Obs_var2) >= critTFCE_var2;

P_Values_var1 = nan(nChan, nTime);
P_Values_var2 = nan(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        P_Values_var1(ch, tp) = ...
            (sum(TFCE_permMax_var1 >= abs(TFCE_Obs_var1(ch, tp))) + 1) / ...
            (nPerms + 1);

        P_Values_var2(ch, tp) = ...
            (sum(TFCE_permMax_var2 >= abs(TFCE_Obs_var2(ch, tp))) + 1) / ...
            (nPerms + 1);

    end
end

fprintf('TFCE-corrected significance completed.\n');
%% ==========================================================
% Step 5: Store results
%% ==========================================================
fprintf('Step 5: Storing results...\n');

Results = struct();

Results.Obs_var1       = t_Obs_var1;
Results.TFCE_Obs_var1  = TFCE_Obs_var1;
Results.TFCE_Null_var1 = TFCE_permMax_var1;
Results.critTFCE_var1  = critTFCE_var1;
Results.P_Values_var1  = P_Values_var1;
Results.Mask_var1      = Mask_var1;

Results.Obs_var2       = t_Obs_var2;
Results.TFCE_Obs_var2  = TFCE_Obs_var2;
Results.TFCE_Null_var2 = TFCE_permMax_var2;
Results.critTFCE_var2  = critTFCE_var2;
Results.P_Values_var2  = P_Values_var2;
Results.Mask_var2      = Mask_var2;

Results.alpha = alpha;
Results.nPerm = nPerms;
Results.model = 'EEG ~ var1 + var2';

fprintf('Results stored.\n');
%% ==========================================================
% Step 6: Plot significant observed effects
%% ==========================================================

plot_tfce_results(Results.Obs_var1, Results.Mask_var1, ...
                  times, e_loc, ...
                  'TFCE-corrected Main Effect: var1');

plot_tfce_results(Results.Obs_var2, Results.Mask_var2, ...
                  times, e_loc, ...
                  'TFCE-corrected Main Effect: var2');

%% ==========================================================
% Step 7: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/03_TFCE_between_subject_2by2_no_interaction_results.mat', ...
     'Results', ...
     't_Obs_var1', ...
     't_Obs_var2', ...
     'TFCE_Obs_var1', ...
     'TFCE_Obs_var2', ...
     'TFCE_permMax_var1', ...
     'TFCE_permMax_var2', ...
     'Mask_var1', ...
     'Mask_var2', ...
     'P_Values_var1', ...
     'P_Values_var2', ...
     'times', ...
     'e_loc');

disp('TFCE analysis completed and saved.');
