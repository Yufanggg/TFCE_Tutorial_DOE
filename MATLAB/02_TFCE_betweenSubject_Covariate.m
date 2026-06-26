%% ==========================================================
% TFCE for between-subject design with covariate
% -1 = Control, 1 = Treatment
% Model:
% EEG ~ Covariate + Group
%
% Test:
% Group effect, adjusted for covariate
%
% Permutation:
% Freedman-Lane permutation
%% ==========================================================

clear; clc; close all;

load('../data/04_simulated_between_subject_covariate_EEG.mat');

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
% Basic settings
%% ==========================================================

[nSubj, nChan, nTime] = size(data);

group = group(:);
covariate = covariate(:);

if length(group) ~= nSubj
    error('group length does not match number of subjects.');
end

if length(covariate) ~= nSubj
    error('covariate length does not match number of subjects.');
end

%% ==========================================================
% Step 1: Observed t-map
%% ==========================================================
X = [covariate, group];
X_red = covariate;
t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        Y = double(data(:, ch, tpoint));
        lm_local = fitlm(X,Y);
        % Column 3 = Group coefficient
        t_Obs(ch,tpoint) = lm_local.Coefficients.tStat(3);

    end
end

%% ==========================================================
% Step 2: Observed TFCE map
%% ==========================================================

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

%% ==========================================================
% Step 3: Freedman-Lane permutation
%% ==========================================================
% Reduced model: intercept + covariate

nPerm = 999;
TFCE_permMax = nan(nPerm,1);

parfor p = 1:nPerm

    perm_t_local = nan(nChan, nTime);

    perm_idx = randperm(nSubj);

    for ch = 1:nChan
        for tpoint = 1:nTime

            Y = double(data(:, ch, tpoint));

             % Reduced model: EEG ~ Covariate
            lm_red = fitlm(X_red, Y);
            Y_hat_red = fitted(lm_red);
            resid_red = residuals(lm_red);

            % Freedman-Lane permuted response
            Y_perm = Y_hat_red + resid_red(perm_idx);

            % Full model on permuted data: EEG_perm ~ Covariate + Group
            lm_perm = fitlm(X_full, Y_perm);
            perm_t_local(ch,tpoint) = lm_perm.Coefficients.tStat(3);

        end
    end

    TFCE_perm = ept_mex_TFCE2D(perm_t_local, ChN, E_H);
    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end

%% ==========================================================
% Step 4: TFCE-corrected significance
%% ==========================================================

alpha = 0.05;

critTFCE = prctile(TFCE_permMax, 100 * (1 - alpha));

Mask = abs(TFCE_Obs) >= critTFCE;

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        P_Values(ch,tpoint) = ...
            (sum(TFCE_permMax >= abs(TFCE_Obs(ch,tpoint))) + 1) / ...
            (nPerm + 1);

    end
end

%% ==========================================================
% Step 5: Store results
%% ==========================================================

Results = struct();

Results.Obs       = t_Obs;
Results.TFCE_Obs  = TFCE_Obs;
Results.TFCE_Null = TFCE_permMax;
Results.critTFCE  = critTFCE;
Results.P_Values  = P_Values;
Results.Mask      = Mask;
Results.alpha     = alpha;
Results.nPerm     = nPerm;
Results.model     = 'EEG ~ Covariate + Group';
Results.test      = 'Group effect adjusted for covariate';

%% ==========================================================
% Step 6: Plot significant effects
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
title('TFCE-corrected Group Effect Adjusted for Covariate');

colorbar;

%% ==========================================================
% Step 7: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/04_TFCE_between_subject_covariate_results.mat', ...
     'Results', ...
     't_Obs', ...
     'TFCE_Obs', ...
     'TFCE_permMax', ...
     'critTFCE', ...
     'P_Values', ...
     'Mask', ...
     'times', ...
     'e_loc');

disp('TFCE covariate-adjusted between-subject analysis completed.');

