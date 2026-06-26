%% ==========================================================
% TFCE analysis
% Between-subject 2 x 2 design with interaction
%
% Model:
%   EEG ~ var1 + var2 + var1:var2
%
% Test:
%   Interaction effect
%
% Factor coding:
%   var1 = -1 / +1
%   var2 = -1 / +1
%
% Permutation:
%   Freedman-Lane
%% ==========================================================

clear; clc; close all;

%% ==========================================================
% Load data
%% ==========================================================

load('data/03_simulated_between_subject_2by2Int_EEG.mat');

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
% Full model:
%   EEG ~ var1 + var2 + interaction
%
% Coefficient rows:
%   1 = Intercept
%   2 = var1
%   3 = var2
%   4 = interaction

interaction = var1 .* var2;

X_full = [var1, var2, interaction];

% Reduced model for testing interaction:
%   EEG ~ var1 + var2
X_red = [var1, var2];

interactionCoefRow = 4;

%% ==========================================================
% Step 1: Observed interaction t-map
%% ==========================================================

t_Obs_Int = zeros(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        Y = double(data(:, ch, tpoint));

        lm_full = fitlm(X_full, Y);

        t_Obs_Int(ch, tpoint) = ...
            lm_full.Coefficients.tStat(interactionCoefRow);

    end
end

%% ==========================================================
% Step 2: Observed TFCE map
%% ==========================================================

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs_Int = ept_mex_TFCE2D(t_Obs_Int, ChN, E_H);

%% ==========================================================
% Step 3: Freedman-Lane permutation test
%% ==========================================================

nPerm = 999;

TFCE_permMax_Int = nan(nPerm, 1);

parfor p = 1:nPerm

    perm_t_local = nan(nChan, nTime);

    perm_idx = randperm(nSubj);

    for ch = 1:nChan
        for tpoint = 1:nTime

            Y = double(data(:, ch, tpoint));

            % Reduced model:
            % EEG ~ var1 + var2
            lm_red = fitlm(X_red, Y);

            Y_hat_red = fitted(lm_red);
            resid_red = residuals(lm_red);

            % Freedman-Lane permuted response
            Y_perm = Y_hat_red + resid_red(perm_idx);

            % Full model on permuted data:
            % EEG_perm ~ var1 + var2 + interaction
            lm_perm = fitlm(X_full, Y_perm);

            perm_t_local(ch, tpoint) = ...
                lm_perm.Coefficients.tStat(interactionCoefRow);

        end
    end

    TFCE_perm = ept_mex_TFCE2D(perm_t_local, ChN, E_H);

    TFCE_permMax_Int(p) = max(abs(TFCE_perm(:)));

end

%% ==========================================================
% Step 4: TFCE-corrected significance
%% ==========================================================

alpha = 0.05;

critTFCE_Int = prctile(TFCE_permMax_Int, 100 * (1 - alpha));

Mask_Int = abs(TFCE_Obs_Int) >= critTFCE_Int;

P_Values_Int = nan(nChan, nTime);

P_Values_Int = ...
    (sum(TFCE_permMax_Int >= abs(TFCE_Obs_Int(:))',1)+1)/(nPerm+1);

P_Values_Int = reshape(P_Values_Int,size(TFCE_Obs_Int));

% for ch = 1:nChan
%     for tpoint = 1:nTime
% 
%         P_Values_Int(ch, tpoint) = ...
%             (sum(TFCE_permMax_Int >= abs(TFCE_Obs_Int(ch, tpoint))) + 1) / ...
%             (nPerm + 1);
% 
%     end
% end

%% ==========================================================
% Step 5: Store results
%% ==========================================================

Results = struct();

Results.Obs_Int       = t_Obs_Int;
Results.TFCE_Obs_Int  = TFCE_Obs_Int;
Results.TFCE_Null_Int = TFCE_permMax_Int;
Results.critTFCE_Int  = critTFCE_Int;
Results.P_Values_Int  = P_Values_Int;
Results.Mask_Int      = Mask_Int;

Results.alpha = alpha;
Results.nPerm = nPerm;
Results.model = 'EEG ~ var1 + var2 + var1:var2';
Results.test  = 'Interaction effect';

%% ==========================================================
% Step 6: Plot significant interaction effects
%% ==========================================================

mT = t_Obs_Int;
mT(~Mask_Int) = 0;

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
title('TFCE-corrected Interaction Effect');

colorbar;

%% ==========================================================
% Step 7: Plot observed TFCE interaction map
%% ==========================================================

figure;

imagesc(times, 1:nChan, TFCE_Obs_Int);
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
title('Observed TFCE Map: Interaction Effect');

colorbar;

%% ==========================================================
% Step 8: Save results
%% ==========================================================

if ~exist('results', 'dir')
    mkdir('results');
end

save('results/03_TFCE_between_subject_2by2_interaction_results.mat', ...
     'Results', ...
     't_Obs_Int', ...
     'TFCE_Obs_Int', ...
     'TFCE_permMax_Int', ...
     'critTFCE_Int', ...
     'P_Values_Int', ...
     'Mask_Int', ...
     'times', ...
     'e_loc');

disp('TFCE interaction analysis completed and saved.');
