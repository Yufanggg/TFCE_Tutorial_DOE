%% ==========================================================
% TFCE analysis
% Between-subject 2 x 2 design without interaction
%
% Input file contains only:
%   EEGdata
%   designTable
%
% Model:
%   EEG ~ FactorA + FactorB
%
% Tests: 
% Main effect of FactorA, adjusted for FactorB and interaction 
% Main effect of FactorB, adjusted for FactorA and interaction 
% FactorA x FactorB interaction 
% 
% Permutation strategies: 
% 
% FactorA: 
% Permute FactorA within each level of FactorB. 
% 
% FactorB & Interaction: 
% Permute FactorB within each level of FactorA. 
% 
% 
% Fit the reduced model EEG ~ FactorA + FactorB,
% permute its residuals across subjects, reconstruct the 
% response, and fit the full factorial model. 
%% ==========================================================
clear; clc; close all;
fprintf('\nStarting TFCE analysis...\n');

%% Load simulated data

load('../Data/03_simulated_between_subject_2by2Int_EEG.mat', ...
     'EEGdata', 'designTable');

var1 = designTable.FactorA;
var2 = designTable.FactorB;

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

[nSubj, nChan, nTime] = size(EEGdata);

var1 = var1(:);
var2 = var2(:);

if height(designTable) ~= nSubj
    error('Number of rows in designTable does not match number of subjects.');
end

if length(var1) ~= nSubj
    error('Length of FactorA does not match number of subjects.');
end

if length(var2) ~= nSubj
    error('Length of FactorB does not match number of subjects.');
end

if length(times) ~= nTime
    error('Length of times does not match number of time points.');
end

if length(e_loc) ~= nChan
    error('Number of channel locations does not match number of channels.');
end

%% Design matrices

X_full = [var1, var2, var1.*var2];

coefRow_var1 = 2;
coefRow_var2 = 3;
coefRow_varInt = 4;

%% Step 1: Observed t-maps

fprintf('Step 1: Computing observed t-statistic maps...\n');

t_Obs_var1 = zeros(nChan, nTime);
t_Obs_var2 = zeros(nChan, nTime);
t_Obs_Int  = zeros(nChan,nTime);

for ch = 1:nChan
    for tp = 1:nTime

        Y = double(EEGdata(:, ch, tp));

        lm_full = fitlm(X_full, Y);

        t_Obs_var1(ch, tp) = lm_full.Coefficients.tStat(coefRow_var1);
        t_Obs_var2(ch, tp) = lm_full.Coefficients.tStat(coefRow_var2);
        t_Obs_Int(ch,tp)  = lm_full.Coefficients.tStat(4);

    end
end

fprintf('Observed t-statistic maps completed.\n');

%% Step 2: Observed TFCE maps

fprintf('Step 2: Computing observed TFCE maps...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs_var1 = ept_mex_TFCE2D(t_Obs_var1, ChN, E_H);
TFCE_Obs_var2 = ept_mex_TFCE2D(t_Obs_var2, ChN, E_H);
TFCE_Obs_Int = ept_mex_TFCE2D(t_Obs_Int,ChN,E_H);

fprintf('Observed TFCE maps completed.\n');

%% Step 3: Restricted permutation testing

nPerms = 999;

TFCE_permMax_var1 = nan(nPerms, 1);
TFCE_permMax_var2 = nan(nPerms, 1);
TFCE_permMax_Int  = nan(nPerms,1);

fprintf('Step 3: Starting restricted permutation testing: %d permutations...\n', nPerms);

var1_levels = unique(var1);
var2_levels = unique(var2);

if numel(var1_levels) ~= 2 || numel(var2_levels) ~= 2
    error('FactorA and FactorB must each have exactly two levels.');
end

parfor p = 1:nPerms

    perm_t_var1 = zeros(nChan, nTime);
    perm_t_var2 = zeros(nChan, nTime);
    perm_t_Int  = zeros(nChan, nTime);

    %% Main effect of FactorA
    % Permute FactorA within each level of FactorB

    var1_perm = var1;

    for lv = 1:numel(var2_levels)

        idx_lv = find(var2 == var2_levels(lv));
        var1_perm(idx_lv) = var1_perm(idx_lv(randperm(numel(idx_lv))));

    end
    
    interaction_perm = var1_perm .* var2;

    X_perm_var1 = [var1_perm, var2, interaction_perm];

    %% Main effect of FactorB
    % Permute FactorB within each level of FactorA

    var2_perm = var2;

    for lv = 1:numel(var1_levels)

        idx_lv = find(var1 == var1_levels(lv));
        var2_perm(idx_lv) = var2_perm(idx_lv(randperm(numel(idx_lv))));

    end
    
    interaction_perm = var1 .* var2_perm;

    X_perm_var2 = [var1, var2_perm, interaction_perm];

    %% Mass-univariate regression

    for ch = 1:nChan
        for tp = 1:nTime

            Y = double(EEGdata(:, ch, tp));

            lm_perm_var1 = fitlm(X_perm_var1, Y);
            perm_t_var1(ch, tp) = ...
                lm_perm_var1.Coefficients.tStat(coefRow_var1);

            lm_perm_var2 = fitlm(X_perm_var2, Y);
            perm_t_var2(ch, tp) = ...
                lm_perm_var2.Coefficients.tStat(coefRow_var2);
            
            perm_t_Int(ch,tp) = ...
                lm_perm_var2.Coefficients.tStat(coefRow_varInt);

        end
    end

    fprintf('At the %dth permutation\n', p);

    TFCE_perm_var1 = ept_mex_TFCE2D(perm_t_var1, ChN, E_H);
    TFCE_perm_var2 = ept_mex_TFCE2D(perm_t_var2, ChN, E_H);
    TFCE_perm_Int = ept_mex_TFCE2D(perm_t_Int,ChN,E_H);

    TFCE_permMax_var1(p) = max(abs(TFCE_perm_var1(:)));
    TFCE_permMax_var2(p) = max(abs(TFCE_perm_var2(:)));
    TFCE_permMax_Int(p)  = max(abs(TFCE_perm_Int(:)));

end

fprintf('\nPermutation testing completed.\n');

%% Step 4: TFCE-corrected significance

fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;
nPerm = length(TFCE_permMax_var1);

maxTFCE_var1 = sort([TFCE_permMax_var1;max(abs(TFCE_Obs_var1(:)))]);
maxTFCEcrit_var1 = TFCE_permMax_var1(round(nPerm*(1-alpha)));

maxTFCE_var2 = sort([TFCE_permMax_var2;max(abs(TFCE_Obs_var2(:)))]);
maxTFCEcrit_var2 = TFCE_permMax_var2(round(nPerm*(1-alpha)));

maxTFCE_Int = sort([TFCE_permMax_Int; max(abs(TFCE_Obs_Int(:)))]);
maxTFCEcrit_Int = maxTFCE_Int(round(nPerm*(1-alpha)));

Mask_var1 = abs(TFCE_Obs_var1) >= maxTFCEcrit_var1;
Mask_var2 = abs(TFCE_Obs_var2) >= maxTFCEcrit_var2;
Mask_Int =  abs(TFCE_Obs_Int) >= maxTFCEcrit_Int;

P_Values_var1 = nan(nChan, nTime);
P_Values_var2 = nan(nChan, nTime);
P_Values_Int  = nan(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        P_Values_var1(ch, tp) = ...
            (sum(TFCE_permMax_var1 >= abs(TFCE_Obs_var1(ch, tp))) + 1) / ...
            (nPerms + 1);

        P_Values_var2(ch, tp) = ...
            (sum(TFCE_permMax_var2 >= abs(TFCE_Obs_var2(ch, tp))) + 1) / ...
            (nPerms + 1);
        
        P_Values_Int(ch,tp) = ...
            (sum(TFCE_permMax_Int >= abs(TFCE_Obs_Int(ch,tp)))+1) ...
            / (nPerms+1);
    end
end

fprintf('TFCE-corrected significance completed.\n');

%% Step 5: Store results

fprintf('Step 5: Storing results...\n');

Results = struct();

Results.Obs_var1       = t_Obs_var1;
Results.TFCE_Obs_var1  = TFCE_Obs_var1;
Results.TFCE_Null_var1 = TFCE_permMax_var1;
Results.maxTFCEcrit_var1  = maxTFCEcrit_var1;
Results.P_Values_var1  = P_Values_var1;
Results.Mask_var1      = Mask_var1;

Results.Obs_var2       = t_Obs_var2;
Results.TFCE_Obs_var2  = TFCE_Obs_var2;
Results.TFCE_Null_var2 = TFCE_permMax_var2;
Results.maxTFCEcrit_var2  = maxTFCEcrit_var2;
Results.P_Values_var2  = P_Values_var2;
Results.Mask_var2      = Mask_var2;

Results.Obs_Int = t_Obs_Int;
Results.TFCE_Obs_Int = TFCE_Obs_Int;
Results.TFCE_Null_Int = TFCE_permMax_Int;
Results.maxTFCEcrit_Int = maxTFCEcrit_Int;
Results.P_Values_Int = P_Values_Int;
Results.Mask_Int = Mask_Int;

Results.alpha = alpha;
Results.nPerm = nPerms;
Results.model = 'EEG ~ FactorA * FactorB';

fprintf('Results stored.\n');

%% Step 6: Save results

if ~exist('../Results', 'dir')
    mkdir('../Results');
end

save('../Results/03_TFCE_between_subject_2by2_results.mat', ...
     'Results', ...
     'nChan', ...
     'times', ...
     'e_loc');

disp('TFCE analysis completed and saved.');

%% Step 7: Plot significant observed effects
clear all; close all; clc
load('../Results/03_TFCE_between_subject_2by2_results.mat')
plot_tfce_results(Results.Obs_var1, Results.Mask_var1, ...
                  times, e_loc, ...
                  'TFCE-corrected Main Effect: FactorA');

plot_tfce_results(Results.Obs_var2, Results.Mask_var2, ...
                  times, e_loc, ...
                  'TFCE-corrected Main Effect: FactorB');
              
plot_tfce_results(Results.Obs_Int, Results.Mask_Int, ...
                  times, e_loc, ...
                  'TFCE-corrected Interaction: Factor A ¡Á Factor B');
