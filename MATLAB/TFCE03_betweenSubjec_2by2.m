%% ==========================================================
% TFCE analysis
% Balanced between-subject 2 x 2 factorial design
%
% Model:
%   EEG ~ FactorA * FactorB
%
% Factors coded as -1 / +1.
%
% Synchronized label-permutation scheme:
%
% FactorA:
%   Permute FactorA within each FactorB level using the
%   same permutation order across both strata.
%
% FactorB and interaction:
%   Permute FactorB within each FactorA level using the
%   same permutation order across both strata.
%   Reconstruct the interaction after permuting FactorB.
%
% IMPORTANT:
%   This implements the requested synchronized label scheme.
%   Synchronization alone does not establish validity for
%   testing an individual effect with arbitrary other effects
%   present. In particular, the interaction permutation also
%   changes FactorB and therefore does not isolate the
%   interaction null while preserving the FactorB effect.
%
% TFCE correction:
%   Maximum absolute TFCE across channels and time.
%   Each effect is corrected separately.
%
% Requirements:
%   readlocs, ept_ChN2, ept_mex_TFCE2D,
%   plot_tfce_results, and Parallel Computing Toolbox.
%% ==========================================================

clear; clc; close all;

fprintf('\nStarting TFCE analysis...\n');
%% ====================================
%% Load simulated data
%% ====================================
load('../Data/03_simulated_between_subject_2by2Int_EEG.mat', ...
     'EEGdata', 'designTable');

[nSubj, nChan, nTime] = size(EEGdata);

times = -200:4:800;

assert(height(designTable) == nSubj, ...
    'Number of rows in designTable must match nSubj.');

assert(numel(times) == nTime, ...
    'Length of times must match nTime.');

assert(all(isfinite(EEGdata(:))), ...
    'EEGdata must contain only finite values.');

%% ====================================
%% Recode factors as -1 / +1
%% ====================================
rawA = categorical(designTable.FactorA(:));
rawB = categorical(designTable.FactorB(:));

levelsA = categories(rawA);
levelsB = categories(rawB);

assert(numel(levelsA) == 2 && numel(levelsB) == 2, ...
    'FactorA and FactorB must each have exactly two levels.');

FactorA = 2 * double(rawA == levelsA{2}) - 1;
FactorB = 2 * double(rawB == levelsB{2}) - 1;

fprintf('FactorA: %s = -1; %s = +1\n', ...
    levelsA{1}, levelsA{2});

fprintf('FactorB: %s = -1; %s = +1\n', ...
    levelsB{1}, levelsB{2});
%% ====================================
%% Check balance and prepare aligned strata
%% ====================================
cellCounts = [
    sum(FactorA == -1 & FactorB == -1), ...
    sum(FactorA == -1 & FactorB == +1);
    sum(FactorA == +1 & FactorB == -1), ...
    sum(FactorA == +1 & FactorB == +1)
];

assert(all(cellCounts(:) > 0), ...
    'All four factorial cells must contain observations.');

assert(all(cellCounts(:) == cellCounts(1)), ...
    'This implementation requires equal sample sizes in all four cells.');

%% ====================================
%% Other settings
%% ====================================
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
    error('Missing channels: %s', ...
        strjoin(chanLabels_32(~tf), ', '));
end

% EEGdata channel order must match chanLabels_32.
e_loc = chanlocs_1020(idx);

%% ====================================
%% Step 1: get tObs
%% ====================================
%% Design matrices

X_full = [FactorA, FactorB, FactorA .* FactorB];

coefRow_A = 2;
coefRow_B = 3;
coefRow_AB = 4;

fprintf('Step 1: Computing observed t-statistic maps...\n');

t_Obs_A = zeros(nChan, nTime);
t_Obs_B = zeros(nChan, nTime);
t_Obs_AB  = zeros(nChan,nTime);

for ch = 1:nChan
    for tp = 1:nTime

        Y = double(EEGdata(:, ch, tp));

        lm_full = fitlm(X_full, Y);

        t_Obs_A(ch, tp) = lm_full.Coefficients.tStat(coefRow_var1);
        t_Obs_B(ch, tp) = lm_full.Coefficients.tStat(coefRow_var2);
        t_Obs_AB(ch,tp)  = lm_full.Coefficients.tStat(4);

    end
end

fprintf('Observed t-statistic maps completed.\n');

%% ====================================
%% Step 2: Observed TFCE maps
%% ====================================

fprintf('Step 2: Computing observed TFCE maps...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs_A = ept_mex_TFCE2D(t_Obs_A, ChN, E_H);
TFCE_Obs_B = ept_mex_TFCE2D(t_Obs_B, ChN, E_H);
TFCE_Obs_AB = ept_mex_TFCE2D(t_Obs_AB,ChN,E_H);

fprintf('Observed TFCE maps completed.\n');

%% ====================================
%% Step 3: SPT permutations
%% ====================================
nPerms = 999;

TFCE_permMaxPos_A = nan(nPerms, 1);
TFCE_permMaxPos_B = nan(nPerms, 1);
TFCE_permMaxPos_AB= nan(nPerms,1);

TFCE_permMinNeg_A = nan(nPerms, 1);
TFCE_permMinNeg_B = nan(nPerms, 1);
TFCE_permMinNeg_AB= nan(nPerms,1);

fprintf('Step 3: Starting SPT: %d permutations...\n', nPerms);


%% FactorB strata for synchronized FactorA permutations.
idx_B1 = find(FactorB == -1);
idx_B2 = find(FactorB == +1);

assert(numel(idx_B1) == numel(idx_B2), ...
    'Synchronized FactorA permutations require equally sized strata.');

assert(isequal(FactorA(idx_B1), FactorA(idx_B2)), ...
    'FactorA-label sequences must align across FactorB strata.');

% FactorA strata for synchronized FactorB permutations.
idx_A1 = find(FactorA == -1);
idx_A2 = find(FactorA == +1);

assert(numel(idx_A1) == numel(idx_A2), ...
    'Synchronized FactorB permutations require equally sized strata.');

assert(isequal(FactorB(idx_A1), FactorB(idx_A2)), ...
    'FactorB-label sequences must align across FactorA strata.');


fprintf('Starting %d synchronized permutations...\n', nPerms);

parfor p = 1:nPerms

    %% FactorA: synchronized permutation across FactorB strata

    FactorA_perm = FactorA;

    % One shared order for both FactorB strata.
    permOrderA =  randperm(numel(idx_B1));
    
    FactorA_perm(idx_B1) = FactorA(idx_B1(permOrderA));
    FactorA_perm(idx_B2) = FactorA(idx_B2(permOrderA));

    % Reconstruct the interaction.
    Interaction_perm_A = FactorA_perm .* FactorB;
    
    X_perm_A = [
        FactorA_perm, ...
        FactorB, ...
        Interaction_perm_A
    ];

    % Synchronized permutation across FactorA strata

    FactorB_perm = FactorB;

    % One shared order for both FactorA strata.
    permOrderB = randperm(numel(idx_A1));

    FactorB_perm(idx_A1) = FactorB(idx_A1(permOrderB));
    FactorB_perm(idx_A2) = FactorB(idx_A2(permOrderB));

    % Reconstruct the interaction.
    Interaction_perm_B = FactorA .* FactorB_perm;

    X_perm_B = [
        FactorA, ...
        FactorB_perm, ...
        Interaction_perm_B
    ];

    %% Mass-univariate regression

    for ch = 1:nChan
        for tp = 1:nTime

            Y = double(EEGdata(:, ch, tp));

            lm_perm_A = fitlm(X_perm_A, Y);
            tPermA(ch, tp) = lm_perm_A.Coefficients.tStat(coefRow_var1);
            
            tPermAB(ch,tp) = lm_perm_A.Coefficients.tStat(coefRow_varInt);
            
            lm_perm_B = fitlm(X_perm_B, Y);
            tPermB(ch, tp) = lm_perm_B.Coefficients.tStat(coefRow_var1);        

        end
    end

    TFCEperm_A = ept_mex_TFCE2D(tPermA, ChN, E_H);
    TFCEperm_B = ept_mex_TFCE2D(tPermB, ChN, E_H);
    TFCEperm_AB  = ept_mex_TFCE2D(tPermAB,  ChN, E_H);

    %% Maximum absolute TFCE across channels and time
    TFCE_permMaxPos_A(p) = max(TFCEperm_A(:));
    TFCE_permMaxPos_B(p) = max(TFCEperm_B(:));
    TFCE_permMaxPos_AB(p)= max(TFCEperm_AB(:));
    
    TFCE_permMinNeg_A(p) = min(TFCEperm_A(:));
    TFCE_permMinNeg_B(p) = min(TFCEperm_B(:));
    TFCE_permMinNeg_AB(p)= min(TFCEperm_AB(:));
end

fprintf('Permutation testing completed.\n');

%% ====================================
%% Get TFCE-corrected p-values
%% ====================================
fprintf('Computing TFCE-corrected p-values...\n');

alpha = 0.05;

maxTFCEPos_A = sort([TFCE_permMaxPos_A; max(TFCE_Obs_A(:))]);
minTFCENeg_A = sort([TFCE_permMinNeg_A; min(TFCE_Obs_A(:))]);

maxTFCEPos_B = sort([TFCE_permMaxPos_B; max(TFCE_Obs_B(:))]);
minTFCENeg_B = sort([TFCE_permMinNeg_B; min(TFCE_Obs_B(:))]);

maxTFCEPos_AB = sort([TFCE_permMaxPos_AB; max(TFCE_Obs_AB(:))]);
minTFCENeg_AB = sort([TFCE_permMinNeg_AB; min(TFCE_Obs_AB(:))]);

P_Values_A = nan(nChan,nTime);
P_Values_B = nan(nChan,nTime);
P_Values_AB = nan(nChan,nTime);

for ch = 1:nChan
    for tp = 1:nTime
        pr_A = (1 + sum(maxTFCEPos_A >= TFCE_Obs_A(ch,tp))) / (nPerm + 1);
        pl_A = (1 + sum(minTFCENeg_A <= TFCE_Obs_A(ch,tp))) / (nPerm + 1);        
        P_Values_A(ch,tp) = min(1, 2 * min(pr_A, pl_A));
        
        pr_B = (1 + sum(maxTFCEPos_B >= TFCE_Obs_B(ch,tp))) / (nPerm + 1);
        pl_B = (1 + sum(minTFCENeg_B <= TFCE_Obs_B(ch,tp))) / (nPerm + 1);        
        P_Values_B(ch,tp) = min(1, 2 * min(pr_B, pl_B));
        
        pr_AB = (1 + sum(maxTFCEPos_AB >= TFCE_Obs_AB(ch,tp))) / (nPerm + 1);
        pl_AB = (1 + sum(minTFCENeg_AB <= TFCE_Obs_AB(ch,tp))) / (nPerm + 1);        
        P_Values_AB(ch,tp) = min(1, 2 * min(pr_AB, pl_AB));
    end
end

fprintf('TFCE-corrected significance completed.\n');

%% ====================================
%% Store results
%% ====================================
Results = struct();

Results.t_Obs_A       = t_Obs_A;
Results.TFCE_Obs_A    = TFCE_Obs_A;
Results.maxTFCEPos_A  = maxTFCEPos_A;
Results.minTFCENeg_A  = minTFCENeg_A;
Results.P_Values_A    = P_Values_A;

Results.t_Obs_B       = t_Obs_B;
Results.TFCE_Obs_B    = TFCE_Obs_B;
Results.maxTFCEPos_B  = maxTFCEPos_B;
Results.minTFCENeg_B  = minTFCENeg_B;
Results.P_Values_B    = P_Values_B;

Results.t_Obs_AB       = t_Obs_AB;
Results.TFCE_Obs_AB    = TFCE_Obs_AB;
Results.maxTFCEPos_AB  = maxTFCEPos_AB;
Results.minTFCENeg_AB  = minTFCENeg_AB;
Results.P_Values_AB    = P_Values_AB;


Results.alpha        = alpha;
Results.nPerm        = nPerms;
Results.E_H          = E_H;
Results.model        = 'EEG ~ FactorA * FactorB';
Results.factorCoding = '-1 / +1';
Results.levelsA      = levelsA;
Results.levelsB      = levelsB;
Results.cellCounts   = cellCounts;
Results.randomSeed   = randomSeed;

Results.permutation_var1 = ...
    'Shared A-label permutation within aligned B strata';

Results.permutation_var2 = ...
    'Shared B-label permutation within aligned A strata';

Results.permutation_Int = ...
    ['Shared B-label permutation within aligned A strata; ' ...
     'interaction reconstructed as FactorA .* FactorB_perm'];

Results.correction = ...
    'Maximum absolute TFCE across channels/time, separately per effect';

fprintf('Results stored.\n');

%% 12. Save results

if ~exist('../Results', 'dir')
    mkdir('../Results');
end

save('../Results/03_TFCE_between_subject_2by2_results.mat', ...
    'Results', 'nChan', 'times', 'e_loc');

fprintf('TFCE analysis completed and saved.\n');


%% ==========================================================
% Plot the observed TFCE map
%% ==========================================================

clear all; clc; close all
load('../Results/03_TFCE_between_subject_2by2_results.mat')

figure;
sigT = Results.t_Obs_A;

sigT(Results.P_Values_A > 0.05) = 0;

imagesc(time, 1:nChan, sigT);

axis xy;

xlim([-200 700]);

set(gca,...
    'YTick',1:nChan,...
    'YTickLabel',{channelinfo.labels},...
    'XTick',-200:150:700,...
    'TickLength',[0 0],...
    'FontSize',15,...
    'FontName','Arial');

xlabel('Time (ms)');

ylabel('Channel');

title('Observed t Map of A after correction');

cb=colorbar;

ylabel(cb,'t value',...
    'FontSize',15,...
    'FontName','Arial');

figure;
sigTB = Results.t_Obs_B;

sigTB(Results.P_Values_B > 0.05) = 0;

imagesc(time, 1:nChan, sigTB);

axis xy;

xlim([-200 700]);

set(gca,...
    'YTick',1:nChan,...
    'YTickLabel',{channelinfo.labels},...
    'XTick',-200:150:700,...
    'TickLength',[0 0],...
    'FontSize',15,...
    'FontName','Arial');

xlabel('Time (ms)');

ylabel('Channel');

title('Observed t Map of B after correction');

cb=colorbar;

ylabel(cb,'t value',...
    'FontSize',15,...
    'FontName','Arial');


figure;

sigTB = Results.t_Obs_AB;

sigTB(Results.P_Values_AB > 0.05) = 0;

imagesc(time, 1:nChan, sigTB);

axis xy;

xlim([-200 700]);

set(gca,...
    'YTick',1:nChan,...
    'YTickLabel',{channelinfo.labels},...
    'XTick',-200:150:700,...
    'TickLength',[0 0],...
    'FontSize',15,...
    'FontName','Arial');

xlabel('Time (ms)');

ylabel('Channel');

title('Observed t Map of AB after correction');

cb=colorbar;

ylabel(cb,'t value',...
    'FontSize',15,...
    'FontName','Arial');