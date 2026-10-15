%% ==========================================================
% TFCE analysis
% Split-plot mixed design
%
% Input file contains only:
%   EEGdata
%   designTable
%
% EEGdata:
%   Subject-condition rows x Channels x Time
%
% designTable:
%   Subject
%   GroupCode   % -1 = HC, 1 = Patient
%   GroupName
%   CondCode    % -1 = Noun, 1 = Verb
%   CondName
%
% Model:
%   EEG ~ GroupCode * CondCode + (1|Subject)
%
% Tests:
%   Group main effect
%   Condition main effect
%% ==========================================================

clear; clc; close all; rng(123);
fprintf('\nStarting split-plot TFCE analysis...\n');

%% Load data

load('../Data/06_simulated_split_plot_EEG.mat', ...
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

Subject   = designTable.Subject;
GroupCode = designTable.GroupCode;
CondCode  = designTable.CondCode;

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

if nRows ~= nSubj * 2
    error('Expected exactly two condition rows per subject.');
end

for s = 1:nSubj

    subj = subjects(s);
    idxSubj = Subject == subj;

    if sum(idxSubj) ~= 2
        error('Each subject must have exactly two rows.');
    end

    if ~all(sort(CondCode(idxSubj)) == [-1; 1])
        error('Each subject must have one Noun (-1) and one Verb (1).');
    end

end
assert(size(Y1,2) == 2 && size(Y2,2) == 2, ...
    'Each group must have two within-subject conditions.');
assert(all(isfinite(Y1(:))) && all(isfinite(Y2(:))), ...
    'Inputs must be finite with no missing cells.');

%% Variables for LME

SubjectLME = categorical(Subject);

Group = GroupCode;
Cond  = CondCode;

%% TFCE settings

nPerm = 999;
alpha = 0.05;

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

%% ==========================================================
% Step 1: Observed LME t-maps
%% ==========================================================

fprintf('Step 1: Computing observed t-statistic maps...\n');

t_Obs_Group = zeros(nChan, nTime);
t_Obs_Cond  = zeros(nChan, nTime);
t_Obs_Int   = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        EEG = double(squeeze(EEGdata(:, ch, tp)));

        tbl = table(EEG, Group, Cond, SubjectLME, ...
            'VariableNames', {'EEG','Group','Cond','Subject'});

        lme = fitlme(tbl, 'EEG ~ Group * Cond + (1|Subject)');

        % Coefficient rows:
        % 1 = Intercept
        % 2 = Group
        % 3 = Cond
        % 4 = Interaction
        t_Obs_Group(ch,tp) = lme.Coefficients.tStat(2);
        t_Obs_Cond(ch,tp)  = lme.Coefficients.tStat(3);
        t_Obs_Int(ch, tp)  = lme.Coefficients.tStat(4);

    end
end

fprintf('Observed t-statistic maps completed.\n');

%% Step 2: Observed TFCE maps

fprintf('Step 2: Computing observed TFCE maps...\n');

TFCE_Obs_Group = ept_mex_TFCE2D(t_Obs_Group, ChN, E_H);
TFCE_Obs_Cond  = ept_mex_TFCE2D(t_Obs_Cond,  ChN, E_H);
TFCE_Obs_Int   = ept_mex_TFCE2D(t_Obs_Int,  ChN, E_H);

fprintf('Observed TFCE maps completed.\n');

%% ==========================================================
% Step 3: Permutation testing
%
% Group main effect:
%   symmetry permutation test
%
% Condition main effect:
%   synchronized permutation test
%
% Interaction:
%   symmetry permutation
%% ==========================================================
TFCE_permMaxPos_Group = nan(nPerm,1);
TFCE_permMaxPos_Cond  = nan(nPerm,1);
TFCE_permMaxPos_Int   = nan(nPerm,1);

TFCE_permMinNeg_Group = nan(nPerm,1);
TFCE_permMinNeg_Cond  = nan(nPerm,1);
TFCE_permMinNeg_Int   = nan(nPerm,1);

fprintf('Step 3: Starting permutation testing: %d permutations...\n', nPerm);

% ---------------------------------------------------------
% Step 3.1(SPY for testing Group effects): 
% Centre all observations using the pooled median
% ---------------------------------------------------------
Y1_idx = designTable.GroupCode == -1;
Y2_idx = designTable.GroupCode == 1;
idx_Group1 = find(designTable.GroupCode == designTable.GroupCode(1));

Cond = designTable.CondCode;

% [~, ~, subjectIndex1] = designTable.Subject(Y1_idx);
% nSubjects1 = length(unique(subjectIndex1));
% 
% [~, ~, subjectIndex2] =  designTable.Subject(Y2_idx);
% nSubjects2 = length(unique(subjectIndex2));

parfor p = 1:nPerm

    perm_t_Group = nan(nChan, nTime);
    perm_t_Cond  = nan(nChan, nTime);
    perm_t_Int   = nan(nChan, nTime);
    
    Signs = 2 * randi([0, 1], nRows, 1) - 1;
    permOrder = randperm(numel(idx_Group1));

    %% Mass-univariate LME

    for ch = 1:nChan
        for tp = 1:nTime
            
            % Group effect permutation
            EEG = double(squeeze(EEGdata(:, ch, tp)));
            pooledMedian = median(EEG(:), 'omitnan');
            Y = EEG - pooledMedian; Y_perm = Y .* Signs;
            Y1_perm = Y_perm(Y1_idx); Y2_perm = Y_perm(Y2_idx);
            
            tbl = table(Y1_perm, Group, Cond_perm, SubjectLME, ...
                'VariableNames', {'EEG','Group','Cond','Subject'});
            lme = fitlme(tbl, 'EEG ~ Group * Cond + (1|Subject)');

            perm_t_Group(ch,tp) = lme.Coefficients.tStat(2);
            
            perm_t_Int(ch,tp)= lme_C.Coefficients.tStat(4);
            
            % Interaction effect between Group and within-subject variables
            
            perm_t_Int(ch,tp)  = lme_C.Coefficients.tStat(4);
            

            % within-subject effect
            Cond_perm(Y1_idx) = Cond(Y1_idx(permOrder));
            Cond_perm(Y1_idx) = Cond(Y1_idx(permOrder));
            tbl_C = table(EEG, Group, Cond_perm, SubjectLME, ...
                'VariableNames', {'EEG','Group','Cond','Subject'});

            lme_C = fitlme(tbl_C, 'EEG ~ Group * Cond + (1|Subject)');

            perm_t_Cond(ch,tp) = lme_C.Coefficients.tStat(3);
            
        end
    end
    
    TFCE_perm_Group = ept_mex_TFCE2D(perm_t_Group, ChN, E_H);
    TFCE_perm_Cond  = ept_mex_TFCE2D(perm_t_Cond,  ChN, E_H);
    TFCE_perm_Int   = ept_mex_TFCE2D(perm_t_Int,  ChN, E_H);
    
    TFCE_permMaxPos_Group(p) = max(TFCE_perm_Group(:));
    TFCE_permMaxPos_Cond(p)  = max(TFCE_perm_Cond(:));
    TFCE_permMaxPos_Int(p)   = max(TFCE_perm_Int(:));
    
    TFCE_permMinNeg_Group(p) = min(TFCE_perm_Group(:));
    TFCE_permMinNeg_Cond(p)  = min(TFCE_perm_Cond(:));
    TFCE_permMinNeg_Int(p)   = min(TFCE_perm_Int(:));
    
    progressbar(p/nPerm)

end

fprintf('\nPermutation testing completed.\n');

%% Step 4: TFCE correction

fprintf('Step 4: Computing TFCE-corrected significance...\n');

maxTFCEPos_Group = sort([TFCE_permMaxPos_Group;max(TFCE_Obs_Group(:))]);
maxTFCEPos_Cond = sort([TFCE_permMaxPos_Cond;max(TFCE_Obs_Cond(:))]);
maxTFCEPos_Int = sort([TFCE_permMaxPos_Int;max(TFCE_Obs_Int(:))]);

minTFCENeg_Group = sort([TFCE_permMinNeg_Group;min(TFCE_Obs_Group(:))]);
minTFCENeg_Cond = sort([TFCE_permMinNeg_Group;min(TFCE_Obs_Group(:))]);
minTFCENeg_Int = sort([TFCE_permMinNeg_Group;min(TFCE_Obs_Group(:))]);


P_Values_Group = nan(nChan, nTime);
P_Values_Cond  = nan(nChan, nTime);
P_Values_Int  = nan(nChan, nTime);


for ch = 1:nChan
    for tpoint = 1:nTime
        
        pr_Group = (1 + sum(maxTFCEPos_Group >= TFCE_Obs_Group(ch,tp))) / (nPerm + 1);
        pl_Group = (1 + sum(minTFCENeg_Group <= TFCE_Obs_Group(ch,tp))) / (nPerm + 1);        
        P_Values_Group(ch,tp) = min(1, 2 * min(pr_Group, pl_Group));
        
        pr_Cond = (1 + sum(maxTFCEPos_Cond >= TFCE_Obs_Cond(ch,tp))) / (nPerm + 1);
        pl_Cond = (1 + sum(minTFCENeg_Cond <= TFCE_Obs_Cond(ch,tp))) / (nPerm + 1);        
        P_Values_Cond(ch,tp) = min(1, 2 * min(pr_Cond, pl_Cond));
        
        pr_Int = (1 + sum(maxTFCEPos_Int >= TFCE_Obs_Int(ch,tp))) / (nPerm + 1);
        pl_Int = (1 + sum(minTFCENeg_Int <= TFCE_Obs_Int(ch,tp))) / (nPerm + 1);        
        P_Values_Int(ch,tp) = min(1, 2 * min(pr_Int, pl_Int));
        
    end
end

%% Step 5: Store results

Results = struct();

Results.tObs_Group       = t_Obs_Group;
Results.TFCE_Obs_Group   = TFCE_Obs_Group;
Results.maxTFCEPos_Group = maxTFCEPos_Group;
Results.minTFCENeg_Group = minTFCENeg_Group;
Results.P_Values_Group  = P_Values_Group;

Results.tObs_Cond       = t_Obs_Cond;
Results.TFCE_Obs_Cond   = TFCE_Obs_Cond;
Results.maxTFCEPos_Cond = maxTFCEPos_Cond;
Results.minTFCENeg_Cond = minTFCENeg_Cond;
Results.P_Values_Cond   = P_Values_Cond;

Results.tObs_Int       = t_Obs_Int;
Results.TFCE_Obs_Int   = TFCE_Obs_Int;
Results.maxTFCEPos_Int = maxTFCEPos_Int;
Results.minTFCENeg_Int = minTFCENeg_Int;
Results.P_Values_Int  = P_Values_Int;

Results.alpha = alpha;
Results.nPerm = nPerm;
Results.model = 'EEG ~ Group + Cond + Interaction + (1|Subject)';

%% Step 6: Save results

if ~exist('../Results', 'dir')
    mkdir('../Results');
end
 
save('../Results/06_TFCE_split_plot_results.mat', ...
     'Results', ...
     'nChan', ...
     'times', ...
     'e_loc');

disp('Split-plot TFCE analysis completed and saved.');

%% Step 7: Plot results
clear all; clc; close all

load('../Results/06_TFCE_split_plot_results.mat')

plot_tfce_results(Results.tObs_Group, Results.Mask_Group, times, e_loc, ...
    'TFCE-corrected Group Effect');

plot_tfce_results(Results.tObs_Cond, Results.Mask_Cond, times, e_loc, ...
    'TFCE-corrected Condition Effect');

plot_tfce_results(Results.tObs_Int, Results.Mask_Int, times, e_loc, ...
    'TFCE-corrected Condition Effect');
