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

clear; clc; close all;
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
        t_Obs_Int(ch, tp) = lme.Coefficients.tStat(4);

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
%   shuffle GroupCode between subjects, keeping both rows together
%
% Condition main effect:
%   flip CondCode within subjects
%
% Interaction:
%   flip CondCode within subjects
%% ==========================================================

TFCE_permMax_Group = nan(nPerm,1);
TFCE_permMax_Cond  = nan(nPerm,1);

fprintf('Step 3: Starting permutation testing: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    perm_t_Group = nan(nChan, nTime);
    perm_t_Cond  = nan(nChan, nTime);

    %% Permute Group between subjects

    subjGroup = zeros(nSubj,1);

    for s = 1:nSubj
        idxSubj = Subject == subjects(s);
        subjGroup(s) = unique(GroupCode(idxSubj));
    end

    subjGroup_perm = subjGroup(randperm(nSubj));

    Group_perm = GroupCode;

    for s = 1:nSubj
        idxSubj = Subject == subjects(s);
        Group_perm(idxSubj) = subjGroup_perm(s);
    end


    %% Flip condition labels within subjects

    Cond_perm = CondCode;

    for s = 1:nSubj

        idxSubj = find(Subject == subjects(s));

        if rand > 0.5
            Cond_perm(idxSubj) = flipud(Cond_perm(idxSubj));
        end

    end

    %% Mass-univariate LME

    for ch = 1:nChan
        for tp = 1:nTime

            EEG = double(squeeze(EEGdata(:, ch, tp)));

            % Group effect permutation
            tbl_G = table(EEG, Group_perm, Cond, SubjectLME, ...
                'VariableNames', {'EEG','Group','Cond','Subject'});

            lme_G = fitlme(tbl_G, ...
                'EEG ~ Group * Cond + (1|Subject)');

            perm_t_Group(ch,tp) = lme_G.Coefficients.tStat(2);

            % Condition effect permutation
            tbl_C = table(EEG, Group, Cond_perm, SubjectLME, ...
                'VariableNames', {'EEG','Group','Cond','Subject'});

            lme_C = fitlme(tbl_C, ...
                'EEG ~ Group * Cond + (1|Subject)');

            perm_t_Cond(ch,tp) = lme_C.Coefficients.tStat(3);

        end
    end

    fprintf('At the %dth permutation\n', p);

    TFCE_perm_Group = ept_mex_TFCE2D(perm_t_Group, ChN, E_H);
    TFCE_perm_Cond  = ept_mex_TFCE2D(perm_t_Cond,  ChN, E_H);

    TFCE_permMax_Group(p) = max(abs(TFCE_perm_Group(:)));
    TFCE_permMax_Cond(p)  = max(abs(TFCE_perm_Cond(:)));

end

fprintf('\nPermutation testing completed.\n');

%% Step 4: TFCE correction

fprintf('Step 4: Computing TFCE-corrected significance...\n');

nPerm = length(TFCE_permMax_Group);
maxTFCE_Group = sort([TFCE_permMax_Group;max(abs(TFCE_Obs_Group(:)))]);
maxTFCEcrit_Group = maxTFCE_Group(round(nPerm*(1-alpha)));

maxTFCE_Cond = sort([TFCE_permMax_Cond;max(abs(TFCE_Obs_Cond(:)))]);
maxTFCEcrit_Cond = maxTFCE_Cond(round(nPerm*(1-alpha)));


Mask_Group = abs(TFCE_Obs_Group) >= maxTFCEcrit_Group;
Mask_Cond  = abs(TFCE_Obs_Cond)  >= maxTFCEcrit_Cond;

P_Values_Group = nan(nChan, nTime);
P_Values_Cond  = nan(nChan, nTime);


for ch = 1:nChan
    for tpoint = 1:nTime
        P_Values_Group(ch, tpoint) = ...
            (sum(TFCE_permMax_Group >= abs(TFCE_Obs_Group(ch, tpoint))) + 1) / ...
            (nPerm + 1);
        
        P_Values_Cond(ch, tpoint) = ...
            (sum(TFCE_permMax_Cond >= abs(TFCE_Obs_Cond(ch, tpoint))) + 1) / ...
            (nPerm + 1);
    end
end

%% Step 5: Store results

Results = struct();

Results.tObs_Group       = t_Obs_Group;
Results.TFCE_Obs_Group  = TFCE_Obs_Group;
Results.TFCE_Null_Group = TFCE_permMax_Group;
Results.maxTFCEcrit_Group  = maxTFCEcrit_Group;
Results.P_Values_Group  = P_Values_Group;
Results.Mask_Group      = Mask_Group;

Results.tObs_Cond       = t_Obs_Cond;
Results.TFCE_Obs_Cond  = TFCE_Obs_Cond;
Results.TFCE_Null_Cond = TFCE_permMax_Cond;
Results.maxTFCEcrit_Cond  = maxTFCEcrit_Cond;
Results.P_Values_Cond  = P_Values_Cond;
Results.Mask_Cond      = Mask_Cond;

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
