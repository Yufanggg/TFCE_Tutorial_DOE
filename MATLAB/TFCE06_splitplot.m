%% ==========================================================
% TFCE analysis for split-plot EEG design
%
% Group:
%   -1 = HC, 1 = Patient
%
% WordType:
%   -1 = Noun, 1 = Verb
%
% Effects tested:
%   1. Group main effect
%   2. WordType main effect
%   3. Group x WordType interaction
%
% Data:
%   Subjects x Conditions x Channels x Time
%% ==========================================================

clear; clc; close all;
fprintf('\nStarting TFCE analysis...\n');

%% Load data

load('../data/06_simulated_split_plot_EEG.mat');

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

[nSubj, nCond, nChan, nTime] = size(data);

if nCond ~= 2
    error('Expected data format: Subjects x 2 Conditions x Channels x Time');
end

group = group(:);  % -1 = HC, 1 = Patient

if length(group) ~= nSubj
    error('Length of group does not match number of subjects.');
end

if ~all(ismember(unique(group), [-1 1]))
    error('Group must be coded as -1 = HC and 1 = Patient.');
end

%% Long-format design variables
%% Long-format design variables

% Subject ID repeated for Noun and Verb
Subject = kron((1:nSubj)', ones(nCond,1));
Subject = categorical(Subject);

% Group:
% -1 = HC
%  1 = Patient
Group = kron(group, ones(nCond,1));

% WordType:
% -1 = Noun
%  1 = Verb
WordType = repmat([-1; 1], nSubj, 1);

nObs = nSubj * nCond;
%% Long-format EEG data

data_long = zeros(nObs, nChan, nTime);

row = 0;
for s = 1:nSubj
    for c = 1:nCond
        row = row + 1;
        data_long(row,:,:) = squeeze(data(s,c,:,:));
    end
end

%% ==========================================================
% Step 1: Observed LME t-maps
%% ==========================================================
fprintf('Step 1: Computing observed t-statistic map...\n');

t_Group_Obs = zeros(nChan, nTime);
t_Word_Obs  = zeros(nChan, nTime);
t_Int_Obs   = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime
        
        EEG = double(data_long(:, ch, tp));

        tbl = table(EEG, Group, WordType, Subject, ...
            'VariableNames', {'EEG','Group','WordType','Subject'});

        lme = fitlme(tbl, 'EEG ~ Group * WordType + (1|Subject)');

        coefNames = lme.CoefficientNames;

        row_Group = find(strcmp(coefNames, 'Group'));
        row_WordType = find(strcmp(coefNames, 'WordType'));
        row_Interaction = find(strcmp(coefNames, 'Group:WordType'));

        t_Group_Obs(ch,tp) = lme.Coefficients.tStat(row_Group);
        t_Word_Obs(ch,tp)  = lme.Coefficients.tStat(row_WordType);
        t_Int_Obs(ch,tp)   = lme.Coefficients.tStat(row_Interaction);

    end
end

fprintf('Observed t-statistic map completed.\n');

%% ==========================================================
% Step 2: TFCE transformation
%% ==========================================================
fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Group_Obs = ept_mex_TFCE2D(t_Group_Obs, ChN, E_H);
TFCE_Word_Obs  = ept_mex_TFCE2D(t_Word_Obs,  ChN, E_H);
TFCE_Int_Obs   = ept_mex_TFCE2D(t_Int_Obs,   ChN, E_H);

fprintf('Observed TFCE map completed.\n');
%% ==========================================================
% Step 3: Permutation tests
%% ==========================================================

nPerm = 999;

TFCE_permMax_Group = nan(nPerm,1);
TFCE_permMax_Word  = nan(nPerm,1);
TFCE_permMax_Int   = nan(nPerm,1);

fprintf('Step 3: Starting permutation testing: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    perm_t_Group = zeros(nChan, nTime);
    perm_t_Word  = zeros(nChan, nTime);
    perm_t_Int   = zeros(nChan, nTime);
    
    
    %% ------------------------------------------------------
    % Group main effect:
    % permute group labels across subjects
    %% ------------------------------------------------------
    group_perm_subject = group(randperm(nSubj));
    Group_perm = kron(group_perm_subject, ones(nCond,1));
    
        %% ------------------------------------------------------
    % WordType main effect:
    % swap Noun / Verb labels within subjects
    %% ------------------------------------------------------

    WordType_perm = WordType;

    row = 0;
    for s = 1:nSubj

        rows_s = row + (1:nCond);
        row = row + nCond;

        if rand > 0.5
            WordType_perm(rows_s) = flip(WordType_perm(rows_s));
        end

    end

    %% ------------------------------------------------------
    % Interaction:
    % permute group labels across subject-level
    % Verb-minus-Noun difference waves
    %% ------------------------------------------------------

    group_perm_interaction = group(randperm(nSubj));

    for ch = 1:nChan
        for tp = 1:nTime
            
            
            EEG = double(data_long(:, ch, tp));

            %% Group permutation model

            tbl_Group = table(EEG, Subject, Group_perm, WordType, ...
                'VariableNames', {'EEG','Subject','Group','WordType'});
            
            lme_Group = fitlme(tbl_Group, ...
                'EEG ~ Group * WordType + (1|Subject)');

            coefNames = lme_Group.CoefficientNames;
            row_Group = find(strcmp(coefNames, 'Group'));

            perm_t_Group(ch,tp) = ...
                lme_Group.Coefficients.tStat(row_Group);
            
            %% WordType permutation model

            tbl_WordType = table(EEG, Subject, Group, WordType_perm, ...
                'VariableNames', {'EEG','Subject','Group','WordType'});

            lme_WordType = fitlme(tbl_WordType, ...
                'EEG ~ Group * WordType + (1|Subject)');

            coefNames = lme_WordType.CoefficientNames;
            row_WordType = find(strcmp(coefNames, 'WordType'));

            perm_t_Word(ch,tp) = ...
                lme_WordType.Coefficients.tStat(row_WordType);
            
            
            %% Interaction permutation using difference scores
            row_Int = find(strcmp(coefNames, 'Group:WordType'));

            perm_t_Int(ch,tp) = ...
                lme_Group.Coefficients.tStat(row_Int);

        end
    end
    
    fprintf('At the %dth permutation\n', p);
    
    %% TFCE transform
    
    TFCE_perm_Group = ept_mex_TFCE2D(perm_t_Group, ChN, E_H);
    TFCE_perm_Word  = ept_mex_TFCE2D(perm_t_Word,  ChN, E_H);
    TFCE_perm_Int   = ept_mex_TFCE2D(perm_t_Int,   ChN, E_H);
    
    %% Max absolute TFCE statistic

    TFCE_permMax_Group(p) = max(abs(TFCE_perm_Group(:)));
    TFCE_permMax_Word(p)  = max(abs(TFCE_perm_Word(:)));
    TFCE_permMax_Int(p)   = max(abs(TFCE_perm_Int(:)));

end

fprintf('\nPermutation testing completed.\n');
%% ==========================================================
% Step 4: TFCE correction
%% ==========================================================
fprintf('Step 4: Computing TFCE-corrected significance...\n');
alpha = 0.05;

crit_Group = prctile(TFCE_permMax_Group, 100*(1-alpha));
crit_Word  = prctile(TFCE_permMax_Word,  100*(1-alpha));
crit_Int   = prctile(TFCE_permMax_Int,   100*(1-alpha));

Mask_Group = abs(TFCE_Group_Obs) >= crit_Group;
Mask_Word  = abs(TFCE_Word_Obs)  >= crit_Word;
Mask_Int   = abs(TFCE_Int_Obs)   >= crit_Int;

P_Group = nan(nChan,nTime);
P_Word  = nan(nChan,nTime);
P_Int   = nan(nChan,nTime);

for i = 1:numel(TFCE_Group_Obs)

    P_Group(i) = ...
        (sum(TFCE_permMax_Group >= abs(TFCE_Group_Obs(i))) + 1) / ...
        (nPerm + 1);

    P_Word(i) = ...
        (sum(TFCE_permMax_Word >= abs(TFCE_Word_Obs(i))) + 1) / ...
        (nPerm + 1);

    P_Int(i) = ...
        (sum(TFCE_permMax_Int >= abs(TFCE_Int_Obs(i))) + 1) / ...
        (nPerm + 1);

end

fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', critTFCE);
%% ==========================================================
% Step 5: Store results
%% ==========================================================
fprintf('Step 5: Storing results...\n');
Results = struct();

Results.t_Group_Obs = t_Group_Obs;
Results.t_Word_Obs  = t_Word_Obs;
Results.t_Int_Obs   = t_Int_Obs;

Results.TFCE_Group_Obs = TFCE_Group_Obs;
Results.TFCE_Word_Obs  = TFCE_Word_Obs;
Results.TFCE_Int_Obs   = TFCE_Int_Obs;

Results.TFCE_permMax_Group = TFCE_permMax_Group;
Results.TFCE_permMax_Word  = TFCE_permMax_Word;
Results.TFCE_permMax_Int   = TFCE_permMax_Int;

Results.crit_Group = crit_Group;
Results.crit_Word  = crit_Word;
Results.crit_Int   = crit_Int;

Results.Mask_Group = Mask_Group;
Results.Mask_Word  = Mask_Word;
Results.Mask_Int   = Mask_Int;

Results.P_Group = P_Group;
Results.P_Word  = P_Word;
Results.P_Int   = P_Int;

Results.alpha = alpha;
Results.nPerm = nPerm;
Results.model = 'EEG ~ Group * WordType + (1|Subject)';

fprintf('Results stored.\n');
%% ==========================================================
% Step 6: Plot significant effects
%% ==========================================================

plot_tfce_result(t_Group_Obs, Mask_Group, times, e_loc, ...
    'TFCE-corrected Group Main Effect');

plot_tfce_result(t_Word_Obs, Mask_Word, times, e_loc, ...
    'TFCE-corrected WordType Main Effect');

%plot_tfce_result(t_Int_Obs, Mask_Int, times, e_loc, ...
plot_tfce_result('TFCE-corrected Group x WordType Interaction');

%% ==========================================================
% Step 7: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/06_TFCE_split_plot_LME_results.mat', ...
     'Results', ...
     't_Group_Obs', ...
     't_Word_Obs', ...
     't_Int_Obs', ...
     'TFCE_Group_Obs', ...
     'TFCE_Word_Obs', ...
     'TFCE_Int_Obs', ...
     'TFCE_permMax_Group', ...
     'TFCE_permMax_Word', ...
     'TFCE_permMax_Int', ...
     'Mask_Group', ...
     'Mask_Word', ...
     'Mask_Int', ...
     'P_Group', ...
     'P_Word', ...
     'P_Int', ...
     'times', ...
     'e_loc');

disp('Split-plot LME TFCE analysis completed and saved.');

