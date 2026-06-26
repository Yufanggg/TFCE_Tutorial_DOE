%% ==========================================================
% TFCE analysis for split-plot EEG design
%
% Between-subject factor:
%   Group: HC vs Patient
%
% Within-subject factor:
%   WordType: Noun vs Verb
%
% Model:
%   EEG ~ Group * WordType + (1|Subject)
%
% Data format:
%   data = Subjects x Conditions x Channels x Time
%
% Condition 1 = Noun
% Condition 2 = Verb
%% ==========================================================

clear; clc; close all;

%% Load data

load('../data/10_simulated_split_plot_EEG.mat');

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

group = group(:);  % 0 = HC, 1 = Patient

if length(group) ~= nSubj
    error('Length of group does not match number of subjects.');
end

%% Long-format design variables

Subject = kron((1:nSubj)', ones(nCond,1));
Subject = categorical(Subject);

% Group: 0 = HC, 1 = Patient
Group = kron(group, ones(nCond,1));

% WordType: 0 = Noun, 1 = Verb
WordType = repmat([0; 1], nSubj, 1);

%% ==========================================================
% Step 1: Observed LME t-maps
%% ==========================================================

t_Group_Obs = zeros(nChan, nTime);
t_Word_Obs  = zeros(nChan, nTime);
t_Int_Obs   = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        nounVals = squeeze(data(:,1,ch,tp));
        verbVals = squeeze(data(:,2,ch,tp));

        EEG = reshape([nounVals verbVals]', [], 1);

        tbl = table(EEG, Group, WordType, Subject, ...
            'VariableNames', {'EEG','Group','WordType','Subject'});

        lme = fitlme(tbl, 'EEG ~ Group * WordType + (1|Subject)');

        % Coefficient rows:
        % 1 = Intercept
        % 2 = Group
        % 3 = WordType
        % 4 = Group:WordType

        t_Group_Obs(ch,tp) = lme.Coefficients.tStat(2);
        t_Word_Obs(ch,tp)  = lme.Coefficients.tStat(3);
        t_Int_Obs(ch,tp)   = lme.Coefficients.tStat(4);

    end
end

%% ==========================================================
% Step 2: TFCE transformation
%% ==========================================================

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Group_Obs = ept_mex_TFCE2D(t_Group_Obs, ChN, E_H);
TFCE_Word_Obs  = ept_mex_TFCE2D(t_Word_Obs,  ChN, E_H);
TFCE_Int_Obs   = ept_mex_TFCE2D(t_Int_Obs,   ChN, E_H);

%% ==========================================================
% Step 3: Permutation tests
%% ==========================================================

nPerm = 999;

TFCE_permMax_Group = nan(nPerm,1);
TFCE_permMax_Word  = nan(nPerm,1);
TFCE_permMax_Int   = nan(nPerm,1);

parfor p = 1:nPerm

    perm_t_Group = nan(nChan,nTime);
    perm_t_Word  = nan(nChan,nTime);
    perm_t_Int   = nan(nChan,nTime);

    %% Group permutation: shuffle group labels between subjects
    permGroup = group(randperm(nSubj));
    Group_perm = kron(permGroup, ones(nCond,1));

    %% WordType permutation: flip Noun/Verb labels within subjects
    flip = randi([0 1], nSubj, 1);

    WordType_perm = zeros(nSubj*nCond, 1);

    for s = 1:nSubj
        idx_s = (2*s-1):(2*s);

        if flip(s) == 0
            WordType_perm(idx_s) = [0; 1];
        else
            WordType_perm(idx_s) = [1; 0];
        end
    end

    %% Interaction permutation:
    % shuffle group labels between subjects while preserving within-subject structure
    permGroup_Int = group(randperm(nSubj));
    Group_perm_Int = kron(permGroup_Int, ones(nCond,1));

    for ch = 1:nChan
        for tp = 1:nTime

            nounVals = squeeze(data(:,1,ch,tp));
            verbVals = squeeze(data(:,2,ch,tp));

            EEG = reshape([nounVals verbVals]', [], 1);

            %% Group main effect permutation
            tbl_group = table(EEG, Group_perm, WordType, Subject, ...
                'VariableNames', {'EEG','Group','WordType','Subject'});

            lme_group = fitlme(tbl_group, ...
                'EEG ~ Group * WordType + (1|Subject)');

            perm_t_Group(ch,tp) = lme_group.Coefficients.tStat(2);

            %% WordType main effect permutation
            tbl_word = table(EEG, Group, WordType_perm, Subject, ...
                'VariableNames', {'EEG','Group','WordType','Subject'});

            lme_word = fitlme(tbl_word, ...
                'EEG ~ Group * WordType + (1|Subject)');

            perm_t_Word(ch,tp) = lme_word.Coefficients.tStat(3);

            %% Interaction permutation
            tbl_int = table(EEG, Group_perm_Int, WordType, Subject, ...
                'VariableNames', {'EEG','Group','WordType','Subject'});

            lme_int = fitlme(tbl_int, ...
                'EEG ~ Group * WordType + (1|Subject)');

            perm_t_Int(ch,tp) = lme_int.Coefficients.tStat(4);

        end
    end

    TFCE_perm_Group = ept_mex_TFCE2D(perm_t_Group, ChN, E_H);
    TFCE_perm_Word  = ept_mex_TFCE2D(perm_t_Word,  ChN, E_H);
    TFCE_perm_Int   = ept_mex_TFCE2D(perm_t_Int,   ChN, E_H);

    TFCE_permMax_Group(p) = max(abs(TFCE_perm_Group(:)));
    TFCE_permMax_Word(p)  = max(abs(TFCE_perm_Word(:)));
    TFCE_permMax_Int(p)   = max(abs(TFCE_perm_Int(:)));

end

%% ==========================================================
% Step 4: TFCE correction
%% ==========================================================

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

%% ==========================================================
% Step 5: Store results
%% ==========================================================

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

%% ==========================================================
% Step 6: Plot significant effects
%% ==========================================================

plot_tfce_result(t_Group_Obs, Mask_Group, times, e_loc, ...
    'TFCE-corrected Group Main Effect');

plot_tfce_result(t_Word_Obs, Mask_Word, times, e_loc, ...
    'TFCE-corrected WordType Main Effect');

%plot_tfce_result(t_Int_Obs, Mask_Int, times, e_loc, ...
    'TFCE-corrected Group x WordType Interaction');

%% ==========================================================
% Step 7: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/10_TFCE_split_plot_LME_results.mat', ...
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local plotting function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_tfce_result(tMap, Mask, times, e_loc, plotTitle)

    mT = tMap;
    mT(~Mask) = 0;

    figure;

    imagesc(times, 1:size(mT,1), mT);
    axis xy;

    xlim([-200 800]);

    set(gca, ...
        'YTick', 1:size(mT,1), ...
        'YTickLabel', {e_loc.labels}, ...
        'XTick', -200:200:800, ...
        'TickLength', [0 0], ...
        'FontSize', 15, ...
        'FontName', 'Arial');

    xlabel('Time (ms)');
    ylabel('Channel');
    title(plotTitle);

    colorbar;

end