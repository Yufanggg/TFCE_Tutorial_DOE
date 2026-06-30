%% ==========================================================
% TFCE analysis for fully crossed Subject x Item EEG design
%
% Original data:
%   data = Subjects x Items x Conditions x Channels x Time
%
% Long-format data:
%   data_long = Observations x Channels x Time
%
% Design:
%   Subject
%   Item
%   ConditionCode: -1 = Control, 1 = Treatment
%
% Model:
%   EEG ~ Condition + (1|Subject) + (1|Item)
%% ==========================================================

clear; clc; close all;

load('../data/08_simulated_fully_crossed_subject_item_EEG.mat');

%% Load channel locations

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

%% Basic checks

[nSubj, nItem, nCond, nChan, nTime] = size(data);

if nCond ~= 2
    error('Expected data format: Subjects x Items x 2 Conditions x Channels x Time');
end

%% ==========================================================
% Step 0: Convert to long format
%% ==========================================================

nObs = nSubj * nItem * nCond;

Subject = zeros(nObs, 1);
Item = zeros(nObs, 1);
Condition = zeros(nObs, 1);

data_long = zeros(nObs, nChan, nTime);

row = 0;

for s = 1:nSubj
    for i = 1:nItem
        for c = 1:nCond

            row = row + 1;

            Subject(row) = s;
            Item(row) = i;

            % -1 = Control, 1 = Treatment
            Condition(row) = 2*c - 3;

            data_long(row,:,:) = squeeze(data(s,i,c,:,:));

        end
    end
end

Subject = categorical(Subject);
Item = categorical(Item);

designTableLong = table(Subject, Item, Condition);

disp(designTableLong(1:20,:));

%% ==========================================================
% Step 1: Observed LME t-map
%% ==========================================================

fprintf('Computing observed t-map...\n');

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        EEG = double(data_long(:, ch, tpoint));

        tbl = table(EEG, Condition, Subject, Item, ...
            'VariableNames', {'EEG','Condition','Subject','Item'});

        lme = fitlme(tbl, ...
            'EEG ~ Condition + (1|Subject) + (1|Item)');

        t_Obs(ch,tpoint) = lme.Coefficients.tStat(2);

    end
end

fprintf('Observed t-map completed.\n');

%% ==========================================================
% Step 2: TFCE transform of observed t-map
%% ==========================================================

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

%% ==========================================================
% Step 3: Permutation test
%
% Fully crossed design:
%   Subject and Item are kept fixed.
%   Condition labels are swapped within each Subject x Item cell.
%
% This preserves:
%   subject structure,
%   item structure,
%   and the paired Control/Treatment comparison.
%% ==========================================================

nperms = 999;

fprintf('Starting block-restricted permutation testing: %d permutations...\n', nperms);

TFCE_permMax = nan(nperms, 1);

parfor p = 1:nperms

    perm_t_local = zeros(nChan, nTime);

    %% Swap Control/Treatment within each Subject x Item cell

    data_perm = data;

    for s = 1:nSubj
        for i = 1:nItem

            if rand < 0.5

                data_perm(s,i,[1 2],:,:) = ...
                    data_perm(s,i,[2 1],:,:);

            end

        end
    end

    %% Convert permuted data to long format

    data_perm_long = zeros(nObs, nChan, nTime);

    row = 0;

    for s = 1:nSubj
        for i = 1:nItem
            for c = 1:nCond

                row = row + 1;
                data_perm_long(row,:,:) = squeeze(data_perm(s,i,c,:,:));

            end
        end
    end

    %% Compute permuted LME t-map

    for ch = 1:nChan
        for tpoint = 1:nTime

            EEG = double(data_perm_long(:, ch, tpoint));

            tbl = table(EEG, Condition, Subject, Item, ...
                'VariableNames', {'EEG','Condition','Subject','Item'});

            lme = fitlme(tbl, ...
                'EEG ~ Condition + (1|Subject) + (1|Item)');

            perm_t_local(ch,tpoint) = lme.Coefficients.tStat(2);

        end
    end

    %% TFCE transform and max statistic

    TFCE_perm = ept_mex_TFCE2D(perm_t_local, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

    fprintf('Finished permutation %d\n', p);

end

fprintf('Permutation testing completed.\n');

%% ==========================================================
% Step 4: TFCE correction
%% ==========================================================

Alpha = 0.05;

maxTFCEcrit = prctile(TFCE_permMax, 100 * (1 - Alpha));

Mask = abs(TFCE_Obs) >= maxTFCEcrit;

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        P_Values(ch,tpoint) = ...
            (sum(TFCE_permMax >= abs(TFCE_Obs(ch,tpoint))) + 1) / ...
            (nperms + 1);

    end
end

Results.Obs          = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_permMax = TFCE_permMax;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.data_long    = data_long;
Results.designTable  = designTableLong;

%% ==========================================================
% Step 5: Plot significant observed t-values
%% ==========================================================

mT = Results.Obs;
mT(~Results.Mask) = 0;

tick_labels = {e_loc.labels};

figure;

imagesc(times, 1:nChan, mT);
axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tick_labels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Significant Condition Effects: Treatment - Control');

colorbar;

%% ==========================================================
% Step 6: Save results
%% ==========================================================

save('../data/08_TFCE_fully_crossed_subject_item_results.mat', ...
    'Results', ...
    't_Obs', ...
    'TFCE_Obs', ...
    'TFCE_permMax', ...
    'P_Values', ...
    'Mask', ...
    'data_long', ...
    'designTableLong', ...
    'times', ...
    'e_loc');

disp('TFCE results saved: ../data/08_TFCE_fully_crossed_subject_item_results.mat');