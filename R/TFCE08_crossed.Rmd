%% ==========================================================
% TFCE analysis
% Fully crossed subject-item design
%
% Input file contains only:
%   EEGdata
%   designTable
%
% EEGdata:
%   Subject-item-condition rows x Channels x Time
%   2400 x 32 x 251
%
% designTable:
%   Subject
%   Item
%   CondCode    % -1 = Control, 1 = Treatment
%   CondName
%
% Model:
%   EEG ~ CondCode + (1|Subject) + (1|Item)
%
% Test:
%   Treatment - Control fixed effect
%
% Permutation:
%   Flip condition labels within each Subject x Item pair
%% ==========================================================

clear; clc; close all;
fprintf('\nStarting fully crossed TFCE analysis...\n');

%% ==========================================================
% Load data
%% ==========================================================

load('../data/08_simulated_fully_crossed_subject_item_EEG.mat', ...
     'EEGdata', 'designTable');

times = -200:4:800;

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
% Basic checks
%% ==========================================================

[nRows, nChan, nTime] = size(EEGdata);

Subject  = designTable.Subject(:);
Item     = designTable.Item(:);
CondCode = designTable.CondCode(:);

if height(designTable) ~= nRows
    error('Rows in designTable must match rows in EEGdata.');
end

if length(times) ~= nTime
    error('Length of times does not match number of time points.');
end

if length(e_loc) ~= nChan
    error('Number of channel locations does not match number of channels.');
end

pairID = findgroups(Subject, Item);
nPairs = max(pairID);

if nRows ~= nPairs * 2
    error('Expected exactly two condition rows per Subject x Item pair.');
end

for pID = 1:nPairs

    idxPair = pairID == pID;

    if sum(idxPair) ~= 2
        error('Each Subject x Item pair must have exactly two rows.');
    end

    if ~all(sort(CondCode(idxPair)) == [-1; 1])
        error('Each Subject x Item pair must have one Control (-1) and one Treatment (1).');
    end

end

%% ==========================================================
% Variables for LME
%% ==========================================================

SubjectLME = categorical(Subject);
ItemLME    = categorical(Item);

% 0 = Control, 1 = Treatment
Condition = double(CondCode == 1);

%% ==========================================================
% Step 1: Observed LME t-map
%% ==========================================================

fprintf('Step 1: Computing observed t-statistic map...\n');

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        EEG = double(squeeze(EEGdata(:, ch, tp)));

        tbl = table(EEG, Condition, SubjectLME, ItemLME, ...
            'VariableNames', {'EEG','Condition','Subject','Item'});

        lme = fitlme(tbl, ...
            'EEG ~ Condition + (1|Subject) + (1|Item)');

        % Row 2 = Treatment - Control fixed effect
        t_Obs(ch,tp) = lme.Coefficients.tStat(2);

    end
end

fprintf('Observed t-statistic map completed.\n');

%% ==========================================================
% Step 2: Observed TFCE map
%% ==========================================================

fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');

%% ==========================================================
% Step 3: Permutation by flipping labels within Subject x Item
%% ==========================================================

nPerm = 999;
TFCE_permMax = nan(nPerm,1);

fprintf('Step 3: Starting condition-label flipping: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    perm_t = nan(nChan, nTime);

    CondCode_perm = CondCode;

    for pID = 1:nPairs

        idxPair = find(pairID == pID);

        if rand > 0.5
            CondCode_perm(idxPair) = flipud(CondCode_perm(idxPair));
        end

    end

    Condition_perm = double(CondCode_perm == 1);

    Subject_perm = categorical(Subject);
    Item_perm    = categorical(Item);

    for ch = 1:nChan
        for tp = 1:nTime

            EEG = double(squeeze(EEGdata(:, ch, tp)));

            tbl = table(EEG, Condition_perm, Subject_perm, Item_perm, ...
                'VariableNames', {'EEG','Condition','Subject','Item'});

            lme = fitlme(tbl, ...
                'EEG ~ Condition + (1|Subject) + (1|Item)');

            perm_t(ch,tp) = lme.Coefficients.tStat(2);

        end
    end

    fprintf('At the %dth permutation\n', p);

    TFCE_perm = ept_mex_TFCE2D(perm_t, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end

fprintf('\nPermutation testing completed.\n');

%% ==========================================================
% Step 4: TFCE correction
%% ==========================================================

fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;

nPerm = length(TFCE_permMax);
maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);
maxTFCEcrit = maxTFCE(round(nPerm*(1-Alpha)));

Mask = abs(TFCE_Obs) >= maxTFCEcrit;

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime
        P_Values(ch, tpoint) = (sum(TFCE_permMax >= abs(TFCE_Obs(ch, tpoint))) + 1) / ...
        (nPerm + 1);
    end
end

fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', maxTFCEcrit);

%% ==========================================================
% Step 5: Store results
%% ==========================================================

fprintf('Step 5: Storing results...\n');

Results = struct();

Results.Obs          = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_Null    = TFCE_permMax;
Results.maxTFCEcrit     = maxTFCEcrit;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.alpha        = alpha;
Results.nPerm        = nPerm;
Results.model        = 'EEG ~ Condition + (1|Subject) + (1|Item)';
Results.test         = 'Treatment - Control fixed effect';
Results.permutation  = 'Condition-label flipping within Subject x Item pair';
Results.times        = times;
Results.e_loc        = e_loc;

fprintf('Results stored.\n');

%% ==========================================================
% Step 6: Plot significant observed t-values
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
title('TFCE-corrected Fully Crossed LME Effect: Treatment - Control');

colorbar;

%% ==========================================================
% Step 7: Plot observed TFCE map
%% ==========================================================

figure;

imagesc(times, 1:nChan, TFCE_Obs);
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
title('Observed TFCE Map: Fully Crossed LME Treatment Effect');

colorbar;

%% ==========================================================
% Step 8: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/08_TFCE_fully_crossed_subject_item_results.mat', ...
     'Results');

disp('Fully crossed subject-item TFCE analysis completed and saved.');