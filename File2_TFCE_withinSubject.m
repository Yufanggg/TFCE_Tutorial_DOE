
clear all; clc; close all;
load('data/05_simulated_within_subject_EEG.mat')
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
% Data format expected:
% data = subjects x conditions x channels x time
% condition 1 = Control
% condition 2 = Treatment
%% ==========================================================

[nSubj, nCond, nChan, nTime] = size(data);

if nCond ~= 2
    error('Expected data format: subjects x 2 conditions x channels x time');
end

%% ==========================================================
% Step 1: observed paired / within-subject t-map
%% ==========================================================

% Subject-level difference waves
% Subjects x Channels x Time
diffData = squeeze(data(:,2,:,:) - data(:,1,:,:));

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan

    for tpoint = 1:nTime

        diffVals = double(squeeze(diffData(:,ch,tpoint)));

        [~,p,~,stats] = ttest(diffVals, 0);

        t_Obs(ch,tpoint) = stats.tstat;

    end

end

%% ==========================================================
% Step 2: TFCE transform of observed t-map
%% ==========================================================

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

%% ==========================================================
% Step 3: permutation test using random sign-flipping
%% ==========================================================

nperms = 999;

TFCE_permMax = nan(nperms,1);

parfor p = 1:nperms

    % Randomly flip the sign of each subject's difference wave
    signs = randi([0 1], nSubj, 1) * 2 - 1;

    permDiff = diffData;

    for s = 1:nSubj
        permDiff(s,:,:) = signs(s) * permDiff(s,:,:);
    end

    perm_t_local = nan(nChan, nTime);

    for ch = 1:nChan

        for tpoint = 1:nTime

            diffVals = double(squeeze(permDiff(:,ch,tpoint)));

            [~,~,~,stats] = ttest(diffVals, 0);

            perm_t_local(ch,tpoint) = stats.tstat;

        end

    end

    TFCE_perm = ept_mex_TFCE2D(perm_t_local, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end

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
Results.P_Obs        = p_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_permMax = TFCE_permMax;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.diffData     = diffData;

%% ==========================================================
% Step 5: plot significant observed t-values
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
title('Significant Within-subject Effects: Treatment - Control');

colorbar;