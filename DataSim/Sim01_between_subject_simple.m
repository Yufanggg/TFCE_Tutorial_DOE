%% ==========================================================
% Simulated EEG/ERP dataset
% Simple between-subject design: Control vs Treatment
%
% Final saved variables:
%   EEGdata
%   designTable
%% ==========================================================

clear; clc; close all;
rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nControl   = 30;
nTreatment = 30;
nSub       = nControl + nTreatment;

condNames = {'Control', 'Treatment'};

times = -200:4:800;
nTime = length(times);

noiseSD = 1.5;

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

chanlocs_EEG = chanlocs_1020(idx);
nChan = length(chanlocs_EEG);

%% ==========================================================
% Group labels
%% ==========================================================

group = [-ones(nControl,1); ones(nTreatment,1)];
subjectID = (1:nSub)';

%% ==========================================================
% Define P300 waveform
%% ==========================================================

p300Latency = 300;
p300Width   = 70;

controlAmp   = 3.0;
treatmentAmp = 6.0;

p300Shape = exp(-(times - p300Latency).^2 ./ ...
                (2 * p300Width^2));
p300Shape = p300Shape(:);

controlP300   = controlAmp   * p300Shape;
treatmentP300 = treatmentAmp * p300Shape;

%% ==========================================================
% Define effect channels
%% ==========================================================

effectChanLabels = {'Cz','CP1','CP2','Pz','P3','P4'};

[tf, effectChans] = ismember(effectChanLabels, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChanLabels(~tf), ', '));
end

weights = [0.6 0.8 0.8 1.0 0.75 0.75];

%% ==========================================================
% Simulate EEG/ERP data
% Dimensions: Subjects x Channels x Time
%% ==========================================================

EEGdata = noiseSD * randn(nSub, nChan, nTime);

for s = 1:nSub

    if group(s) == -1
        p300 = controlP300;
    else
        p300 = treatmentP300;
    end

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(EEGdata(s, ch, :));
        tmp = tmp + weights(ch_idx) * p300;

        EEGdata(s, ch, :) = reshape(tmp, 1, 1, nTime);

    end

end

%% ==========================================================
% Compute group-level ERPs
%% ==========================================================

controlData   = EEGdata(group == -1, :, :);
treatmentData = EEGdata(group ==  1, :, :);

controlERP   = squeeze(mean(controlData, 1));
treatmentERP = squeeze(mean(treatmentData, 1));

groupDiff = treatmentERP - controlERP;

%% ==========================================================
% Figure 1: ERP waveform at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlWave   = controlERP(channelToPlot, :);
treatmentWave = treatmentERP(channelToPlot, :);

figure;

plot(times, controlWave, 'LineWidth', 2);
hold on;

plot(times, treatmentWave, 'r', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title(sprintf('ERP waveform at Pz, Channel %d', channelToPlot));

legend(condNames);
grid on;

%% ==========================================================
% Figure 2: Observed channel x time effect map
%% ==========================================================

figure;

imagesc(times, 1:nChan, groupDiff);
axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {chanlocs_EEG.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed Group Difference: Treatment - Control');

colorbar;

%% ==========================================================
% Figure 3: Ground-truth injected effect
%% ==========================================================

truthDiff = zeros(nChan, nTime);

trueDifference = treatmentP300 - controlP300;

for ch_idx = 1:length(effectChans)

    ch = effectChans(ch_idx);

    truthDiff(ch, :) = weights(ch_idx) * trueDifference';

end

figure;

imagesc(times, 1:nChan, truthDiff);
axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {chanlocs_EEG.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Ground-Truth Simulated Effect: Treatment - Control');

colorbar;

%% ==========================================================
% Figure 4: Topography of observed group difference at 300 ms
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~, peakIdx] = min(abs(times - p300Latency));

topo = groupDiff(:, peakIdx);

if exist('topoplot', 'file')

    figure;

    topoplot(topo, chanlocs_plot, 'electrodes', 'labels');

    colorbar;

    title(sprintf('Observed Group Difference at %d ms', p300Latency));

else

    fprintf('topoplot not found. Please add EEGLAB to the MATLAB path.\n');

end

%% ==========================================================
% Figure 5: Topography of ground-truth effect at 300 ms
%% ==========================================================

topoTruth = truthDiff(:, peakIdx);

if exist('topoplot', 'file')

    figure;

    topoplot(topoTruth, chanlocs_plot, 'electrodes', 'labels');

    colorbar;

    title(sprintf('Ground-Truth Effect at %d ms', p300Latency));

end

%% ==========================================================
% Design table
%% ==========================================================

GroupLabel = cell(nSub,1);
GroupLabel(group == -1) = {'Control'};
GroupLabel(group ==  1) = {'Treatment'};

designTable = table(subjectID, group, GroupLabel, ...
    'VariableNames', {'Subject','GroupCode','GroupLabel'});

disp(designTable(1:10,:));

%% ==========================================================
% Save dataset
% Only EEGdata and designTable will be saved
%% ==========================================================

if ~exist('../data', 'dir')
    mkdir('../data');
end

save('../data/01_simulated_between_subject_EEG.mat', ...
     'EEGdata', ...
     'designTable');

disp('Dataset saved: ../data/01_simulated_between_subject_EEG.mat');
disp('Saved variables: EEGdata, designTable');