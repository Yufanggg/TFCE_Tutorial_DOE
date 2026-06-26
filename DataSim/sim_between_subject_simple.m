%% ==========================================================
% Simulated EEG/ERP dataset
% Between-subject design: Control vs Treatment
%
% Control:   n = 30
% Treatment: n = 30
%
% Data dimensions:
% Subjects x Channels x Time
%
% Noise:
% Background noise only
%
% Ground truth:
% P300 effect around 300 ms
% Larger P300 amplitude in Treatment than Control
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

chanlocs_EEG = chanlocs_1020(idx);
nChan = length(chanlocs_EEG);

%% ==========================================================
% Group labels
%% ==========================================================

% 1 = Control, 2 = Treatment
group = [ones(nControl,1); 2 * ones(nTreatment,1)];

%% ==========================================================
% Define P300 waveform
%% ==========================================================

p300Latency = 300;
p300Width   = 70;

controlAmp   = 3.0;
treatmentAmp = 6.0;

controlP300 = controlAmp * exp(-(times - p300Latency).^2 ./ ...
              (2 * p300Width^2));

treatmentP300 = treatmentAmp * exp(-(times - p300Latency).^2 ./ ...
                (2 * p300Width^2));

controlP300   = controlP300(:);
treatmentP300 = treatmentP300(:);

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
%% ==========================================================

% Data dimensions:
% Subjects x Channels x Time
data = noiseSD * randn(nSub, nChan, nTime);

for s = 1:nSub

    if group(s) == 1
        p300 = controlP300;
    elseif group(s) == 2
        p300 = treatmentP300;
    end

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s, ch, :));
        tmp = tmp + weights(ch_idx) * p300;

        data(s, ch, :) = reshape(tmp, 1, 1, nTime);

    end

end
%% ==========================================================
% Figure 1: ERP waveform
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlWave   = squeeze(mean(data(group == 1, channelToPlot, :), 1));
treatmentWave = squeeze(mean(data(group == 2, channelToPlot, :), 1));

figure;

plot(times, controlWave, 'LineWidth', 2);
hold on;

plot(times, treatmentWave, 'r', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title(sprintf('ERP waveform at Pz, Channel %d', channelToPlot));

legend('Control', 'Treatment');
grid on;

%% ==========================================================
% Figure 2: Observed channel ¡Á time effect map
%% ==========================================================

controlERP   = squeeze(mean(data(group == 1, :, :), 1));
treatmentERP = squeeze(mean(data(group == 2, :, :), 1));

groupDiff = treatmentERP - controlERP;

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
% Figure 4: Topography at peak latency
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~, peakIdx] = min(abs(times - 300));

topo = groupDiff(:, peakIdx);

if exist('topoplot', 'file')

    figure;

    topoplot(topo, chanlocs_plot, 'electrodes', 'labels');

    colorbar;

    title('Observed Group Difference at 300 ms');

else

    fprintf('topoplot not found. Please add EEGLAB to the MATLAB path.\n');

end
%% ==========================================================
% Save dataset
%% ==========================================================
if ~exist('../data', 'dir')
    mkdir('../data');
end
save('../data/01_simulated_between_subject_EEG.mat',...
     'data','group','times','effectChans');