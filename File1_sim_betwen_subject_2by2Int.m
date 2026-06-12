clear; clc; close all;
rng(123);
clear; clc; close all;
rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nPerCell = 30;
nSub = 4 * nPerCell;

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
% Define 2 x 2 between-subject design
%% ==========================================================

% group 0 = A- B-
% group 1 = A- B+
% group 2 = A+ B-
% group 3 = A+ B+

group = [
    zeros(nPerCell,1);
    ones(nPerCell,1);
    2 * ones(nPerCell,1);
    3 * ones(nPerCell,1)
];

% Effect coding: -1 / +1
A = [
    -ones(nPerCell,1);
    -ones(nPerCell,1);
     ones(nPerCell,1);
     ones(nPerCell,1)
];

B = [
    -ones(nPerCell,1);
     ones(nPerCell,1);
    -ones(nPerCell,1);
     ones(nPerCell,1)
];

AB = A .* B;

%% ==========================================================
% Simulate baseline EEG noise
%% ==========================================================

data = noiseSD * randn(nSub, nChan, nTime);

%% ==========================================================
% Define P300 effect
%% ==========================================================

p300Latency = 300;
p300Width = 70;

p300 = exp(-(times - p300Latency).^2 ./ (2 * p300Width^2));
p300 = p300(:);

amp_Aminus_Bminus = 3.0;
amp_Aminus_Bplus  = 6.0;
amp_Aplus_Bminus  = 4.0;
amp_Aplus_Bplus   = 12.0;

cellAmps = [
    amp_Aminus_Bminus;
    amp_Aminus_Bplus;
    amp_Aplus_Bminus;
    amp_Aplus_Bplus
];

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
% Inject P300 into each group
%% ==========================================================

for g = 0:3

    subjIdx = find(group == g);
    amp = cellAmps(g + 1);

    for chIdx = 1:length(effectChans)

        ch = effectChans(chIdx);
        effectWave = weights(chIdx) * amp * p300;

        for s = subjIdx'

            tmp = squeeze(data(s, ch, :));
            tmp = tmp + effectWave;

            data(s, ch, :) = reshape(tmp, 1, 1, nTime);

        end
    end
end

%% ==========================================================
% Figure 1: ERP waveform at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

Aminus_ERP = squeeze(mean(data(A == -1, channelToPlot, :), 1));
Aplus_ERP  = squeeze(mean(data(A ==  1, channelToPlot, :), 1));

Bminus_ERP = squeeze(mean(data(B == -1, channelToPlot, :), 1));
Bplus_ERP  = squeeze(mean(data(B ==  1, channelToPlot, :), 1));

figure;

plot(times, Aminus_ERP, ...
    'Color', [0.00 0.45 0.74], 'LineWidth', 2); hold on;  % blue

plot(times, Aplus_ERP, ...
    'Color', [0.85 0.33 0.10], 'LineWidth', 2);           % orange

plot(times, Bminus_ERP, ...
    'Color', [0.47 0.67 0.19], 'LineWidth', 2);           % green

plot(times, Bplus_ERP, ...
    'Color', [0.49 0.18 0.56], 'LineWidth', 2);           % purple

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('ERP waveform at Pz');

legend('A-', 'A+', 'B-', 'B+');
grid on;

%% ==========================================================
% Figure 2: Observed main effect of A
%% ==========================================================

ADiff = squeeze(mean(data(A == -1, :, :), 1) - ...
                mean(data(A ==  1, :, :), 1));

figure;

imagesc(times, 1:nChan, ADiff);
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
title('Observed A- minus A+ Difference');
colorbar;

%% ==========================================================
% Figure 3: Observed main effect of B
%% ==========================================================

BDiff = squeeze(mean(data(B == -1, :, :), 1) - ...
                mean(data(B ==  1, :, :), 1));

figure;

imagesc(times, 1:nChan, BDiff);
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
title('Observed B- minus B+ Difference');
colorbar;

%% ==========================================================
% Figure 4: Observed interaction effect
%% ==========================================================

interactionDiff = squeeze( ...
    mean(data(AB ==  1, :, :), 1) - ...
    mean(data(AB == -1, :, :), 1));

figure;

imagesc(times, 1:nChan, interactionDiff);
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
title('Observed Interaction Difference: AB+ minus AB-');
colorbar;

%% ==========================================================
% Figure 5: Ground-truth A effect
%% ==========================================================

meanAmp_Aminus = mean([amp_Aminus_Bminus, amp_Aminus_Bplus]);
meanAmp_Aplus  = mean([amp_Aplus_Bminus,  amp_Aplus_Bplus]);

trueADiffWave = (meanAmp_Aminus - meanAmp_Aplus) * p300;

truthADiff = zeros(nChan, nTime);

for chIdx = 1:length(effectChans)

    ch = effectChans(chIdx);
    truthADiff(ch, :) = weights(chIdx) * trueADiffWave';

end

figure;

imagesc(times, 1:nChan, truthADiff);
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
title('Ground-Truth A Effect');
colorbar;

%% ==========================================================
% Figure 6: Ground-truth B effect
%% ==========================================================

meanAmp_Bminus = mean([amp_Aminus_Bminus, amp_Aplus_Bminus]);
meanAmp_Bplus  = mean([amp_Aminus_Bplus,  amp_Aplus_Bplus]);

trueBDiffWave = (meanAmp_Bminus - meanAmp_Bplus) * p300;

truthBDiff = zeros(nChan, nTime);

for chIdx = 1:length(effectChans)

    ch = effectChans(chIdx);
    truthBDiff(ch, :) = weights(chIdx) * trueBDiffWave';

end

figure;

imagesc(times, 1:nChan, truthBDiff);
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
title('Ground-Truth B Effect');
colorbar;

%% ==========================================================
% Figure 7: Ground-truth interaction effect
%% ==========================================================

meanAmp_ABplus  = mean([amp_Aminus_Bminus, amp_Aplus_Bplus]);
meanAmp_ABminus = mean([amp_Aminus_Bplus,  amp_Aplus_Bminus]);

trueInteractionWave = (meanAmp_ABplus - meanAmp_ABminus) * p300;

truthInteraction = zeros(nChan, nTime);

for chIdx = 1:length(effectChans)

    ch = effectChans(chIdx);
    truthInteraction(ch, :) = weights(chIdx) * trueInteractionWave';

end

figure;

imagesc(times, 1:nChan, truthInteraction);
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
title('Ground-Truth Interaction Effect');
colorbar;

%% ==========================================================
% Figure 8: Topography at 300 ms
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~, peakIdx] = min(abs(times - 300));

if exist('topoplot', 'file')

    figure;
    topoplot(ADiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('A- minus A+ Difference at 300 ms');

    figure;
    topoplot(BDiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('B- minus B+ Difference at 300 ms');

    figure;
    topoplot(interactionDiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Interaction Difference at 300 ms');

else

    warning('topoplot not found. Please add EEGLAB to your MATLAB path.');

end

%% ==========================================================
% Design matrix for later GLM
%% ==========================================================

X = [A, B, AB];

%% ==========================================================
% Save dataset
%% ==========================================================

if ~exist('./data', 'dir')
    mkdir('./data');
end

save('./data/03_simulated_between_subject_2by2Int_EEG.mat', ...
    'data', ...
    'group', ...
    'A', ...
    'B', ...
    'AB', ...
    'X', ...
    'times', ...
    'effectChans', ...
    'chanlocs_EEG');