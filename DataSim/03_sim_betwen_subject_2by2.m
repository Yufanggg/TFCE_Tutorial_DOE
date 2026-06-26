%% ==========================================================
% Simulated EEG/ERP dataset
% Between-subject 2 x 2 factorial design
%
% Factor A: A- / A+
% Factor B: B- / B+
%
% Groups:
% 0 = A- B-
% 1 = A- B+
% 2 = A+ B-
% 3 = A+ B+
%
% Data dimensions:
% Subjects x Channels x Time
%
% Noise:
% Background Gaussian noise only
%% ==========================================================

clear; clc; close all;
rng(123);

%% Simulation settings

nFactor00 = 30;
nFactor01 = 30;
nFactor10 = 30;
nFactor11 = 30;

nSub = nFactor00 + nFactor01 + nFactor10 + nFactor11;

times = -200:4:800;
nTime = length(times);

noiseSD = 1.5;

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

chanlocs_EEG = chanlocs_1020(idx);
nChan = length(chanlocs_EEG);
tickLabels = {chanlocs_EEG.labels};

%% Create 2 x 2 between-subject design

group = [
    zeros(nFactor00, 1);
    ones(nFactor01, 1);
    2 * ones(nFactor10, 1);
    3 * ones(nFactor11, 1)
];

var1 = 2 * ismember(group, [2 3]) - 1;   % Factor A
var2 = 2 * ismember(group, [1 3]) - 1;   % Factor B
varInt = var1 .* var2;                   % A x B

designCheck = table(group, var1, var2, varInt, ...
    'VariableNames', {'group','FactorA','FactorB','Interaction'});

disp(unique(designCheck, 'rows'));

%% Define P300 signal

p300Latency = 300;
p300Width   = 70;

p300 = exp(-(times - p300Latency).^2 ./ (2 * p300Width^2));
p300 = p300(:);

amp00 = 3.0;   % A- B-
amp01 = 6.0;   % A- B+
amp10 = 4.0;   % A+ B-
amp11 = 7.0;   % A+ B+

cellAmps = [amp00; amp01; amp10; amp11];

%% Define effect channels

effectChanLabels = {'Cz','CP1','CP2','Pz','P3','P4'};

[tf, effectChans] = ismember(effectChanLabels, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChanLabels(~tf), ', '));
end

weights = [0.6 0.8 0.8 1.0 0.75 0.75];

%% Simulate EEG/ERP data

% Background noise only
data = noiseSD * randn(nSub, nChan, nTime);

% Add deterministic group-specific P300 signal
for s = 1:nSub

    amp = cellAmps(group(s) + 1);

    for chIdx = 1:length(effectChans)

        ch = effectChans(chIdx);
        effectWave = weights(chIdx) * amp * p300;

        tmp = squeeze(data(s, ch, :));
        tmp = tmp + effectWave;

        data(s, ch, :) = reshape(tmp, 1, 1, nTime);

    end
end

%% Compute observed effects

erp_Aminus = squeeze(mean(data(var1 == -1, :, :), 1));
erp_Aplus  = squeeze(mean(data(var1 ==  1, :, :), 1));

erp_Bminus = squeeze(mean(data(var2 == -1, :, :), 1));
erp_Bplus  = squeeze(mean(data(var2 ==  1, :, :), 1));

groupADiff = erp_Aplus - erp_Aminus;
groupBDiff = erp_Bplus - erp_Bminus;

interactionDiff = squeeze(mean(data(varInt == 1, :, :), 1) - ...
                          mean(data(varInt == -1, :, :), 1));

%% Figure 1: ERP waveform at Pz

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

figure;

plot(times, erp_Aminus(channelToPlot, :), 'b', 'LineWidth', 2); hold on;
plot(times, erp_Aplus(channelToPlot, :),  'r', 'LineWidth', 2);
plot(times, erp_Bminus(channelToPlot, :), 'g', 'LineWidth', 2);
plot(times, erp_Bplus(channelToPlot, :),  'm', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title(sprintf('ERP waveform at %s', chanlocs_EEG(channelToPlot).labels));

legend('Factor A-', 'Factor A+', 'Factor B-', 'Factor B+');
grid on;

%% Figure 2: Observed Factor A effect

figure;

imagesc(times, 1:nChan, groupADiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tickLabels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed Factor A Effect: A+ minus A-');
colorbar;

%% Figure 3: Observed Factor B effect

figure;

imagesc(times, 1:nChan, groupBDiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tickLabels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed Factor B Effect: B+ minus B-');
colorbar;

%% Figure 4: Observed interaction effect

figure;

imagesc(times, 1:nChan, interactionDiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tickLabels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed Interaction Effect');
colorbar;

%% Ground-truth effects

amp_Aminus = mean([amp00, amp01]);
amp_Aplus  = mean([amp10, amp11]);

amp_Bminus = mean([amp00, amp10]);
amp_Bplus  = mean([amp01, amp11]);

amp_ABplus  = mean([amp00, amp11]);
amp_ABminus = mean([amp01, amp10]);

trueADifference = (amp_Aplus - amp_Aminus) * p300;
trueBDifference = (amp_Bplus - amp_Bminus) * p300;
trueInteraction = (amp_ABplus - amp_ABminus) * p300;

truthADiff = zeros(nChan, nTime);
truthBDiff = zeros(nChan, nTime);
truthInteraction = zeros(nChan, nTime);

for chIdx = 1:length(effectChans)

    ch = effectChans(chIdx);

    truthADiff(ch, :) = weights(chIdx) * trueADifference';
    truthBDiff(ch, :) = weights(chIdx) * trueBDifference';
    truthInteraction(ch, :) = weights(chIdx) * trueInteraction';

end

%% Figure 5: Ground-truth Factor A effect

figure;

imagesc(times, 1:nChan, truthADiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tickLabels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Ground-Truth Factor A Effect');
colorbar;

%% Figure 6: Ground-truth Factor B effect

figure;

imagesc(times, 1:nChan, truthBDiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tickLabels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Ground-Truth Factor B Effect');
colorbar;

%% Figure 7: Ground-truth interaction effect

figure;

imagesc(times, 1:nChan, truthInteraction);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tickLabels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Ground-Truth Interaction Effect');
colorbar;

%% Figure 8: Topography at peak latency

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~, peakIdx] = min(abs(times - p300Latency));

if exist('topoplot', 'file')

    figure;
    topoplot(groupADiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed Factor A Effect at 300 ms');

    figure;
    topoplot(groupBDiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed Factor B Effect at 300 ms');

    figure;
    topoplot(interactionDiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed Interaction Effect at 300 ms');

else

    fprintf('topoplot not found. Please add EEGLAB to the MATLAB path.\n');

end

%% ==========================================================
% Save dataset
%% ==========================================================

if ~exist('../data', 'dir')
    mkdir('../data');
end

save('../data/02_simulated_between_subject_2by2_EEG.mat', ...
     'data', ...
     'group', ...
     'var1', ...
     'var2', ...
     'varInt', ...
     'times', ...
     'effectChans', ...
     'chanlocs_EEG');