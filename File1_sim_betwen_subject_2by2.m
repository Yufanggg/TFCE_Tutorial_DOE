%% ==========================================================
% Simulated EEG/ERP dataset
% Between-subject 2 x 2 factorial design with interaction
%
% group 0 = A- B-
% group 1 = A- B+
% group 2 = A+ B-
% group 3 = A+ B+
%
% var1 = Factor A, coded -1 / +1
% var2 = Factor B, coded -1 / +1
% varInt = A x B interaction
%
% Data dimensions:
% Subjects x Channels x Time
%% ==========================================================

clear; clc; close all;
rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nFactor00 = 30;   % A- B-
nFactor01 = 30;   % A- B+
nFactor10 = 30;   % A+ B-
nFactor11 = 30;   % A+ B+

nSub = nFactor00 + nFactor01 + nFactor10 + nFactor11;

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
% Create 2 x 2 between-subject design
%% ==========================================================

group = [
    zeros(nFactor00,1);
    ones(nFactor01,1);
    2 * ones(nFactor10,1);
    3 * ones(nFactor11,1)
];

% Effect-coded predictors
var1 = 2 * ismember(group, [2 3]) - 1;   % Factor A: A- = -1, A+ = +1
var2 = 2 * ismember(group, [1 3]) - 1;   % Factor B: B- = -1, B+ = +1
varInt = var1 .* var2;                   % A x B interaction

designCheck = table(group, var1, var2, varInt, ...
    'VariableNames', {'group','FactorA','FactorB','Interaction'});

disp(unique(designCheck, 'rows'));

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

amp00 = 3.0;   % A- B-
amp01 = 6.0;   % A- B+
amp10 = 4.0;   % A+ B-
amp11 = 7.0;   % A+ B+

cellAmps = [amp00; amp01; amp10; amp11];

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
% Inject P300 effect into each group
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

FactorA_minus_ERP = squeeze(mean(data(var1 == -1, channelToPlot, :), 1));
FactorA_plus_ERP  = squeeze(mean(data(var1 ==  1, channelToPlot, :), 1));

FactorB_minus_ERP = squeeze(mean(data(var2 == -1, channelToPlot, :), 1));
FactorB_plus_ERP  = squeeze(mean(data(var2 ==  1, channelToPlot, :), 1));

figure;

plot(times, FactorA_minus_ERP, 'b', 'LineWidth', 2); hold on;   % blue
plot(times, FactorA_plus_ERP,  'r', 'LineWidth', 2);            % red
plot(times, FactorB_minus_ERP, 'g', 'LineWidth', 2);            % green
plot(times, FactorB_plus_ERP,  'm', 'LineWidth', 2);            % magenta

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title(sprintf('ERP waveform at %s', chanlocs_EEG(channelToPlot).labels));

legend('Factor A-', 'Factor A+', 'Factor B-', 'Factor B+');
grid on;

%% ==========================================================
% Figure 2: Observed Factor A effect
%% ==========================================================

tickLabels = {chanlocs_EEG.labels};

groupADiff = squeeze(mean(data(var1 == 1, :, :), 1) - ...
                     mean(data(var1 == -1, :, :), 1));

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

%% ==========================================================
% Figure 3: Observed Factor B effect
%% ==========================================================

groupBDiff = squeeze(mean(data(var2 == 1, :, :), 1) - ...
                     mean(data(var2 == -1, :, :), 1));

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

%% ==========================================================
% Figure 4: Observed interaction effect
%% ==========================================================

interactionDiff = squeeze(mean(data(varInt == 1, :, :), 1) - ...
                          mean(data(varInt == -1, :, :), 1));

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
title('Observed Interaction Effect: AB+ minus AB-');
colorbar;

%% ==========================================================
% Figure 5: Ground-truth Factor A effect
%% ==========================================================

amp_Aminus = mean([amp00, amp01]);
amp_Aplus  = mean([amp10, amp11]);

trueADifference = (amp_Aplus - amp_Aminus) * p300;

truthADiff = zeros(nChan, nTime);

for chIdx = 1:length(effectChans)

    ch = effectChans(chIdx);
    truthADiff(ch, :) = weights(chIdx) * trueADifference';

end

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
title('Ground-Truth Simulated Effect for Factor A');
colorbar;

%% ==========================================================
% Figure 6: Ground-truth Factor B effect
%% ==========================================================

amp_Bminus = mean([amp00, amp10]);
amp_Bplus  = mean([amp01, amp11]);

trueBDifference = (amp_Bplus - amp_Bminus) * p300;

truthBDiff = zeros(nChan, nTime);

for chIdx = 1:length(effectChans)

    ch = effectChans(chIdx);
    truthBDiff(ch, :) = weights(chIdx) * trueBDifference';

end

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
title('Ground-Truth Simulated Effect for Factor B');
colorbar;

%% ==========================================================
% Figure 7: Ground-truth interaction effect
%% ==========================================================

amp_ABplus  = mean([amp00, amp11]);
amp_ABminus = mean([amp01, amp10]);

trueInteraction = (amp_ABplus - amp_ABminus) * p300;

truthInteraction = zeros(nChan, nTime);

for chIdx = 1:length(effectChans)

    ch = effectChans(chIdx);
    truthInteraction(ch, :) = weights(chIdx) * trueInteraction';

end

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
title('Ground-Truth Simulated Interaction Effect');
colorbar;

%% ==========================================================
% Figure 8: Topography at peak latency
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~, peakIdx] = min(abs(times - 300));

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

if ~exist('./data', 'dir')
    mkdir('./data');
end

save('./data/02_simulated_between_subject_2by2_EEG.mat', ...
     'data', ...
     'group', ...
     'var1', ...
     'var2', ...
     'varInt', ...
     'times', ...
     'effectChans', ...
     'chanlocs_EEG');