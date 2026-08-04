%% ==========================================================
% Simulated EEG/ERP dataset
% 2 x 2 between-subject factorial design
%
% Final saved variables:
%   EEGdata
%   designTable
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
%% ==========================================================

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
tickLabels = {chanlocs_EEG.labels};

%% ==========================================================
% Define 2 x 2 between-subject design
%% ==========================================================

group = [
    zeros(nPerCell,1);        % A- B-
    ones(nPerCell,1);         % A- B+
    2 * ones(nPerCell,1);     % A+ B-
    3 * ones(nPerCell,1)      % A+ B+
];

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
% Design table
%% ==========================================================

Subject = (1:nSub)';

GroupLabel = cell(nSub,1);
GroupLabel(group == 0) = {'A- B-'};
GroupLabel(group == 1) = {'A- B+'};
GroupLabel(group == 2) = {'A+ B-'};
GroupLabel(group == 3) = {'A+ B+'};

FactorA_Label = cell(nSub,1);
FactorA_Label(A == -1) = {'A-'};
FactorA_Label(A ==  1) = {'A+'};

FactorB_Label = cell(nSub,1);
FactorB_Label(B == -1) = {'B-'};
FactorB_Label(B ==  1) = {'B+'};

designTable = table(Subject, group, GroupLabel, A, FactorA_Label, ...
    B, FactorB_Label, AB, ...
    'VariableNames', {'Subject','GroupCode','GroupLabel', ...
    'FactorA','FactorA_Label','FactorB','FactorB_Label','Interaction'});

disp(unique(designTable(:, {'GroupCode','GroupLabel','FactorA','FactorB','Interaction'}), 'rows'));

%% ==========================================================
% Define P300 signal
%% ==========================================================

p300Latency = 300;
p300Width   = 70;

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
% Simulate EEG/ERP data
%% ==========================================================

EEGdata = noiseSD * randn(nSub, nChan, nTime);

for s = 1:nSub

    amp = cellAmps(group(s) + 1);

    for chIdx = 1:length(effectChans)

        ch = effectChans(chIdx);

        effectWave = weights(chIdx) * amp * p300;

        tmp = squeeze(EEGdata(s, ch, :));
        tmp = tmp + effectWave;

        EEGdata(s, ch, :) = reshape(tmp, 1, 1, nTime);

    end
end

%% ==========================================================
% Randomly remove subjects (unbalanced design)
%% ==========================================================

%% ==========================================================
% Randomly remove subjects within each cell
%% ==========================================================

keep = true(nSub,1);

for g = 0:3

    idx = find(group == g);

    % Randomly remove 0--8 subjects from this group
    nRemove = randi([0 8]);

    removeIdx = idx(randperm(length(idx), nRemove));

    keep(removeIdx) = false;

end

fprintf('\nFinal sample sizes\n');

for g = 0:3
    fprintf('Group %d : %d\n', g, sum(group(keep)==g));
end

EEGdata = EEGdata(keep,:,:);
designTable = designTable(keep,:);

group = designTable.GroupCode;
A     = designTable.FactorA;
B     = designTable.FactorB;
AB    = designTable.Interaction;

nSub = height(designTable);
designTable.Subject = (1:nSub)';

%% ==========================================================
% Compute cell-level ERPs
%% ==========================================================

erp00 = squeeze(mean(EEGdata(group == 0, :, :), 1));   % A- B-
erp01 = squeeze(mean(EEGdata(group == 1, :, :), 1));   % A- B+
erp10 = squeeze(mean(EEGdata(group == 2, :, :), 1));   % A+ B-
erp11 = squeeze(mean(EEGdata(group == 3, :, :), 1));   % A+ B+

%% ==========================================================
% Compute observed factorial effects
%% ==========================================================

erp_Aminus = squeeze(mean(EEGdata(A == -1, :, :), 1));
erp_Aplus  = squeeze(mean(EEGdata(A ==  1, :, :), 1));

erp_Bminus = squeeze(mean(EEGdata(B == -1, :, :), 1));
erp_Bplus  = squeeze(mean(EEGdata(B ==  1, :, :), 1));

ADiff = erp_Aplus - erp_Aminus;
BDiff = erp_Bplus - erp_Bminus;

interactionDiff = (erp11 - erp10) - (erp01 - erp00);

%% ==========================================================
% Figure 1: ERP waveforms at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

figure;

plot(times, erp_Aminus(channelToPlot, :), ...
    'Color', [0.00 0.45 0.74], 'LineWidth', 2); hold on;

plot(times, erp_Aplus(channelToPlot, :), ...
    'Color', [0.85 0.33 0.10], 'LineWidth', 2);

plot(times, erp_Bminus(channelToPlot, :), ...
    'Color', [0.47 0.67 0.19], 'LineWidth', 2);

plot(times, erp_Bplus(channelToPlot, :), ...
    'Color', [0.49 0.18 0.56], 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title(sprintf('ERP waveform at %s', chanlocs_EEG(channelToPlot).labels));

legend('A-', 'A+', 'B-', 'B+');
grid on;

%% ==========================================================
% Figure 2: Observed main effect of A
%% ==========================================================

figure;

imagesc(times, 1:nChan, ADiff);
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
title('Observed A Effect: A+ minus A-');
colorbar;

%% ==========================================================
% Figure 3: Observed main effect of B
%% ==========================================================

figure;

imagesc(times, 1:nChan, BDiff);
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
title('Observed B Effect: B+ minus B-');
colorbar;

%% ==========================================================
% Figure 4: Observed interaction effect
%% ==========================================================

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

%% ==========================================================
% Ground-truth factorial effects
%% ==========================================================

meanAmp_Aminus = mean([amp_Aminus_Bminus, amp_Aminus_Bplus]);
meanAmp_Aplus  = mean([amp_Aplus_Bminus,  amp_Aplus_Bplus]);

meanAmp_Bminus = mean([amp_Aminus_Bminus, amp_Aplus_Bminus]);
meanAmp_Bplus  = mean([amp_Aminus_Bplus,  amp_Aplus_Bplus]);

trueADiffWave = (meanAmp_Aplus - meanAmp_Aminus) * p300;
trueBDiffWave = (meanAmp_Bplus - meanAmp_Bminus) * p300;

trueInteractionWave = ...
    ((amp_Aplus_Bplus - amp_Aplus_Bminus) - ...
     (amp_Aminus_Bplus - amp_Aminus_Bminus)) * p300;

truthADiff = zeros(nChan, nTime);
truthBDiff = zeros(nChan, nTime);
truthInteraction = zeros(nChan, nTime);

for chIdx = 1:length(effectChans)

    ch = effectChans(chIdx);

    truthADiff(ch, :) = weights(chIdx) * trueADiffWave';
    truthBDiff(ch, :) = weights(chIdx) * trueBDiffWave';
    truthInteraction(ch, :) = weights(chIdx) * trueInteractionWave';

end

%% ==========================================================
% Figure 5: Ground-truth A effect
%% ==========================================================

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
title('Ground-Truth A Effect: A+ minus A-');
colorbar;

%% ==========================================================
% Figure 6: Ground-truth B effect
%% ==========================================================

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
title('Ground-Truth B Effect: B+ minus B-');
colorbar;

%% ==========================================================
% Figure 7: Ground-truth interaction effect
%% ==========================================================

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

%% ==========================================================
% Figure 8: Topography at 300 ms
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~, peakIdx] = min(abs(times - p300Latency));

if exist('topoplot', 'file')

    figure;
    topoplot(ADiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed A Effect at 300 ms');

    figure;
    topoplot(BDiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed B Effect at 300 ms');

    figure;
    topoplot(interactionDiff(:, peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed Interaction Effect at 300 ms');

else

    warning('topoplot not found. Please add EEGLAB to your MATLAB path.');

end

%% ==========================================================
% Save dataset
% Only EEGdata and designTable are saved
%% ==========================================================

if ~exist('../Data', 'dir')
    mkdir('../Data');
end

save('../Data/04_simulated_between_subject_2by2_EEG_unbalanced.mat', ...
    'EEGdata', ...
    'designTable');

disp('Dataset saved: ../Data/04_simulated_between_subject_2by2_EEG_unbalanced.mat');
disp('Saved variables: EEGdata, designTable');