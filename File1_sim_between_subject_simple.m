%%% between-subject design (simple version)
%% ==========================================================
% Simulated EEG/ERP dataset
% Between-subject design:
% Control (n=30) vs Treatment (n=30)
%
% Data dimensions:
% Subjects x Channels x Time
%
% Ground truth:
% Positive ERP effect at 300 ms
% Channels: 30,31,37,38
%% ==========================================================

clear; clc; close all; rng(123);
%% Simulation settings

nControl   = 30;
nTreatment = 30;

nSub  = nControl + nTreatment;
nChan = 32;

times = -200:4:800;
nTime = length(times);
chanlocs_1020 = readlocs('standard_1005.elc');
% chanLabels_64 = {
% 'Fp1','AF7','AF3','F1','F3','F5','F7',...
% 'FT7','FC5','FC3','FC1','C1','C3','C5','T7',...
% 'TP7','CP5','CP3','CP1','P1','P3','P5','P7',...
% 'P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz',...
% 'Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8',...
% 'FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8',...
% 'TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10',...
% 'PO8','PO4','O2'
% };
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
% chanlocs_EEG = chanlocs_1020(ismember({chanlocs_1020.labels}, chanLabels_32));
allLabels = {chanlocs_1020.labels};
[tf, idx] = ismember(chanLabels_32, allLabels);

if any(~tf)
    error('Missing channels: %s', strjoin(chanLabels_32(~tf), ', '));
end

chanlocs_EEG = chanlocs_1020(idx);
nChan = length(chanlocs_EEG);
%% Simulate baseline noise
noiseSD = 1.5;
data = noiseSD * randn(nSub, nChan, nTime);

%% Group labels

group = [zeros(nControl,1); ones(nTreatment,1)]; % 0 = Control, 1 = Treatment

%% Define P300 effect
p300Latency = 300;
p300Width = 70;

controlAmp = 3.0;
treatmentAmp = 6.0;

controlP300 = controlAmp * exp(-(times - p300Latency).^2 / ...
              (2 * p300Width^2));

treatmentP300 = treatmentAmp * exp(-(times - p300Latency).^2 / ...
                (2 * p300Width^2));

controlP300 = controlP300(:);
treatmentP300 = treatmentP300(:);

%% Effect channels
effectChansLabl = {'Cz','CP1','CP2','Pz','P3','P4'};
%roi_labels = {chanlocs_roi.labels};
[tf, effectChans] = ismember(effectChansLabl, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChansLabl(~tf), ', '));
end
%% Inject P300 into both groups
weights = [0.6 0.8 0.8 1.0 0.75 0.75];
for s = 1:nControl

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s, ch, :));

        tmp = tmp + weights(ch_idx) *controlP300;

        data(s, ch, :) = reshape(tmp, 1, 1, nTime);

    end

end

for s = (nControl+1):nSub
    for ch = effectChans
        tmp = squeeze(data(s,ch,:));
        tmp = tmp + weights(ch_idx) * treatmentP300;
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end
%% ==========================================================
% Figure 1: ERP waveform
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlERP = squeeze(mean(data(group==0,channelToPlot,:),1));
treatmentERP = squeeze(mean(data(group==1,channelToPlot,:),1));

figure;

plot(times,controlERP,'LineWidth',2);
hold on;

plot(times,treatmentERP, 'r','LineWidth',2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');

title(sprintf('ERP waveform (Channel %d, Pz)',channelToPlot));

legend('Control','Treatment');

%xline(0,'--');
grid on;

%% ==========================================================
% Figure 2: Ground-truth channel ?? time effect map
%% ==========================================================

groupDiff = squeeze(mean(data(group==1,:,:),1) - ...
                    mean(data(group==0,:,:),1));

figure;

% Plot observed group difference
imagesc(times, 1:nChan, groupDiff);
axis xy;

% Axes formatting
xlim([-200 800]);
set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {chanlocs_EEG.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

% Labels and title
xlabel('Time (ms)');
ylabel('Channel');
title('Observed Group Difference');

% Color scale
colorbar;

%% ==========================================================
% Figure 3: Ground-truth injected effect
%% ==========================================================

truthDiff = zeros(nChan,nTime);

trueDifference = treatmentP300 - controlP300;

for ch_idx = 1:length(effectChans)
    ch = effectChans(ch_idx);
    truthDiff(ch,:) = weights(ch_idx) * trueDifference';
end

figure;

% Channel labels
tick_labels = {chanlocs_EEG.labels};

% Plot ground-truth effect
imagesc(times, 1:nChan, truthDiff);
axis xy;

% Axes formatting
xlim([-200 800]);
set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tick_labels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

% Labels and title
xlabel('Time (ms)');
ylabel('Channel');
title('Ground-Truth Simulated Effect');

% Color scale
colorbar;

%% ==========================================================
% Figure 4: Topography at peak latency (requires EEGLAB)
%% ==========================================================
% rotate the plot to get the right direction
chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end
% visualize the plot
if exist('readlocs','file')

    [~,peakIdx] = min(abs(times-300));

    topo = groupDiff(:,peakIdx);

    try

        figure;

        topoplot(topo,chanlocs_plot, 'electrodes', 'labels');

        colorbar;

        title('Group Difference at 300 ms');

    catch

        fprintf('Could not load channel locations.\n');

    end

end

%% ==========================================================
% Save dataset
%% ==========================================================

save('./data/simulated_between_subject_EEG.mat',...
     'data','group','times','effectChans');