%%% between-subject design (2-by-2 design with interactions)
%% ==========================================================
% Simulated EEG/ERP dataset
% Between-subject design:
% Control (n=30) vs Treatment (n=30)
%
% Data dimensions:
% Subjects x Channels x Time
%
% Ground truth:
% Positive ERP effect at 400 ms
% Channels: 30,31,37,38
%% ==========================================================

clear; clc; close all; rng(123);

%% Simulation settings

nFactor00 = 30;
nFactor01 = 30;
nFactor10 = 30;
nFactor11 = 30;

nSub  = nFactor00 + nFactor01 + nFactor10 + nFactor11;
nChan = 32;

times = -200:4:800;
nTime = length(times);
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

%% Simulate baseline noise
noiseSD = 1.5;
data = noiseSD * randn(nSub, nChan, nTime);

%% Group labels

group = [zeros(nFactor00,1); ones(nFactor01,1); 2*ones(nFactor10,1); 3*ones(nFactor11,1)]; % 0 = Control, 1 = Treatment

%% Define P300 effect
p300Latency = 300;
p300Width = 70;

Factor00 = 3.0
Factor01 = 6.0
Factor10 = 4.0
Fatcor11 = 10.0

Factor00P300 = Factor00 * exp(-(times - p300Latency).^2 / ...
              (2 * p300Width^2));
Factor01P300 = Factor01 * exp(-(times - p300Latency).^2 / ...
                (2 * p300Width^2));
         

Factor10P300 = Factor10 * exp(-(times - p300Latency).^2 / ...
              (2 * p300Width^2));
Fatcor11P300 = Fatcor11 * exp(-(times - p300Latency).^2 / ...
                (2 * p300Width^2));           

Factor00P300 = Factor00P300(:);
Factor01P300 = Factor01P300(:);
Factor10P300 = Factor10P300(:);
Fatcor11P300 = Fatcor11P300(:);

%% Effect channels
effectChansLabl = {'Cz','CP1','CP2','Pz','P3','P4'};
[tf, effectChans] = ismember(effectChansLabl, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChansLabl(~tf), ', '));
end
%% Inject P300 into all groups
weights = [0.6 0.8 0.8 1.0 0.75 0.75];
for s = 1:nFactor00

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s, ch, :));

        tmp = tmp + weights(ch_idx) *Factor00P300;

        data(s,ch,:) = reshape(tmp,1,1,nTime);

    end

end

for s = (nFactor00+1):(nFactor00 + nFactor01)
    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s,ch,:));
        
        tmp = tmp + weights(ch_idx) *Factor01P300;
        
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end

for s = (nFactor00 + nFactor01 + 1):(nFactor00 + nFactor01 + nFactor10)
    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s,ch,:));
        
        tmp = tmp + weights(ch_idx) * Factor10P300;
        
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end

for s = (nFactor00 + nFactor01 + nFactor10 + 1):nSub
    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s,ch,:));
        
        tmp = tmp + weights(ch_idx) * Fatcor11P300;
        
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end
%% ==========================================================
% Figure 1: ERP waveform
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

Factor0_ERP = squeeze(mean(data((group == 0) | (group == 1),channelToPlot,:),1));
Factor1_ERP = squeeze(mean(data((group == 2) | (group == 3),channelToPlot,:),1));
Factor_0ERP = squeeze(mean(data((group == 0) | (group == 2),channelToPlot,:),1));
Factor_1ERP = squeeze(mean(data((group == 1) | (group == 3),channelToPlot,:),1));

figure;

plot(times,Factor0_ERP,'LineWidth',2);
hold on;
plot(times,Factor1_ERP, 'r','LineWidth',2);
hold on;
plot(times,Factor_0ERP, 'y','LineWidth',2);
hold on;
plot(times,Factor_1ERP, 'g','LineWidth',2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');

title(sprintf('ERP waveform (Channel %d)',channelToPlot));

legend('FactorA+','FactorA-', 'FactorC+','FactorC-');

%xline(0,'--');
grid on;

%% ==========================================================
% Figure 2: Ground-truth channel ¡Á time effect map
%% ==========================================================

groupADiff = squeeze(mean(data((group == 0) | (group == 1),:,:),1) - ...
                    mean(data((group == 2) | (group == 3),:,:),1));

figure;

% Plot observed group difference
imagesc(times, 1:nChan, groupADiff);
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
title('Observed A+ vs. A- Difference');

% Color scale
colorbar;


groupBDiff = squeeze(mean(data((group == 0) | (group == 2),:,:),1) - ...
                    mean(data((group == 1) | (group == 3),:,:),1));

figure;

% Plot observed group difference
imagesc(times, 1:nChan, groupBDiff);
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
title('Observed B+ vs. B- Difference');

% Color scale
colorbar;


%% ==========================================================
% Figure 3: Ground-truth injected effect
%% ==========================================================
tick_labels = {chanlocs_EEG.labels};

truthADiff = zeros(nChan,nTime);

trueADifference = Factor_0ERP - Factor_1ERP;

for ch_idx = 1:length(effectChans)
    ch = effectChans(ch_idx);
    truthADiff(ch,:) = weights(ch_idx)*trueADifference';

end

figure;

imagesc(times,1:nChan,truthADiff);
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
title('Ground-Truth Simulated Effect for A');

% Color scale
colorbar;


truthBDiff = zeros(nChan,nTime);

trueBDifference = Factor0_ERP - Factor1_ERP;

for ch = effectChans

    truthBDiff(ch,:) = trueBDifference';

end

figure;

imagesc(times,1:nChan,truthBDiff);

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
title('Ground-Truth Simulated Effect for B');

% Color scale
colorbar;

%% ==========================================================
% Figure 4: Topography at peak latency (requires EEGLAB)
%% ==========================================================
chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

% visualize the plot
if exist('readlocs','file')

    [~,peakIdx] = min(abs(times-300));

    topo = groupADiff(:,peakIdx);

    try
        figure;

        topoplot(topo,chanlocs_plot, 'electrodes', 'labels');

        colorbar;

        title('Group Difference at 300 ms');

    catch

        fprintf('Could not load channel locations.\n');

    end

end

if exist('readlocs','file')

    [~,peakIdx] = min(abs(times-300));

    topo = groupBDiff(:,peakIdx);

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
save('./data/02_simulated_between_subject_2by2Int_EEG.mat',...
     'data','group','times','effectChans');