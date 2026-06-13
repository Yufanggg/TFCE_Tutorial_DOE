

%% ==========================================================
% Simulate ERP data with group-specific P300 effects
%% ==========================================================
% Groups:
% Control (n = 30)
% Treatment (n = 30)
%
% Data dimensions:
% Subjects ¡Á Channels ¡Á Time
%
% Sampling:
% 32 EEG channels
% Epoch: -200 to 800 ms
% Sampling rate: 250 Hz
%
% Ground truth:
% Positive ERP component centered at 300 ms
% Effect present only in Treatment group
% Effect channels: 30, 31, 37, 38
% Peak amplitude increase: +3 ?V
%
% Noise:
% Gaussian trial-to-trial variability
% Subject-specific baseline differences
%
% Expected result:
% Significant Group difference around 300 ms
% Localized to channels 30, 31, 37, 38
%% ==========================================================

clear; clc; close all; rng(123);

%% Simulation settings
nControl   = 30;
nTreatment = 30;
nSub       = nControl + nTreatment;

times = -200:4:800;
nTime = length(times);

%% Load and select EEG channels
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

%% Simulate baseline noise
noiseSD = 1.5;
data = noiseSD * randn(nSub, nChan, nTime);

%% Group labels and covariate
group = [zeros(nControl,1); ones(nTreatment,1)];   % 0 = Control, 1 = Treatment
covariate = randn(nSub,1);

%% Define P300 parameters
p300Latency = 300;
p300Width   = 70;

controlAmp   = 3.0;
treatmentAmp = 6.0;
betaCov      = 3.0;

%% Generate base P300 waveform
p300Shape = exp(-(times - p300Latency).^2 ./ ...
                (2 * p300Width^2));   % 1 x nTime

%% Subject-specific amplitudes
controlAmplitude = controlAmp + ...
                   betaCov * covariate(1:nControl);

treatmentAmplitude = treatmentAmp + ...
                     betaCov * covariate(nControl+1:end);

%% Generate subject-level P300 waveforms
controlP300   = controlAmplitude * p300Shape;     % nControl x nTime
treatmentP300 = treatmentAmplitude * p300Shape;   % nTreatment x nTime

%% Define effect channels
effectChanLabels = {'Cz','CP1','CP2','Pz','P3','P4'};
[tf, effectChans] = ismember(effectChanLabels, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChanLabels(~tf), ', '));
end

%% Inject P300 into selected channels
weights = [0.6 0.8 0.8 1.0 0.75 0.75];

% Control group
for s = 1:nControl
    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s, ch, :));
        tmp = tmp + weights(ch_idx) * controlP300(s, :)';

        data(s, ch, :) = reshape(tmp, 1, 1, nTime);

    end
end

% Treatment group
for i = 1:nTreatment

    s = nControl + i;   % actual subject index in full data matrix

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s, ch, :));
        tmp = tmp + weights(ch_idx) * treatmentP300(i, :)';

        data(s, ch, :) = reshape(tmp, 1, 1, nTime);

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
% Figure 2: Observed channel ¡Á time effect maps
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
truthDiff = zeros(nChan, nTime);

% Average P300 waveform for each group
meanControlP300   = mean(controlP300, 1);      % 1 x nTime
meanTreatmentP300 = mean(treatmentP300, 1);    % 1 x nTime

% True group difference waveform
trueDifference = meanTreatmentP300 - meanControlP300;   % 1 x nTime

% Apply channel-specific weights
for ch_idx = 1:length(effectChans)

    ch = effectChans(ch_idx);

    truthDiff(ch, :) = weights(ch_idx) * trueDifference;

end

%% Plot ground-truth effect over channels and time
figure;

tickLabels = {chanlocs_EEG.labels};

imagesc(times, 1:nChan, truthDiff);
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
title('Ground-Truth Simulated Effect');

colorbar;


%% ==========================================================
% Topography of ground-truth effect at peak latency
% Requires EEGLAB topoplot
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

% Rotate layout for plotting orientation
for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

% Find index closest to P300 peak
[~, peakIdx] = min(abs(times - p300Latency));

% Topographic values at peak latency
topo = truthDiff(:, peakIdx);

if exist('topoplot', 'file')

    figure;

    topoplot(topo, chanlocs_plot, ...
        'electrodes', 'labels');

    colorbar;
    title(sprintf('Ground-Truth Effect at %d ms', p300Latency));

else

    warning('EEGLAB topoplot function not found. Topography was not plotted.');

end



%% ==========================================================
% Save dataset
%% ==========================================================

save('./data/04_simulated_between_subject_covariate_EEG.mat',...
     'data','group','times','effectChans');