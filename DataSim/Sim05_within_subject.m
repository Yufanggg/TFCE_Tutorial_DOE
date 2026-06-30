%% ==========================================================
% Simulated EEG/ERP dataset
% Within-subject experimental design
%
% Final saved variables:
%   EEGdata
%   designTable
%
% Final EEGdata dimensions:
%   Subject-condition rows x Channels x Time
%   60 x 32 x 251
%
% designTable variables:
%   Subject
%   CondCode    % -1 = Control, 1 = Treatment
%   CondName    % Control / Treatment
%% ==========================================================

clear; clc; close all; rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nSub  = 30;
nCond = 2;

condNames = {'Control','Treatment'};
condCodes = [-1, 1];

times = -200:4:800;
nTime = length(times);

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
% Define P300 effect
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

effectChansLabl = {'Cz','CP1','CP2','Pz','P3','P4'};

[tf, effectChans] = ismember(effectChansLabl, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChansLabl(~tf), ', '));
end

weights = [0.6 0.8 0.8 1.0 0.75 0.75];

%% ==========================================================
% Simulate within-subject EEG/ERP data
%% ==========================================================

% Temporary data size:
% Subjects x Conditions x Channels x Time
data4D = zeros(nSub, nCond, nChan, nTime);

subjectNoiseSD    = 1.5;
backgroundNoiseSD = 0.8;

for s = 1:nSub

    subjectNoise = subjectNoiseSD * randn(nChan, nTime);

    for c = 1:nCond

        backgroundNoise = backgroundNoiseSD * randn(nChan, nTime);

        data4D(s,c,:,:) = subjectNoise + backgroundNoise;

    end

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data4D(s,1,ch,:));
        tmp = tmp + weights(ch_idx) * controlP300;
        data4D(s,1,ch,:) = reshape(tmp, 1, 1, 1, nTime);

        tmp = squeeze(data4D(s,2,ch,:));
        tmp = tmp + weights(ch_idx) * treatmentP300;
        data4D(s,2,ch,:) = reshape(tmp, 1, 1, 1, nTime);

    end

end

%% ==========================================================
% Reshape to long format
%
% Final EEGdata:
%   Rows x Channels x Time
%   60 x 32 x 251
%
% designTable:
%   Subject
%   CondCode
%   CondName
%% ==========================================================

EEGdata = zeros(nSub * nCond, nChan, nTime);

Subject  = zeros(nSub * nCond, 1);
CondCode = zeros(nSub * nCond, 1);
CondName = cell(nSub * nCond, 1);

row = 0;

for s = 1:nSub
    for c = 1:nCond

        row = row + 1;

        EEGdata(row,:,:) = squeeze(data4D(s,c,:,:));

        Subject(row)  = s;
        CondCode(row) = condCodes(c);
        CondName{row} = condNames{c};

    end
end

designTable = table(Subject, CondCode, CondName, ...
    'VariableNames', {'Subject','CondCode','CondName'});

disp(designTable(1:10,:));

%% ==========================================================
% Within-subject difference for plotting
%% ==========================================================

subjectDiff = squeeze(data4D(:,2,:,:) - data4D(:,1,:,:));
conditionDiff = squeeze(mean(subjectDiff,1));

%% ==========================================================
% Figure 1: ERP waveform at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlERP   = squeeze(mean(data4D(:,1,channelToPlot,:),1));
treatmentERP = squeeze(mean(data4D(:,2,channelToPlot,:),1));

figure;

plot(times, controlERP, 'LineWidth', 2);
hold on;

plot(times, treatmentERP, 'r', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('Within-subject ERP waveform at Pz');
legend(condNames);
grid on;

%% ==========================================================
% Figure 2: Observed within-subject difference map
%% ==========================================================

figure;

imagesc(times, 1:nChan, conditionDiff);
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
title('Observed Within-subject Difference: Treatment - Control');

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
title('Ground-truth Within-subject Effect');

colorbar;

%% ==========================================================
% Figure 4: Topography at 300 ms
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

if exist('topoplot','file')

    [~,peakIdx] = min(abs(times - 300));

    topo = conditionDiff(:,peakIdx);

    figure;

    topoplot(topo, chanlocs_plot, 'electrodes', 'labels');

    colorbar;

    title('Within-subject Difference at 300 ms');

else

    fprintf('topoplot not found. Please check EEGLAB path.\n');

end

%% ==========================================================
% Save dataset
% Only EEGdata and designTable are saved
%% ==========================================================

if ~exist('../data','dir')
    mkdir('../data');
end

save('../data/05_simulated_within_subject_EEG.mat', ...
     'EEGdata', ...
     'designTable');

disp('Dataset saved: ../data/05_simulated_within_subject_EEG.mat');
disp('Saved variables: EEGdata, designTable');
disp('Final EEGdata size:');
disp(size(EEGdata));