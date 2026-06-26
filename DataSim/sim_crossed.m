%% ==========================================================
% Simulated EEG/ERP dataset
% Fully crossed within-subject and within-item design
%
% Every subject sees every item in every condition
%
% Data dimensions:
% Subjects x Items x Conditions x Channels x Time
%
% Noise sources:
% 1. Background noise
% 2. Subject-wise noise
% 3. Item-wise noise
%
% Model idea:
% EEG ~ Condition + (1|Subject) + (1|Item)
%% ==========================================================

clear; clc; close all;
rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nSub  = 30;
nItem = 40;
nCond = 2;

condNames = {'Control','Treatment'};

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

effectChanLabels = {'Cz','CP1','CP2','Pz','P3','P4'};

[tf, effectChans] = ismember(effectChanLabels, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChanLabels(~tf), ', '));
end

weights = [0.6 0.8 0.8 1.0 0.75 0.75];

%% ==========================================================
% Simulate fully crossed EEG/ERP data
%% ==========================================================

% Data dimensions:
% Subjects x Items x Conditions x Channels x Time
data = zeros(nSub, nItem, nCond, nChan, nTime);

% Noise settings
backgroundNoiseSD = 0.8;
subjectNoiseSD    = 1.5;
itemNoiseSD       = 1.0;

% Subject-wise noise:
% one stable pattern per subject, shared across items and conditions
subjectNoise = subjectNoiseSD * randn(nSub, nChan, nTime);

% Item-wise noise:
% one stable pattern per item, shared across subjects and conditions
itemNoise = itemNoiseSD * randn(nItem, nChan, nTime);

for s = 1:nSub

    for i = 1:nItem

        for c = 1:nCond

            % Background noise:
            % unique for each subject-item-condition observation
            backgroundNoise = backgroundNoiseSD * randn(nChan, nTime);

            % Combine exactly three noise sources
            data(s,i,c,:,:) = ...
                squeeze(subjectNoise(s,:,:)) + ...
                squeeze(itemNoise(i,:,:)) + ...
                backgroundNoise;

        end
    end
end

%% ==========================================================
% Inject deterministic P300 condition effect
%% ==========================================================

for s = 1:nSub

    for i = 1:nItem

        for ch_idx = 1:length(effectChans)

            ch = effectChans(ch_idx);

            % Control condition
            tmp = squeeze(data(s,i,1,ch,:));
            tmp = tmp + weights(ch_idx) * controlP300;
            data(s,i,1,ch,:) = reshape(tmp, 1, 1, 1, 1, nTime);

            % Treatment condition
            tmp = squeeze(data(s,i,2,ch,:));
            tmp = tmp + weights(ch_idx) * treatmentP300;
            data(s,i,2,ch,:) = reshape(tmp, 1, 1, 1, 1, nTime);

        end
    end
end

%% ==========================================================
% Design table
%% ==========================================================

Subject = [];
Item = [];
Condition = {};

for s = 1:nSub
    for i = 1:nItem
        for c = 1:nCond
            Subject(end+1,1) = s;
            Item(end+1,1) = i;
            Condition{end+1,1} = condNames{c};
        end
    end
end

designTable = table(Subject, Item, Condition);

disp(designTable(1:20,:));

%% ==========================================================
% Condition difference
%% ==========================================================

% Subjects x Items x Channels x Time
subjectItemDiff = squeeze(data(:,:,2,:,:) - data(:,:,1,:,:));

% Average over subjects and items
% Channels x Time
conditionDiff = squeeze(mean(mean(subjectItemDiff, 1), 2));

%% ==========================================================
% Figure 1: ERP waveform at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlERP = squeeze(mean(mean(data(:,:,1,channelToPlot,:),1),2));
treatmentERP = squeeze(mean(mean(data(:,:,2,channelToPlot,:),1),2));

figure;

plot(times, controlERP, 'LineWidth', 2);
hold on;

plot(times, treatmentERP, 'r', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('Fully Crossed Subject ¡Á Item ERP at Pz');
legend(condNames);
grid on;

%% ==========================================================
% Figure 2: Observed difference map
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
title('Observed Difference: Treatment - Control');

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
title('Ground-truth Fully Crossed Effect');

colorbar;

% %% ==========================================================
% % Figure 4: Paired t-test map over subject-item observations
% %% ==========================================================
% 
% tMap = zeros(nChan,nTime);
% pMap = zeros(nChan,nTime);
% 
% for ch = 1:nChan
% 
%     for t = 1:nTime
% 
%         controlVals = squeeze(data(:,:,1,ch,t));
%         treatmentVals = squeeze(data(:,:,2,ch,t));
% 
%         controlVals = controlVals(:);
%         treatmentVals = treatmentVals(:);
% 
%         [~,p,~,stats] = ttest(treatmentVals, controlVals);
% 
%         tMap(ch,t) = stats.tstat;
%         pMap(ch,t) = p;
% 
%     end
% end
% 
% figure;
% 
% imagesc(times, 1:nChan, tMap);
% axis xy;
% 
% xlim([-200 800]);
% 
% set(gca, ...
%     'YTick', 1:nChan, ...
%     'YTickLabel', {chanlocs_EEG.labels}, ...
%     'XTick', -200:200:800, ...
%     'TickLength', [0 0], ...
%     'FontSize', 15, ...
%     'FontName', 'Arial');
% 
% xlabel('Time (ms)');
% ylabel('Channel');
% title('Paired t-test Map: Treatment vs Control');
% 
% colorbar;
% 
% %% ==========================================================
% % Figure 5: Significant paired t-test map
% %% ==========================================================
% 
% alpha = 0.05;
% 
% sigMap = tMap;
% sigMap(pMap >= alpha) = 0;
% 
% figure;
% 
% imagesc(times, 1:nChan, sigMap);
% axis xy;
% 
% xlim([-200 800]);
% 
% set(gca, ...
%     'YTick', 1:nChan, ...
%     'YTickLabel', {chanlocs_EEG.labels}, ...
%     'XTick', -200:200:800, ...
%     'TickLength', [0 0], ...
%     'FontSize', 15, ...
%     'FontName', 'Arial');
% 
% xlabel('Time (ms)');
% ylabel('Channel');
% title('Significant Paired t-test Map, p < 0.05');
% 
% colorbar;

%% ==========================================================
% Figure 6: Topography at 300 ms
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

    title('Fully Crossed Difference at 300 ms');

else

    fprintf('topoplot not found. Please check EEGLAB path.\n');

end

%% ==========================================================
% Save dataset
%% ==========================================================

if ~exist('../data','dir')
    mkdir('../data');
end

save('../data/07_simulated_fully_crossed_subject_item_EEG.mat', ...
     'data', ...
     'subjectItemDiff', ...
     'conditionDiff', ...
     'times', ...
     'effectChans', ...
     'effectChanLabels', ...
     'chanlocs_EEG', ...
     'condNames', ...
     'designTable', ...
     'subjectNoise', ...
     'itemNoise', ...
     'backgroundNoiseSD', ...
     'subjectNoiseSD', ...
     'itemNoiseSD');