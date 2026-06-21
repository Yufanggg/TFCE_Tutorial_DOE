%% ==========================================================
% Simulated EEG/ERP dataset
% Fully crossed random-effects design
%
% Random effects:
%   Subject
%   Item
%
% Fully crossed:
%   Every subject sees every item in every condition
%
% Data dimensions:
%   Subjects x Items x Conditions x Channels x Time
%
% Model idea:
%   EEG ~ Condition + (1|Subject) + (1|Item)
%   EEG ~ Condition + (Condition|Subject) + (Condition|Item)
%
% Ground truth:
%   Treatment has larger positive ERP effect at 300 ms
%   Effect channels: Cz, CP1, CP2, Pz, P3, P4
%% ==========================================================

clear; clc; close all; rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nSub  = 30;
nItem = 40;
nCond = 2;

condNames = {'Control','Treatment'};

subjectID = (1:nSub)';
itemID    = (1:nItem)';

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
% Simulate crossed random effects
%% ==========================================================

% Data size:
% Subjects x Items x Conditions x Channels x Time
data = zeros(nSub, nItem, nCond, nChan, nTime);

% Random-intercept variability
subjectOffsetSD = 1.5;
itemOffsetSD    = 1.0;

% Random P300-gain variability
subjectP300SD = 0.25;
itemP300SD    = 0.15;

% Noise variability
sharedNoiseSD    = 1.0;
conditionNoiseSD = 0.8;

% Subject random effects
subjectOffset = subjectOffsetSD * randn(nSub,1);
subjectP300Gain = 1 + subjectP300SD * randn(nSub,1);

% Item random effects
itemOffset = itemOffsetSD * randn(nItem,1);
itemP300Gain = 1 + itemP300SD * randn(nItem,1);

for s = 1:nSub

    for i = 1:nItem

        % Shared noise for this subject-item pair
        sharedNoise = sharedNoiseSD * randn(nChan,nTime);

        % Combined subject + item random intercept
        crossedOffset = subjectOffset(s) + itemOffset(i);

        % Combined subject + item P300 gain
        crossedP300Gain = subjectP300Gain(s) * itemP300Gain(i);

        for c = 1:nCond

            conditionNoise = conditionNoiseSD * randn(nChan,nTime);

            data(s,i,c,:,:) = sharedNoise + ...
                              conditionNoise + ...
                              crossedOffset;

        end

        % Inject P300 effect into selected channels
        for ch_idx = 1:length(effectChans)

            ch = effectChans(ch_idx);

            % Control
            tmp = squeeze(data(s,i,1,ch,:));
            tmp = tmp + crossedP300Gain * weights(ch_idx) * controlP300;
            data(s,i,1,ch,:) = reshape(tmp,1,1,1,1,nTime);

            % Treatment
            tmp = squeeze(data(s,i,2,ch,:));
            tmp = tmp + crossedP300Gain * weights(ch_idx) * treatmentP300;
            data(s,i,2,ch,:) = reshape(tmp,1,1,1,1,nTime);

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
conditionDiff = squeeze(mean(mean(subjectItemDiff,1),2));

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

%% ==========================================================
% Figure 4: Paired t-test map over subject-item observations
%% ==========================================================

tMap = zeros(nChan,nTime);
pMap = zeros(nChan,nTime);

for ch = 1:nChan

    for t = 1:nTime

        controlVals = squeeze(data(:,:,1,ch,t));
        treatmentVals = squeeze(data(:,:,2,ch,t));

        controlVals = controlVals(:);
        treatmentVals = treatmentVals(:);

        [~,p,~,stats] = ttest(treatmentVals, controlVals);

        tMap(ch,t) = stats.tstat;
        pMap(ch,t) = p;

    end
end

figure;

imagesc(times, 1:nChan, tMap);
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
title('Paired t-test Map: Treatment vs Control');

colorbar;

%% ==========================================================
% Figure 5: Significant paired t-test map
%% ==========================================================

alpha = 0.05;

sigMap = tMap;
sigMap(pMap >= alpha) = 0;

figure;

imagesc(times, 1:nChan, sigMap);
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
title('Significant Paired t-test Map, p < 0.05');

colorbar;

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

if ~exist('./data','dir')
    mkdir('./data');
end

save('./data/08_simulated_fully_crossed_subject_item_EEG.mat', ...
     'data', ...
     'subjectItemDiff', ...
     'conditionDiff', ...
     'tMap', ...
     'pMap', ...
     'times', ...
     'effectChans', ...
     'effectChansLabl', ...
     'chanlocs_EEG', ...
     'condNames', ...
     'subjectID', ...
     'itemID', ...
     'designTable', ...
     'subjectOffset', ...
     'itemOffset', ...
     'subjectP300Gain', ...
     'itemP300Gain');