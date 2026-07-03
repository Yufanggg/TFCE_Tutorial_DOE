%% ==========================================================
% Simulated EEG/ERP dataset
% Fully crossed subject-item design
%
% Saved variables:
%
% EEGdata:
%   Subject, Item, Condition, ConditionCode, Channel, Time, Amplitude
%
% designTable:
%   Subject, Item, Condition, ConditionCode
%
% Condition coding:
%   Control   = -1
%   Treatment =  1
%
% Model idea:
%   EEG ~ ConditionCode + (1|Subject) + (1|Item)
%% ==========================================================

clear; clc; close all;
rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nSub  = 20;
nItem = 20;

condNames = {'Control','Treatment'};
conditionCodes = [-1, 1];
nCond = numel(condNames);

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
%
% Temporary data5D:
%   Subjects x Items x Conditions x Channels x Time
%% ==========================================================

data5D = zeros(nSub, nItem, nCond, nChan, nTime);

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

            backgroundNoise = backgroundNoiseSD * randn(nChan, nTime);

            data5D(s,i,c,:,:) = ...
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
            tmp = squeeze(data5D(s,i,1,ch,:));
            tmp = tmp + weights(ch_idx) * controlP300;
            data5D(s,i,1,ch,:) = reshape(tmp, 1, 1, 1, 1, nTime);

            % Treatment condition
            tmp = squeeze(data5D(s,i,2,ch,:));
            tmp = tmp + weights(ch_idx) * treatmentP300;
            data5D(s,i,2,ch,:) = reshape(tmp, 1, 1, 1, 1, nTime);

        end
    end
end

%% ==========================================================
% Condition difference
%% ==========================================================

subjectItemDiff = squeeze(data5D(:,:,2,:,:) - data5D(:,:,1,:,:));

conditionDiff = squeeze(mean(mean(subjectItemDiff, 1), 2));

%% ==========================================================
% Figure 1: ERP waveform at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlERP = squeeze(mean(mean(data5D(:,:,1,channelToPlot,:),1),2));
treatmentERP = squeeze(mean(mean(data5D(:,:,2,channelToPlot,:),1),2));

figure;

plot(times, controlERP, 'LineWidth', 2);
hold on;

plot(times, treatmentERP, 'r', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('Fully Crossed Subject × Item ERP at Pz');
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

    title('Fully Crossed Difference at 300 ms');

else

    fprintf('topoplot not found. Please check EEGLAB path.\n');

end

%% ==========================================================
% Convert to long-format EEGdata
%
% Final EEGdata columns:
%   Subject
%   Item
%   Condition
%   ConditionCode
%   Channel
%   Time
%   Amplitude
%% ==========================================================

nRowsEEG = nSub * nItem * nCond * nChan * nTime;

Subject       = zeros(nRowsEEG, 1);
Item          = zeros(nRowsEEG, 1);
Condition     = cell(nRowsEEG, 1);
ConditionCode = zeros(nRowsEEG, 1);
Channel       = cell(nRowsEEG, 1);
Time          = zeros(nRowsEEG, 1);
Amplitude     = zeros(nRowsEEG, 1);

row = 1;

for s = 1:nSub
    for i = 1:nItem
        for c = 1:nCond
            for ch = 1:nChan
                for t = 1:nTime

                    Subject(row)       = s;
                    Item(row)          = i;
                    Condition{row}     = condNames{c};
                    ConditionCode(row) = conditionCodes(c);
                    Channel{row}       = chanlocs_EEG(ch).labels;
                    Time(row)          = times(t);
                    Amplitude(row)     = data5D(s,i,c,ch,t);

                    row = row + 1;

                end
            end
        end
    end
end

EEGdata = table( ...
    Subject, ...
    Item, ...
    Condition, ...
    ConditionCode, ...
    Channel, ...
    Time, ...
    Amplitude);

%% ==========================================================
% Create design table
%
% Final designTable columns:
%   Subject
%   Item
%   Condition
%   ConditionCode
%% ==========================================================

nRowsDesign = nSub * nItem * nCond;

Subject       = zeros(nRowsDesign, 1);
Item          = zeros(nRowsDesign, 1);
Condition     = cell(nRowsDesign, 1);
ConditionCode = zeros(nRowsDesign, 1);

row = 1;

for s = 1:nSub
    for i = 1:nItem
        for c = 1:nCond

            Subject(row)       = s;
            Item(row)          = i;
            Condition{row}     = condNames{c};
            ConditionCode(row) = conditionCodes(c);

            row = row + 1;

        end
    end
end

designTable = table( ...
    Subject, ...
    Item, ...
    Condition, ...
    ConditionCode);

%% ==========================================================
% Preview
%% ==========================================================

disp(designTable(1:20,:));
disp(EEGdata(1:20,:));

%% ==========================================================
% Save dataset
% Only EEGdata and designTable are saved
%% ==========================================================

if ~exist('../data','dir')
    mkdir('../data');
end

save('../data/08_simulated_fully_crossed_subject_item_EEG.mat', ...
     'EEGdata', ...
     'designTable');

disp('Dataset saved: ../data/08_simulated_fully_crossed_subject_item_EEG.mat');
disp('Saved variables: EEGdata, designTable');
disp('Final EEGdata size:');
disp(size(EEGdata));
disp('Final designTable size:');
disp(size(designTable));