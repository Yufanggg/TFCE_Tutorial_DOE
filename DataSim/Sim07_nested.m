%% ==========================================================
% Simulated EEG/ERP dataset
% Nested within-student design
%
% Model:
% Y ~ Condition + (1|Class) + (1|Class:Student)
%
% Noise sources:
% 1. Class-wise noise
% 2. Student-wise noise nested within class
% 3. Background noise
%
% Data dimensions:
% Subjects x Conditions x Channels x Time
%% ==========================================================

clear; clc; close all;
rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nClass = 6;
nStudentPerClass = 10;
nSub = nClass * nStudentPerClass;

nCond = 2;
condNames = {'Control', 'Treatment'};

times = -200:4:800;
nTime = length(times);

%% ==========================================================
% Subject and class IDs
%% ==========================================================

classID = kron((1:nClass)', ones(nStudentPerClass,1));
studentID = (1:nSub)';

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
% Define P300 condition effect
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
% Simulate nested EEG/ERP data
%% ==========================================================

% Data dimensions:
% Subjects x Conditions x Channels x Time
data = zeros(nSub, nCond, nChan, nTime);

% Noise settings
classNoiseSD      = 1.2;
studentNoiseSD    = 1.5;
backgroundNoiseSD = 0.8;

% Class-wise noise:
% one stable pattern per class, shared by all students in that class
classNoise = classNoiseSD * randn(nClass, nChan, nTime);

% Student-wise noise:
% one stable pattern per student, nested within class
studentNoise = studentNoiseSD * randn(nSub, nChan, nTime);

for s = 1:nSub

    cID = classID(s);

    for cond = 1:nCond

        % Background noise:
        % unique for each subject-condition observation
        backgroundNoise = backgroundNoiseSD * randn(nChan, nTime);

        data(s,cond,:,:) = ...
            squeeze(classNoise(cID,:,:)) + ...
            squeeze(studentNoise(s,:,:)) + ...
            backgroundNoise;

    end
end

%% ==========================================================
% Inject deterministic condition-specific P300 signal
%% ==========================================================

for s = 1:nSub

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        % Control
        tmp = squeeze(data(s,1,ch,:));
        tmp = tmp + weights(ch_idx) * controlP300;
        data(s,1,ch,:) = reshape(tmp, 1, 1, 1, nTime);

        % Treatment
        tmp = squeeze(data(s,2,ch,:));
        tmp = tmp + weights(ch_idx) * treatmentP300;
        data(s,2,ch,:) = reshape(tmp, 1, 1, 1, nTime);

    end
end

%% ==========================================================
% Design table
%% ==========================================================

Subject = [];
Class = [];
Condition = {};

for s = 1:nSub
    for cond = 1:nCond

        Subject(end+1,1) = studentID(s);
        Class(end+1,1) = classID(s);
        Condition{end+1,1} = condNames{cond};

    end
end

designTable = table(Subject, Class, Condition);

disp(designTable(1:20,:));

%% ==========================================================
% Compute condition difference
%% ==========================================================

% Subjects x Channels x Time
subjectDiff = squeeze(data(:,2,:,:) - data(:,1,:,:));

% Channels x Time
conditionDiff = squeeze(mean(subjectDiff, 1));

%% ==========================================================
% Figure 1: ERP waveform at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlERP = squeeze(mean(data(:,1,channelToPlot,:), 1));
treatmentERP = squeeze(mean(data(:,2,channelToPlot,:), 1));

figure;

plot(times, controlERP, 'LineWidth', 2);
hold on;

plot(times, treatmentERP, 'r', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('Nested within-student ERP waveform at Pz');
legend(condNames);
grid on;

%% ==========================================================
% Figure 2: Observed condition difference
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
title('Observed Condition Difference: Treatment - Control');
colorbar;

%% ==========================================================
% Figure 3: Ground-truth condition effect
%% ==========================================================

truthDiff = zeros(nChan, nTime);

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
title('Ground-Truth Condition Effect');
colorbar;

%% ==========================================================
% Figure 4: Topography at 300 ms
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~, peakIdx] = min(abs(times - p300Latency));

if exist('topoplot', 'file')

    figure;
    topoplot(conditionDiff(:,peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed Treatment - Control Difference at 300 ms');

else

    warning('topoplot not found. Please add EEGLAB to your MATLAB path.');

end

%% ==========================================================
% Save simulated dataset
%% ==========================================================

if ~exist('../data', 'dir')
    mkdir('../data');
end

save('../data/07_simulated_nested_class_student_EEG.mat', ...
     'data', ...
     'subjectDiff', ...
     'conditionDiff', ...
     'times', ...
     'effectChans', ...
     'effectChanLabels', ...
     'chanlocs_EEG', ...
     'condNames', ...
     'studentID', ...
     'classID', ...
     'designTable', ...
     'classNoise', ...
     'studentNoise', ...
     'classNoiseSD', ...
     'studentNoiseSD', ...
     'backgroundNoiseSD');