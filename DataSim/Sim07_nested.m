%% ==========================================================
% Simulated EEG/ERP dataset
% Nested within-student design
%
% Saved variables:
%
% EEGdata:
%   Class, Student, Condition, ConditionCode, Channel, Time, Amplitude
%
% designTable:
%   Class, Student, Condition, ConditionCode
%
% Condition coding:
%   Control   = -1
%   Treatment =  1
%% ==========================================================

clear; clc; close all;
rng(123);

%% Simulation settings

nClass = 6;
nStudentPerClass = 10;
nSub = nClass * nStudentPerClass;

condNames = {'Control','Treatment'};
conditionCodes = [-1 1];
nCond = numel(condNames);

times = -200:4:800;
nTime = numel(times);

%% Subject IDs

classID   = kron((1:nClass)', ones(nStudentPerClass,1));
studentID = repmat((1:nStudentPerClass)', nClass, 1);

%% Load channel locations

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

%% P300 signal

p300Latency = 300;
p300Width   = 70;

controlAmp   = 3;
treatmentAmp = 6;

controlP300 = controlAmp .* ...
    exp(-(times - p300Latency).^2 ./ (2 * p300Width^2));

treatmentP300 = treatmentAmp .* ...
    exp(-(times - p300Latency).^2 ./ (2 * p300Width^2));

controlP300   = controlP300(:);
treatmentP300 = treatmentP300(:);

%% Effect channels

effectChanLabels = {'Cz','CP1','CP2','Pz','P3','P4'};

[tf, effectChans] = ismember(effectChanLabels, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChanLabels(~tf), ', '));
end

weights = [0.6 0.8 0.8 1.0 0.75 0.75];

%% Simulate EEG data as 4D array
% Dimensions:
% Subjects ¡Á Conditions ¡Á Channels ¡Á Time

EEGarray = zeros(nSub, nCond, nChan, nTime);

classNoiseSD      = 1.2;
studentNoiseSD    = 1.5;
backgroundNoiseSD = 0.8;

classNoise   = classNoiseSD   .* randn(nClass, nChan, nTime);
studentNoise = studentNoiseSD .* randn(nSub, nChan, nTime);

for s = 1:nSub

    c = classID(s);

    for cond = 1:nCond

        backgroundNoise = backgroundNoiseSD .* randn(nChan, nTime);

        EEGarray(s,cond,:,:) = ...
            squeeze(classNoise(c,:,:)) + ...
            squeeze(studentNoise(s,:,:)) + ...
            backgroundNoise;

    end

end

%% Inject P300 signal

for s = 1:nSub

    for k = 1:length(effectChans)

        ch = effectChans(k);

        tmp = squeeze(EEGarray(s,1,ch,:));
        tmp = tmp + weights(k) .* controlP300;
        EEGarray(s,1,ch,:) = reshape(tmp, 1, 1, 1, nTime);

        tmp = squeeze(EEGarray(s,2,ch,:));
        tmp = tmp + weights(k) .* treatmentP300;
        EEGarray(s,2,ch,:) = reshape(tmp, 1, 1, 1, nTime);

    end

end

%% ==========================================================
% Plotting section
%% ==========================================================

%% Figure 1: ERP waveform at Pz

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlERP = squeeze(mean(EEGarray(:,1,channelToPlot,:), 1));
treatmentERP = squeeze(mean(EEGarray(:,2,channelToPlot,:), 1));

figure;

plot(times, controlERP, 'LineWidth', 2);
hold on;
plot(times, treatmentERP, 'r', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('Nested within-student ERP waveform at Pz');
legend(condNames);
grid on;

%% Figure 2: Observed condition difference

subjectDiff = squeeze(EEGarray(:,2,:,:) - EEGarray(:,1,:,:));
conditionDiff = squeeze(mean(subjectDiff, 1));

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

%% Figure 3: Ground-truth condition effect

truthDiff = zeros(nChan, nTime);

trueDifference = treatmentP300 - controlP300;

for k = 1:length(effectChans)

    ch = effectChans(k);
    truthDiff(ch,:) = weights(k) * trueDifference';

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

%% Figure 4: Topography at 300 ms

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

%% Convert to long-format EEG table
% Columns:
% Class, Student, Condition, ConditionCode, Channel, Time, Amplitude

nRowsEEG = nSub * nCond * nChan * nTime;

Class         = zeros(nRowsEEG,1);
Student       = zeros(nRowsEEG,1);
Condition     = cell(nRowsEEG,1);
ConditionCode = zeros(nRowsEEG,1);
Channel       = cell(nRowsEEG,1);
Time          = zeros(nRowsEEG,1);
Amplitude     = zeros(nRowsEEG,1);

row = 1;

for s = 1:nSub
    for cond = 1:nCond
        for ch = 1:nChan
            for t = 1:nTime

                Class(row)         = classID(s);
                Student(row)       = studentID(s);
                Condition{row}     = condNames{cond};
                ConditionCode(row) = conditionCodes(cond);
                Channel{row}       = chanlocs_EEG(ch).labels;
                Time(row)          = times(t);
                Amplitude(row)     = EEGarray(s,cond,ch,t);

                row = row + 1;

            end
        end
    end
end

EEGdata = table( ...
    Class, ...
    Student, ...
    Condition, ...
    ConditionCode, ...
    Channel, ...
    Time, ...
    Amplitude);

%% Create design table
% Columns:
% Class, Student, Condition, ConditionCode

nRowsDesign = nSub * nCond;

Class         = zeros(nRowsDesign,1);
Student       = zeros(nRowsDesign,1);
Condition     = cell(nRowsDesign,1);
ConditionCode = zeros(nRowsDesign,1);

row = 1;

for s = 1:nSub
    for cond = 1:nCond

        Class(row)         = classID(s);
        Student(row)       = studentID(s);
        Condition{row}     = condNames{cond};
        ConditionCode(row) = conditionCodes(cond);

        row = row + 1;

    end
end

designTable = table( ...
    Class, ...
    Student, ...
    Condition, ...
    ConditionCode);

%% Preview

disp(designTable(1:10,:));
disp(EEGdata(1:10,:));

%% Save dataset

if ~exist('../data','dir')
    mkdir('../data');
end

save('../data/07_simulated_nested_class_student_EEG.mat', ...
     'EEGdata', ...
     'designTable');

disp('Dataset saved: ../data/07_simulated_nested_class_student_EEG.mat');
disp('Saved variables: EEGdata, designTable');