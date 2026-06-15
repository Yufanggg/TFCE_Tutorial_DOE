%% ==========================================================
% Simulated EEG/ERP dataset
% Nested within-subject experimental design
%
% Subjects are nested inside classes.
% Each subject is measured in two conditions:
% Condition 1 = Control
% Condition 2 = Treatment
%
% Data dimensions:
% Subjects x Conditions x Channels x Time
%
% Model structure:
% Condition is within-subject.
% Subjects are nested in Class.
%
% Ground truth:
% Treatment has larger positive ERP effect at 300 ms.
% Effect channels: Cz, CP1, CP2, Pz, P3, P4
%% ==========================================================

clear; clc; close all; rng(123);

%% ==========================================================
% Simulation settings
%% ==========================================================

nClass = 3;
nSubPerClass = 10;
nSub = nClass * nSubPerClass;

nCond = 2;
condNames = {'Control','Treatment'};

% Older-MATLAB-compatible replacement for repelem
classID = kron((1:nClass)', ones(nSubPerClass,1));

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
% Simulate nested within-subject EEG/ERP data
%% ==========================================================

% Data size:
% Subjects x Conditions x Channels x Time
data = zeros(nSub, nCond, nChan, nTime);

% Noise and variability settings
sharedNoiseSD    = 1.0; %noised within the same subject
conditionNoiseSD = 0.8; %noise under each conition

% Class-level variability
classOffsetSD = 1.0; %how much classes differ in overall baseline amplitude
classP300SD   = 0.20;% how much classes differ in P300 size

% Subject-level variability
subjectOffsetSD = 1.5; % how much subjects differ in baseline amplitude within their clss
subjectP300SD   = 0.25; % how much subject differ in P300 size within their class

% Class-level random effects
classOffset = abs(classOffsetSD * randn(nClass,1)); %affects the overall baseline amplitude
classP300Gain = 1 + abs(classP300SD * randn(nClass,1)); %affects only the P300 component amplitude

for s = 1:nSub 

    cl = classID(s);

    % Subject-level baseline offset nested inside class
    subjectOffset = classOffset(cl) + abs(subjectOffsetSD * randn);

    % Shared EEG noise pattern for this subject
    sharedNoise = abs(sharedNoiseSD * randn(nChan, nTime));

    % Subject-specific P300 gain nested inside class
    subjectP300Gain = classP300Gain(cl) + abs(subjectP300SD * randn);

    for c = 1:nCond

        conditionNoise = abs(conditionNoiseSD * randn(nChan, nTime));

        data(s,c,:,:) = sharedNoise + conditionNoise + subjectOffset;

    end

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        % Control condition
        tmp = squeeze(data(s,1,ch,:));
        tmp = tmp + subjectP300Gain * weights(ch_idx) * controlP300;
        data(s,1,ch,:) = reshape(tmp, 1, 1, 1, nTime);

        % Treatment condition
        tmp = squeeze(data(s,2,ch,:));
        tmp = tmp + subjectP300Gain * weights(ch_idx) * treatmentP300;
        data(s,2,ch,:) = reshape(tmp, 1, 1, 1, nTime);

    end

end

%% ==========================================================
% Within-subject difference
%% ==========================================================

subjectDiff = squeeze(data(:,2,:,:) - data(:,1,:,:));
conditionDiff = squeeze(mean(subjectDiff,1));

%% ==========================================================
% Figure 1: ERP waveform at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

controlERP = squeeze(mean(data(:,1,channelToPlot,:),1));
treatmentERP = squeeze(mean(data(:,2,channelToPlot,:),1));

figure;

plot(times, controlERP, 'LineWidth', 2);
hold on;

plot(times, treatmentERP, 'r', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('Nested Within-subject ERP waveform at Pz');
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
title('Ground-truth Within-subject Effect');

colorbar;

%% ==========================================================
% Figure 4: Linear mixed model t-map
%
% Model:
% Y ~ Condition + (1|Class) + (1|Class:Subject)
%
% Condition is within-subject.
% Subject is nested inside Class.
%% ==========================================================

tMap = zeros(nChan,nTime);
pMap = zeros(nChan,nTime);

subjectID = (1:nSubPerClass)';

for ch = 1:nChan

    fprintf('Running LME for channel %d of %d: %s\n', ...
        ch, nChan, chanlocs_EEG(ch).labels);

    for t = 1:nTime

        controlVals   = squeeze(data(:,1,ch,t));
        treatmentVals = squeeze(data(:,2,ch,t));

        Y = [controlVals; treatmentVals];

        Condition = categorical([ ...
            repmat({'Control'}, nSubPerClass*nClass, 1); ...
            repmat({'Treatment'}, nSubPerClass*nClass, 1)]);

        Subject = categorical([subjectID; subjectID; subjectID; subjectID; subjectID; subjectID]);

        Class = categorical([classID; classID]);

        tbl = table(Y, Condition, Subject, Class);

        lme = fitlme(tbl, ...
            'Y ~ Condition + (1|Class) + (1|Class:Subject)');

        coefTable = lme.Coefficients;

        tMap(ch,t) = coefTable.tStat(2);
        pMap(ch,t) = coefTable.pValue(2);

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
title('Linear Mixed Model t-map: Treatment vs Control');

colorbar;

%% ==========================================================
% Figure 5: Significant LME map
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
title('Significant LME Map, p < 0.05');

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

    title('Nested Within-subject Difference at 300 ms');

else

    fprintf('topoplot not found. Please check EEGLAB path.\n');

end

%% ==========================================================
% Save dataset
%% ==========================================================

if ~exist('./data','dir')
    mkdir('./data');
end

save('./data/07_simulated_nested_within_subject_EEG.mat', ...
     'data', ...
     'subjectDiff', ...
     'conditionDiff', ...
     'tMap', ...
     'pMap', ...
     'times', ...
     'effectChans', ...
     'effectChansLabl', ...
     'chanlocs_EEG', ...
     'condNames', ...
     'classID', ...
     'nClass', ...
     'nSubPerClass');