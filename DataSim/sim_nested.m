%% ==========================================================
% Simulated EEG/ERP dataset
% Nested within-subject experimental design
%
% Subjects are nested inside classes.
% Same subjects measured in two conditions:
% Condition 1 = Control
% Condition 2 = Treatment
%
% Data dimensions:
% Subjects x Conditions x Channels x Time
%
% Ground truth:
% Treatment has larger positive ERP effect at 300 ms
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

% Class membership for each subject
classID = kron((1:nClass)', ones(nSubPerClass,1));

% Subject ID within each class: 1,2,...,10,1,2,...,10,...
subIDwithinClass = repmat((1:nSubPerClass)', nClass, 1);

% Global subject ID: 1,...,30
subjectID = (1:nSub)';

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

% Variability settings
classOffsetSD     = 1.0;   % class-level amplitude offset
classP300SD       = 0.15;  % class-level P300 variability

subjectOffsetSD   = 1.5;   % subject-level amplitude offset
subjectP300SD     = 0.25;  % subject-level ERP variability

sharedNoiseSD     = 1.0;   % shared noise across conditions within subject
conditionNoiseSD  = 0.8;   % condition-specific noise

% Class-level random effects
classOffset = classOffsetSD * abs(randn(nClass,1));
classP300Gain = 1 + classP300SD * abs(randn(nClass,1));

for s = 1:nSub

    thisClass = classID(s);

    % Subject-level baseline offset nested inside class
    subjectOffset = classOffset(thisClass) + ...
                    subjectOffsetSD * abs(randn);

    % Shared EEG noise pattern for this subject
    sharedNoise = sharedNoiseSD * abs(randn(nChan, nTime));

    % Subject-specific P300 gain nested inside class
    subjectP300Gain = classP300Gain(thisClass) + ...
                      subjectP300SD * abs(randn);

    for c = 1:nCond

        % Condition-specific noise
        conditionNoise = conditionNoiseSD * abs(randn(nChan, nTime));

        % Shared subject/class structure + condition-specific noise
        data(s,c,:,:) = sharedNoise + conditionNoise + subjectOffset;

    end

    % Inject P300 into Control and Treatment
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
% Design table
%% ==========================================================

designTable = table(subjectID, classID, subIDwithinClass, ...
    'VariableNames', {'SubjectID','ClassID','SubjectWithinClass'});

disp(designTable);

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
title('Observed Nested Within-subject Difference: Treatment - Control');

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
title('Ground-truth Nested Within-subject Effect');

colorbar;

%% ==========================================================
% Figure 4: Paired t-test map
%% ==========================================================

tMap = zeros(nChan,nTime);
pMap = zeros(nChan,nTime);

for ch = 1:nChan

    for t = 1:nTime

        controlVals   = squeeze(data(:,1,ch,t));
        treatmentVals = squeeze(data(:,2,ch,t));

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
     'nClass', ...
     'nSubPerClass', ...
     'classID', ...
     'subIDwithinClass', ...
     'subjectID', ...
     'designTable');