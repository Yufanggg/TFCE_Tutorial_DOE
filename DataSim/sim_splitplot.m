%% ==========================================================
% Simulated EEG/ERP dataset
% Split-plot design
%
% Between-subject factor:
%   Group: G1 vs G2
%
% Within-subject factor:
%   Condition: Control vs Treatment
%
% Effects injected:
%   1. Group effect: G2 - G1
%   2. Treatment effect: Treatment - Control
%   3. Interaction: G2(T-C) - G1(T-C)
%
% Data dimensions:
%   Subjects x Conditions x Channels x Time
%% ==========================================================

clear; clc; close all; rng(123);

%% Simulation settings

nGroup = 2;
nSubPerGroup = 15;
nSub = nGroup * nSubPerGroup;

nCond = 2;
condNames = {'Control','Treatment'};
groupNames = {'G1','G2'};

groupID = kron((1:nGroup)', ones(nSubPerGroup,1));
subIDwithinGroup = repmat((1:nSubPerGroup)', nGroup, 1);
subjectID = (1:nSub)';

times = -200:4:800;
nTime = length(times);

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

%% Define P300 waveform with group, treatment, and interaction effects

p300Latency = 300;
p300Width   = 70;

baseAmp = 3.0;

groupEffectAmp     = 1.0;   % G2 > G1
treatmentEffectAmp = 2.0;   % Treatment > Control
interactionAmp     = 1.5;   % Treatment effect stronger in G2

amp_G1_Control   = baseAmp;
amp_G1_Treatment = baseAmp + treatmentEffectAmp;

amp_G2_Control   = baseAmp + groupEffectAmp;
amp_G2_Treatment = baseAmp + groupEffectAmp + treatmentEffectAmp + interactionAmp;

p300_G1_Control = amp_G1_Control * exp(-(times - p300Latency).^2 ./ ...
                 (2 * p300Width^2));

p300_G1_Treatment = amp_G1_Treatment * exp(-(times - p300Latency).^2 ./ ...
                   (2 * p300Width^2));

p300_G2_Control = amp_G2_Control * exp(-(times - p300Latency).^2 ./ ...
                 (2 * p300Width^2));

p300_G2_Treatment = amp_G2_Treatment * exp(-(times - p300Latency).^2 ./ ...
                   (2 * p300Width^2));

p300_G1_Control   = p300_G1_Control(:);
p300_G1_Treatment = p300_G1_Treatment(:);
p300_G2_Control   = p300_G2_Control(:);
p300_G2_Treatment = p300_G2_Treatment(:);

%% Effect channels

effectChansLabl = {'Cz','CP1','CP2','Pz','P3','P4'};

[tf, effectChans] = ismember(effectChansLabl, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChansLabl(~tf), ', '));
end

weights = [0.6 0.8 0.8 1.0 0.75 0.75];

%% Simulate EEG data

data = zeros(nSub, nCond, nChan, nTime);

groupOffsetSD    = 1.0;
subjectOffsetSD  = 1.5;
subjectP300SD    = 0.25;

sharedNoiseSD    = 1.0;
conditionNoiseSD = 0.8;

groupOffset = groupOffsetSD * randn(nGroup,1);

for s = 1:nSub

    thisGroup = groupID(s);

    subjectOffset = groupOffset(thisGroup) + subjectOffsetSD * randn;

    sharedNoise = sharedNoiseSD * randn(nChan,nTime);

    subjectP300Gain = 1 + subjectP300SD * randn;

    for c = 1:nCond

        conditionNoise = conditionNoiseSD * randn(nChan,nTime);

        data(s,c,:,:) = sharedNoise + conditionNoise + subjectOffset;

    end

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        if thisGroup == 1
            p300_control   = p300_G1_Control;
            p300_treatment = p300_G1_Treatment;
        else
            p300_control   = p300_G2_Control;
            p300_treatment = p300_G2_Treatment;
        end

        tmp = squeeze(data(s,1,ch,:));
        tmp = tmp + subjectP300Gain * weights(ch_idx) * p300_control;
        data(s,1,ch,:) = reshape(tmp,1,1,1,nTime);

        tmp = squeeze(data(s,2,ch,:));
        tmp = tmp + subjectP300Gain * weights(ch_idx) * p300_treatment;
        data(s,2,ch,:) = reshape(tmp,1,1,1,nTime);

    end
end

%% Design table

designTable = table(subjectID, groupID, subIDwithinGroup, ...
    'VariableNames', {'SubjectID','GroupID','SubjectWithinGroup'});

disp(designTable);

%% Compute effects

subjectDiff = squeeze(data(:,2,:,:) - data(:,1,:,:));

conditionDiff = squeeze(mean(subjectDiff,1));

groupMean = zeros(nGroup,nCond,nChan,nTime);

for g = 1:nGroup
    for c = 1:nCond
        groupMean(g,c,:,:) = squeeze(mean(data(groupID==g,c,:,:),1));
    end
end

groupDiff = squeeze(mean(groupMean(2,:,:,:),2) - mean(groupMean(1,:,:,:),2));

groupConditionDiff = zeros(nGroup,nChan,nTime);

for g = 1:nGroup
    groupConditionDiff(g,:,:) = squeeze(mean(subjectDiff(groupID==g,:,:),1));
end

interactionDiff = squeeze(groupConditionDiff(2,:,:) - groupConditionDiff(1,:,:));

%% Figure 1: ERP waveform at Pz

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

figure; hold on;

% ∂®“Â—’…´
col_G1 = [0 0.4470 0.7410];      % ¿∂
col_G2 = [0.8500 0.3250 0.0980]; % ≥»

for g = 1:nGroup

    groupSubjects = groupID == g;

    controlERP = squeeze(mean(data(groupSubjects,1,channelToPlot,:),1));
    treatmentERP = squeeze(mean(data(groupSubjects,2,channelToPlot,:),1));

    if g == 1
        plot(times, controlERP, '-',  'Color', col_G1, 'LineWidth', 2);
        plot(times, treatmentERP, '--', 'Color', col_G1, 'LineWidth', 2);
    elseif g == 2
        plot(times, controlERP, '-',  'Color', col_G2, 'LineWidth', 2);
        plot(times, treatmentERP, '--', 'Color', col_G2, 'LineWidth', 2);
    end

end

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('Split-plot ERP waveform at Pz');
legend({'G1 Control','G1 Treatment','G2 Control','G2 Treatment'});
grid on;

%% Figure 2: Group effect map, G2 - G1

figure;
imagesc(times, 1:nChan, groupDiff);
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
title('Group Effect: G2 - G1');
colorbar;

%% Figure 3: Treatment effect map, T - C

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
title('Treatment Effect: Treatment - Control');
colorbar;

%% Figure 4: Interaction map

figure;
imagesc(times, 1:nChan, interactionDiff);
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
title('Interaction: G2(T-C) - G1(T-C)');
colorbar;

%% Figure 5: t-test for Group effect

tMap_group = zeros(nChan,nTime);
pMap_group = zeros(nChan,nTime);

for ch = 1:nChan
    for t = 1:nTime

        valsG1 = squeeze(mean(data(groupID==1,:,ch,t),2));
        valsG2 = squeeze(mean(data(groupID==2,:,ch,t),2));

        [~,p,~,stats] = ttest2(valsG2, valsG1);

        tMap_group(ch,t) = stats.tstat;
        pMap_group(ch,t) = p;

    end
end

figure;
imagesc(times, 1:nChan, tMap_group);
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
title('t-map: Group Effect G2 - G1');
colorbar;

%% Figure 6: t-test for Treatment effect

tMap_treatment = zeros(nChan,nTime);
pMap_treatment = zeros(nChan,nTime);

for ch = 1:nChan
    for t = 1:nTime

        controlVals = squeeze(data(:,1,ch,t));
        treatmentVals = squeeze(data(:,2,ch,t));

        [~,p,~,stats] = ttest(treatmentVals, controlVals);

        tMap_treatment(ch,t) = stats.tstat;
        pMap_treatment(ch,t) = p;

    end
end

figure;
imagesc(times, 1:nChan, tMap_treatment);
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
title('t-map: Treatment Effect T - C');
colorbar;

%% Figure 7: t-test for Interaction

tMap_interaction = zeros(nChan,nTime);
pMap_interaction = zeros(nChan,nTime);

for ch = 1:nChan
    for t = 1:nTime

        diffG1 = squeeze(subjectDiff(groupID==1,ch,t));
        diffG2 = squeeze(subjectDiff(groupID==2,ch,t));

        [~,p,~,stats] = ttest2(diffG2, diffG1);

        tMap_interaction(ch,t) = stats.tstat;
        pMap_interaction(ch,t) = p;

    end
end

figure;
imagesc(times, 1:nChan, tMap_interaction);
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
title('t-map: Interaction G2(T-C) - G1(T-C)');
colorbar;

%% Figure 8: Topographies at 300 ms

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

if exist('topoplot','file')

    [~,peakIdx] = min(abs(times - 300));

    topo_GroupEffect     = groupDiff(:,peakIdx);
    topo_TreatmentEffect = conditionDiff(:,peakIdx);
    topo_Interaction     = interactionDiff(:,peakIdx);

    figure;

    subplot(1,3,1);
    topoplot(topo_GroupEffect, chanlocs_plot, 'electrodes', 'labels');
    title('Group: G2 - G1');
    colorbar;

    subplot(1,3,2);
    topoplot(topo_TreatmentEffect, chanlocs_plot, 'electrodes', 'labels');
    title('Treatment: T - C');
    colorbar;

    subplot(1,3,3);
    topoplot(topo_Interaction, chanlocs_plot, 'electrodes', 'labels');
    title('Interaction');
    colorbar;

    sgtitle('Split-plot Topographies at 300 ms');

else

    fprintf('topoplot not found. Please check EEGLAB path.\n');

end

%% Save dataset

if ~exist('./data','dir')
    mkdir('./data');
end

save('./data/07_simulated_split_plot_group_treatment_interaction_EEG.mat', ...
     'data', ...
     'subjectDiff', ...
     'conditionDiff', ...
     'groupDiff', ...
     'groupConditionDiff', ...
     'interactionDiff', ...
     'tMap_group', ...
     'pMap_group', ...
     'tMap_treatment', ...
     'pMap_treatment', ...
     'tMap_interaction', ...
     'pMap_interaction', ...
     'times', ...
     'effectChans', ...
     'effectChansLabl', ...
     'chanlocs_EEG', ...
     'condNames', ...
     'groupNames', ...
     'subjectID', ...
     'groupID', ...
     'subIDwithinGroup', ...
     'designTable');