%% ==========================================================
% Simulated EEG/ERP dataset
% Split-plot mixed design
%
% Between-subject factor:
%   Group: -1 = HC, 1 = Patient
%
% Within-subject factor:
%   Condition: -1 = Noun, 1 = Verb
%
% Model:
%   EEG ~ Group * Condition + (1 | Subject)
%
% Noise sources:
%   1. Subject-wise noise
%   2. Background noise
%
% Data dimensions:
%   Subjects x Conditions x Channels x Time
%% ==========================================================

clear; clc; close all;
rng(123);

%% ==========================================================
% 1. Simulation settings
%% ==========================================================

nHC = 30;
nPatient = 30;
nSub = nHC + nPatient;

nCond = 2;

condNames = {'Noun','Verb'};
groupNames = {'HC','Patient'};

times = -200:4:800;
nTime = length(times);

subjectID = (1:nSub)';

% Group coding:
% -1 = HC, 1 = Patient
group = [-ones(nHC,1); ones(nPatient,1)];


%% ==========================================================
% 2. Load and select EEG channels
%% ==========================================================

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


%% ==========================================================
% 3. Define ERP effect
%% ==========================================================

p300Latency = 300;
p300Width   = 70;

p300Shape = exp(-(times - p300Latency).^2 ./ ...
                (2 * p300Width^2));
p300Shape = p300Shape(:);

% Cell amplitudes
% Rows:
%   1 = HC
%   2 = Patient
%
% Columns:
%   1 = Noun
%   2 = Verb

amp_HC_Noun      = 3.0;
amp_HC_Verb      = 6.0;

amp_Patient_Noun = 2.0;
amp_Patient_Verb = 3.5;

cellAmps = [
    amp_HC_Noun,      amp_HC_Verb;
    amp_Patient_Noun, amp_Patient_Verb
];


%% ==========================================================
% 4. Define effect channels
%% ==========================================================

effectChanLabels = {'Cz','CP1','CP2','Pz','P3','P4'};

[tf, effectChans] = ismember(effectChanLabels, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChanLabels(~tf), ', '));
end

weights = [0.6 0.8 0.8 1.0 0.75 0.75];


%% ==========================================================
% 5. Simulate EEG data
%% ==========================================================

% Data:
% Subjects x Conditions x Channels x Time
data = zeros(nSub, nCond, nChan, nTime);

subjectNoiseSD    = 1.5;
backgroundNoiseSD = 0.8;

% Subject-wise noise:
% one stable EEG pattern per subject,
% shared across noun and verb
subjectNoise = subjectNoiseSD * randn(nSub, nChan, nTime);

for s = 1:nSub

    % Convert group code to row index:
    % -1 -> 1, 1 -> 2
    groupIndex = (group(s) + 3) / 2;

    for c = 1:nCond

        % Background noise:
        % unique for each subject x condition observation
        backgroundNoise = backgroundNoiseSD * randn(nChan, nTime);

        data(s,c,:,:) = squeeze(subjectNoise(s,:,:)) + backgroundNoise;

        % Add deterministic ERP signal
        amp = cellAmps(groupIndex, c);
        erpWave = amp * p300Shape;

        for ch_idx = 1:length(effectChans)

            ch = effectChans(ch_idx);

            tmp = squeeze(data(s,c,ch,:));
            tmp = tmp + weights(ch_idx) * erpWave;

            data(s,c,ch,:) = reshape(tmp, 1, 1, 1, nTime);

        end
    end
end


%% ==========================================================
% 6. Create design table
%% ==========================================================

Subject = [];
Group = {};
GroupCode = [];
Condition = {};
ConditionCode = [];

for s = 1:nSub
    for c = 1:nCond

        Subject(end+1,1) = subjectID(s);

        GroupCode(end+1,1) = group(s);

        if group(s) == -1
            Group{end+1,1} = 'HC';
        else
            Group{end+1,1} = 'Patient';
        end

        % Condition coding:
        % -1 = Noun, 1 = Verb
        ConditionCode(end+1,1) = 2*c - 3;

        Condition{end+1,1} = condNames{c};

    end
end

designTable = table(Subject, Group, GroupCode, ...
                    Condition, ConditionCode);

disp(designTable(1:20,:));


%% ==========================================================
% 7. Compute grand average ERPs
%% ==========================================================

HC_Noun = squeeze(mean(data(group == -1, 1, :, :), 1));
HC_Verb = squeeze(mean(data(group == -1, 2, :, :), 1));

Patient_Noun = squeeze(mean(data(group == 1, 1, :, :), 1));
Patient_Verb = squeeze(mean(data(group == 1, 2, :, :), 1));


%% ==========================================================
% 8. Compute observed contrasts
%% ==========================================================

% Group contrast:
% Patients minus HC, averaged over noun and verb
HC_mean = squeeze(mean(mean(data(group == -1,:,:,:), 2), 1));
Patient_mean = squeeze(mean(mean(data(group == 1,:,:,:), 2), 1));

groupDiff = Patient_mean - HC_mean;

% Word-type contrast:
% Verb minus Noun
verbNoun_HC = HC_Verb - HC_Noun;
verbNoun_Patient = Patient_Verb - Patient_Noun;

wordTypeDiff = squeeze(mean(data(:,2,:,:) - data(:,1,:,:), 1));

% Interaction:
% (Verb - Noun) in Patients minus (Verb - Noun) in HC
interactionDiff = verbNoun_Patient - verbNoun_HC;


%% ==========================================================
% 9. Compute ground-truth effects
%% ==========================================================

truth_HC_Noun      = zeros(nChan, nTime);
truth_HC_Verb      = zeros(nChan, nTime);
truth_Patient_Noun = zeros(nChan, nTime);
truth_Patient_Verb = zeros(nChan, nTime);

for ch_idx = 1:length(effectChans)

    ch = effectChans(ch_idx);

    truth_HC_Noun(ch,:)      = weights(ch_idx) * amp_HC_Noun      * p300Shape';
    truth_HC_Verb(ch,:)      = weights(ch_idx) * amp_HC_Verb      * p300Shape';
    truth_Patient_Noun(ch,:) = weights(ch_idx) * amp_Patient_Noun * p300Shape';
    truth_Patient_Verb(ch,:) = weights(ch_idx) * amp_Patient_Verb * p300Shape';

end

truthGroupDiff = ...
    ((truth_Patient_Noun + truth_Patient_Verb) / 2) - ...
    ((truth_HC_Noun + truth_HC_Verb) / 2);

truthWordTypeDiff = ...
    ((truth_HC_Verb - truth_HC_Noun) + ...
     (truth_Patient_Verb - truth_Patient_Noun)) / 2;

truthInteractionDiff = ...
    (truth_Patient_Verb - truth_Patient_Noun) - ...
    (truth_HC_Verb - truth_HC_Noun);


%% ==========================================================
% 10. Figure 1: ERP waveform at Pz
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

figure;

plot(times, HC_Noun(channelToPlot,:), ...
    'Color', [0.00 0.45 0.74], 'LineWidth', 2); 
hold on;

plot(times, HC_Verb(channelToPlot,:), ...
    'Color', [0.00 0.45 0.74], 'LineStyle', '--', 'LineWidth', 2);

plot(times, Patient_Noun(channelToPlot,:), ...
    'Color', [0.85 0.33 0.10], 'LineWidth', 2);

plot(times, Patient_Verb(channelToPlot,:), ...
    'Color', [0.85 0.33 0.10], 'LineStyle', '--', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title('Split-plot ERP waveform at Pz');

legend('HC Noun', 'HC Verb', ...
       'Patient Noun', 'Patient Verb');

grid on;


%% ==========================================================
% 11. Figure 2: HC Verb - Noun
%% ==========================================================

figure;

imagesc(times, 1:nChan, verbNoun_HC);
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
title('HC Within-subject Effect: Verb - Noun');
colorbar;


%% ==========================================================
% 12. Figure 3: Patient Verb - Noun
%% ==========================================================

figure;

imagesc(times, 1:nChan, verbNoun_Patient);
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
title('Patient Within-subject Effect: Verb - Noun');
colorbar;


%% ==========================================================
% 13. Figure 4: Interaction effect
%% ==========================================================

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
title('Interaction: Patient(Verb - Noun) - HC(Verb - Noun)');
colorbar;


%% ==========================================================
% 14. Figure 5: Ground-truth interaction
%% ==========================================================

figure;

imagesc(times, 1:nChan, truthInteractionDiff);
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
title('Ground-Truth Interaction Effect');
colorbar;


%% ==========================================================
% 15. Topography at 300 ms
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~, peakIdx] = min(abs(times - p300Latency));

if exist('topoplot', 'file')

    figure;
    topoplot(interactionDiff(:,peakIdx), ...
        chanlocs_plot, 'electrodes', 'labels');

    colorbar;
    title('Observed Interaction at 300 ms');

else

    warning('topoplot not found. Please add EEGLAB to your MATLAB path.');

end


%% ==========================================================
% 16. Save dataset
%% ==========================================================

if ~exist('../data', 'dir')
    mkdir('../data');
end

save('../data/06_simulated_split_plot_EEG.mat', ...
     'data', ...
     'group', ...
     'subjectID', ...
     'designTable', ...
     'times', ...
     'chanlocs_EEG', ...
     'effectChans', ...
     'effectChanLabels', ...
     'condNames', ...
     'groupNames', ...
     'subjectNoise', ...
     'subjectNoiseSD', ...
     'backgroundNoiseSD', ...
     'groupDiff', ...
     'wordTypeDiff', ...
     'interactionDiff', ...
     'verbNoun_HC', ...
     'verbNoun_Patient', ...
     'truthGroupDiff', ...
     'truthWordTypeDiff', ...
     'truthInteractionDiff');

disp('Dataset saved: ../data/06_simulated_split_plot_EEG.mat');