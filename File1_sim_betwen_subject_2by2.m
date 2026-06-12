%%% between-subject design (2-by-2 design with interactions)
%% ==========================================================
% Simulated EEG/ERP dataset
% Between-subject design:
% Control (n=30) vs Treatment (n=30)
%
% Data dimensions:
% Subjects x Channels x Time
%
% Ground truth:
% Positive ERP effect at 400 ms
% Channels: 30,31,37,38
%% ==========================================================

clear; clc; close all; rng(123);

%% Simulation settings

nFactor00 = 30;
nFactor01 = 30;
nFactor10 = 30;
nFactor11 = 30;

nSub  = nFactor00 + nFactor01 + nFactor10 + nFactor11;
nChan = 32;

times = -200:4:800;
nTime = length(times);
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

%% Simulate baseline noise
noiseSD = 1.5;
data = noiseSD * randn(nSub, nChan, nTime);

%% Group labels, Effect-coded 2x2 design variables
% group 0 = A- B-
% group 1 = A- B+
% group 2 = A+ B-
% group 3 = A+ B+

group = [zeros(nFactor00,1); ones(nFactor01,1); 2*ones(nFactor10,1); ... 
    3*ones(nFactor11,1)]; % 0 = Control, 1 = Treatment

var1 = 2*ismember(group,[2 3]) - 1;   % Factor A: -1 vs +1
var2 = 2*ismember(group,[1 3]) - 1;   % Factor B: -1 vs +1
unique([group var1 var2],'rows')
designCheck = table(group, var1, var2, ...
    'VariableNames', {'group','FactorA','FactorB'});

disp(unique(designCheck,'rows'));
%% Define P300 effect
p300Latency = 300;
p300Width = 70;

Factor00 = 3.0
Factor01 = 6.0
Factor10 = 4.0
Factor11 = 7.0

Factor00P300 = Factor00 * exp(-(times - p300Latency).^2 / ...
              (2 * p300Width^2));
Factor01P300 = Factor01 * exp(-(times - p300Latency).^2 / ...
                (2 * p300Width^2));
         

Factor10P300 = Factor10 * exp(-(times - p300Latency).^2 / ...
              (2 * p300Width^2));
Factor11P300 = Factor11 * exp(-(times - p300Latency).^2 / ...
                (2 * p300Width^2));           

Factor00P300 = Factor00P300(:);
Factor01P300 = Factor01P300(:);
Factor10P300 = Factor10P300(:);
Factor11P300 = Factor11P300(:);

%% Effect channels
effectChansLabl = {'Cz','CP1','CP2','Pz','P3','P4'};
[tf, effectChans] = ismember(effectChansLabl, {chanlocs_EEG.labels});

if any(~tf)
    error('Missing effect channels: %s', strjoin(effectChansLabl(~tf), ', '));
end
%% Inject P300 into all groups
weights = [0.6 0.8 0.8 1.0 0.75 0.75];
for s = 1:nFactor00

    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s, ch, :));

        tmp = tmp + weights(ch_idx) *Factor00P300;

        data(s,ch,:) = reshape(tmp,1,1,nTime);

    end

end

for s = (nFactor00+1):(nFactor00 + nFactor01)
    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s,ch,:));
        
        tmp = tmp + weights(ch_idx) *Factor01P300;
        
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end

for s = (nFactor00 + nFactor01 + 1):(nFactor00 + nFactor01 + nFactor10)
    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s,ch,:));
        
        tmp = tmp + weights(ch_idx) * Factor10P300;
        
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end

for s = (nFactor00 + nFactor01 + nFactor10 + 1):nSub
    for ch_idx = 1:length(effectChans)

        ch = effectChans(ch_idx);

        tmp = squeeze(data(s,ch,:));
        
        tmp = tmp + weights(ch_idx) * Factor11P300;
        
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end
%% ==========================================================
% Figure 1: ERP waveform
%% ==========================================================

channelToPlot = find(strcmp({chanlocs_EEG.labels}, 'Pz'));

FactorA_minus_ERP = squeeze(mean(data(group == 0 | group == 1, channelToPlot, :),1));
FactorA_plus_ERP  = squeeze(mean(data(group == 2 | group == 3, channelToPlot, :),1));

FactorB_minus_ERP = squeeze(mean(data(group == 0 | group == 2, channelToPlot, :),1));
FactorB_plus_ERP  = squeeze(mean(data(group == 1 | group == 3, channelToPlot, :),1));

figure;
plot(times, FactorA_minus_ERP, 'b', 'LineWidth', 2); hold on;
plot(times, FactorA_plus_ERP,  'r', 'LineWidth', 2);
plot(times, FactorB_minus_ERP, 'g', 'LineWidth', 2);
plot(times, FactorB_plus_ERP,  'm', 'LineWidth', 2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');
title(sprintf('ERP waveform at %s', chanlocs_EEG(channelToPlot).labels));
legend('Factor A-','Factor A+','Factor B-','Factor B+');
grid on;

%% ==========================================================
% Figure 2: Observed channel ¡Á time effect maps
%% ==========================================================

tick_labels = {chanlocs_EEG.labels};

% Factor A: A+ minus A-
groupADiff = squeeze(mean(data(group == 2 | group == 3,:,:),1) - ...
                     mean(data(group == 0 | group == 1,:,:),1));

figure;
imagesc(times, 1:nChan, groupADiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tick_labels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed Factor A Effect: A+ minus A-');
colorbar;


% Factor B: B+ minus B-
groupBDiff = squeeze(mean(data(group == 1 | group == 3,:,:),1) - ...
                     mean(data(group == 0 | group == 2,:,:),1));

figure;
imagesc(times, 1:nChan, groupBDiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tick_labels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed Factor B Effect: B+ minus B-');
colorbar;


%% ==========================================================
% Figure 3: Ground-truth injected effect maps
%% ==========================================================

truthADiff = zeros(nChan,nTime);
truthBDiff = zeros(nChan,nTime);

% Pure injected effects, no noise
Aminus = mean([Factor00P300, Factor01P300], 2);
Aplus  = mean([Factor10P300, Factor11P300], 2);

Bminus = mean([Factor00P300, Factor10P300], 2);
Bplus  = mean([Factor01P300, Factor11P300], 2);

trueADifference = Aplus - Aminus;
trueBDifference = Bplus - Bminus;

for ch_idx = 1:length(effectChans)
    ch = effectChans(ch_idx);

    truthADiff(ch,:) = weights(ch_idx) * trueADifference';
    truthBDiff(ch,:) = weights(ch_idx) * trueBDifference';
end


figure;
imagesc(times, 1:nChan, truthADiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tick_labels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Ground-Truth Simulated Effect for Factor A');
colorbar;


figure;
imagesc(times, 1:nChan, truthBDiff);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', tick_labels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Ground-Truth Simulated Effect for Factor B');
colorbar;


%% ==========================================================
% Figure 4: Topography at peak latency
%% ==========================================================

chanlocs_plot = chanlocs_EEG;

for k = 1:length(chanlocs_plot)
    chanlocs_plot(k).theta = chanlocs_plot(k).theta + 90;
end

[~,peakIdx] = min(abs(times - 300));

if exist('topoplot','file')

    figure;
    topoplot(groupADiff(:,peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed Factor A Effect at 300 ms');

    figure;
    topoplot(groupBDiff(:,peakIdx), chanlocs_plot, 'electrodes', 'labels');
    colorbar;
    title('Observed Factor B Effect at 300 ms');

else
    fprintf('topoplot not found. Please add EEGLAB to the MATLAB path.\n');
end


%% ==========================================================
% Save dataset
%% ==========================================================
save('./data/02_simulated_between_subject_2by2_EEG.mat',...
     'data','var1', 'var2','times','effectChans');