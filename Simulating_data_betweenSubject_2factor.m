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
nChan = 64;

times = -200:4:800;
nTime = length(times);

%% Simulate baseline noise
noiseSD = 1.5;
data = noiseSD * randn(nSub, nChan, nTime);

%% Group labels

group = [zeros(nFactor00,1); ones(nFactor01,1); 2*ones(nFactor10,1); 3*ones(nFactor11,1)]; % 0 = Control, 1 = Treatment

%% Define P300 effect
p300Latency = 300;
p300Width = 70;

Factor00 = 3.0
Factor01 = 6.0
Factor10 = 4.0
Fatcor11 = 10.0

Factor00P300 = Factor00 * exp(-(times - p300Latency).^2 / ...
              (2 * p300Width^2));
Factor01P300 = Factor01 * exp(-(times - p300Latency).^2 / ...
                (2 * p300Width^2));
         

Factor10P300 = Factor10 * exp(-(times - p300Latency).^2 / ...
              (2 * p300Width^2));
Fatcor11P300 = Fatcor11 * exp(-(times - p300Latency).^2 / ...
                (2 * p300Width^2));           

Factor00P300 = Factor00P300(:);
Factor01P300 = Factor01P300(:);
Factor10P300 = Factor10P300(:);
Fatcor11P300 = Fatcor11P300(:);

%% Effect channels

effectChans = [30 31 37 38];

%% Inject P300 into all groups
for s = 1:nFactor00

    for ch = effectChans

        tmp = squeeze(data(s,ch,:));

        tmp = tmp + Factor00P300;

        data(s,ch,:) = reshape(tmp,1,1,nTime);

    end

end

for s = (nFactor00+1):(nFactor00 + nFactor01)
    for ch = effectChans
        tmp = squeeze(data(s,ch,:));
        tmp = tmp + Factor01P300;
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end

for s = (nFactor00 + nFactor01 + 1):(nFactor00 + nFactor01 + nFactor10)
    for ch = effectChans
        tmp = squeeze(data(s,ch,:));
        tmp = tmp + Factor10P300;
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end

for s = (nFactor00 + nFactor01 + nFactor10 + 1):nSub
    for ch = effectChans
        tmp = squeeze(data(s,ch,:));
        tmp = tmp + Fatcor11P300;
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end
%% ==========================================================
% Figure 1: ERP waveform
%% ==========================================================

channelToPlot = 30;

Factor0_ERP = squeeze(mean(data((group == 0) | (group == 1),channelToPlot,:),1));
Factor1_ERP = squeeze(mean(data((group == 2) | (group == 3),channelToPlot,:),1));
Factor_0ERP = squeeze(mean(data((group == 0) | (group == 2),channelToPlot,:),1));
Factor_1ERP = squeeze(mean(data((group == 1) | (group == 3),channelToPlot,:),1));

figure;

plot(times,Factor0_ERP,'LineWidth',2);
hold on;
plot(times,Factor1_ERP, 'r','LineWidth',2);
hold on;
plot(times,Factor_0ERP, 'y','LineWidth',2);
hold on;
plot(times,Factor_1ERP, 'g','LineWidth',2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');

title(sprintf('ERP waveform (Channel %d)',channelToPlot));

legend('FactorA+','FactorA-', 'FactorC+','FactorC-');

%xline(0,'--');
grid on;

%% ==========================================================
% Figure 2: Ground-truth channel ¡Á time effect map
%% ==========================================================

groupADiff = squeeze(mean(data((group == 0) | (group == 1),:,:),1) - ...
                    mean(data((group == 2) | (group == 3),:,:),1));

figure;

imagesc(times,1:nChan,groupADiff);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('Observed VariableA Difference');

colorbar;

groupBDiff = squeeze(mean(data((group == 0) | (group == 2),:,:),1) - ...
                    mean(data((group == 1) | (group == 3),:,:),1));

figure;

imagesc(times,1:nChan,groupBDiff);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('Observed VariableC Difference');

colorbar;

%% ==========================================================
% Figure 3: Ground-truth injected effect
%% ==========================================================

truthADiff = zeros(nChan,nTime);

trueADifference = Factor_0ERP - Factor_1ERP;

for ch = effectChans

    truthADiff(ch,:) = trueADifference';

end

figure;

imagesc(times,1:nChan,truthADiff);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('Ground-Truth Simulated Effect for Variable A');

colorbar;


truthBDiff = zeros(nChan,nTime);

trueBDifference = Factor0_ERP - Factor1_ERP;

for ch = effectChans

    truthBDiff(ch,:) = trueBDifference';

end

figure;

imagesc(times,1:nChan,truthBDiff);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('Ground-Truth Simulated Effect for Variable C');
colorbar;
%% ==========================================================
% Figure 4: Topography at peak latency (requires EEGLAB)
%% ==========================================================

if exist('readlocs','file')

    [~,peakIdx] = min(abs(times-400));

    topo = groupADiff(:,peakIdx);

    try

        chanlocs = readlocs('standard_1005.elc');

        figure;

        topoplot(topo,chanlocs(1:64));

        colorbar;

        title('Group Difference at 400 ms');

    catch

        fprintf('Could not load channel locations.\n');

    end

end

if exist('readlocs','file')

    [~,peakIdx] = min(abs(times-400));

    topo = groupBDiff(:,peakIdx);

    try

        chanlocs = readlocs('standard_1005.elc');

        figure;

        topoplot(topo,chanlocs(1:64));

        colorbar;

        title('Group Difference at 400 ms');

    catch

        fprintf('Could not load channel locations.\n');

    end

end

%% ==========================================================
% Save dataset
%% ==========================================================
save('./data/simulated_between_subject_2by2Int_EEG.mat',...
     'data','group','times','effectChans');