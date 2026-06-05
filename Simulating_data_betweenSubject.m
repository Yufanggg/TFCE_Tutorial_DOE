%%% between-subject design (simple version)
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

nControl   = 30;
nTreatment = 30;

nSub  = nControl + nTreatment;
nChan = 64;

times = -200:4:800;
nTime = length(times);

%% Simulate baseline noise
noiseSD = 1.5;
data = noiseSD * randn(nSub, nChan, nTime);

%% Group labels

group = [zeros(nControl,1); ones(nTreatment,1)]; % 0 = Control, 1 = Treatment

%% Define P300 effect
p300Latency = 300;
p300Width = 70;

controlAmp = 3.0;
treatmentAmp = 6.0;

controlP300 = controlAmp * exp(-(times - p300Latency).^2 / ...
              (2 * p300Width^2));

treatmentP300 = treatmentAmp * exp(-(times - p300Latency).^2 / ...
                (2 * p300Width^2));

controlP300 = controlP300(:);
treatmentP300 = treatmentP300(:);

%% Effect channels

effectChans = [30 31 37 38];

%% Inject P300 into both groups

for s = 1:nControl

    for ch = effectChans

        tmp = squeeze(data(s,ch,:));

        tmp = tmp + controlP300;

        data(s,ch,:) = reshape(tmp,1,1,nTime);

    end

end

for s = (nControl+1):nSub
    for ch = effectChans
        tmp = squeeze(data(s,ch,:));
        tmp = tmp + treatmentP300;
        data(s,ch,:) = reshape(tmp,1,1,nTime);
    end
end
%% ==========================================================
% Figure 1: ERP waveform
%% ==========================================================

channelToPlot = 30;

controlERP = squeeze(mean(data(group==0,channelToPlot,:),1));
treatmentERP = squeeze(mean(data(group==1,channelToPlot,:),1));

figure;

plot(times,controlERP,'LineWidth',2);
hold on;

plot(times,treatmentERP, 'r','LineWidth',2);

xlabel('Time (ms)');
ylabel('Amplitude (\muV)');

title(sprintf('ERP waveform (Channel %d)',channelToPlot));

legend('Control','Treatment');

%xline(0,'--');
grid on;

%% ==========================================================
% Figure 2: Ground-truth channel ¡Á time effect map
%% ==========================================================

groupDiff = squeeze(mean(data(group==1,:,:),1) - ...
                    mean(data(group==0,:,:),1));

figure;

imagesc(times,1:nChan,groupDiff);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('Observed Group Difference');

colorbar;

%% ==========================================================
% Figure 3: Ground-truth injected effect
%% ==========================================================

truthDiff = zeros(nChan,nTime);

trueDifference = treatmentP300 - controlP300;

for ch = effectChans

    truthDiff(ch,:) = trueDifference';

end

figure;

imagesc(times,1:nChan,truthDiff);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('Ground-Truth Simulated Effect');

colorbar;

%% ==========================================================
% Figure 4: Topography at peak latency (requires EEGLAB)
%% ==========================================================

if exist('readlocs','file')

    [~,peakIdx] = min(abs(times-400));

    topo = groupDiff(:,peakIdx);

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

save('./data/simulated_between_subject_EEG.mat',...
     'data','group','times','effectChans');