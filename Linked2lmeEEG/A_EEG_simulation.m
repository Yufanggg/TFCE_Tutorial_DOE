%% EEG SIMULATION

clear all
eeglab nogui
addpath(genpath('C:\Users\Antonino\Documents\MATLAB\Toolbox\SEREEGA-master'))
rng('default');rng(1986) %Set seed

%Set montage and obtain a lead field
lf = lf_generate_fromnyhead('labels', {'Fp1','Fp2','F7','F3','Fz','F4','F8',...
    'T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'});

% Epochs
epochs = struct();
epochs.n = 1;          % The number of epochs to simulate for each Factor level, Subject, and Item
epochs.srate = 100;     % Sampling rate in Hz
epochs.length = 1100;   % Epoch length in ms
epochs.prestim = 100;


p3effect = .2;        % Fixed effect


%% Create P3


% P300
p300 = utl_get_component_fromtemplate('p3a_erp', lf);
p300 = utl_shift_latency(p300, epochs.prestim);
p300.signal{1, 1}.peakAmplitudeDv=0;
p300.signal{1, 1}.peakLatencyDv=0;
p300.signal{1, 1}.peakWidthDv=0;
p300.signal{1, 1}.peakAmplitudeSlope=0;
P3 = generate_scalpdata(p300, lf, epochs);
EEG = utl_create_eeglabdataset(P3, epochs, lf, 'marker', 'ConditionA');
chanlocs = EEG.chanlocs;
sEEG_times = EEG.times;


% P300
p300 = utl_get_component_fromtemplate('p3a_erp', lf);
p300 = utl_shift_latency(p300, epochs.prestim);
p300.signal{1, 1}.peakAmplitude=p300.signal{1, 1}.peakAmplitude+p3effect
p300.signal{1, 1}.peakAmplitudeDv=0;
p300.signal{1, 1}.peakLatencyDv=0;
p300.signal{1, 1}.peakWidthDv=0;
p300.signal{1, 1}.peakAmplitudeSlope=0;
P3(:,:,2) = generate_scalpdata(p300, lf, epochs);


save(['..',filesep,'DATA',filesep,'A_SIM',filesep,'sEEG_all.mat'],'P3','chanlocs','sEEG_times','-v7.3');










