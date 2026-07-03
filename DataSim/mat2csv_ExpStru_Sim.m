%% ==========================================================
% Load saved MAT file and export to CSV long format
%% ==========================================================

clear; clc;

%% Load MAT file
load('../data/04_simulated_between_subject_2by2Int_EEG.mat', ...
     'EEGdata', ...
     'designTable');

%% Output folder
if ~exist('../data', 'dir')
    mkdir('../data');
end

%% Recreate time vector
times = -200:4:800;

%% Define channel labels
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

%% Check dimensions
[nSub, nChan, nTime] = size(EEGdata);

if nChan ~= length(chanLabels_32)
    error('Number of channels in EEGdata does not match chanLabels_32.');
end

if nTime ~= length(times)
    error('Number of time points in EEGdata does not match times.');
end

%% Save design table
writetable(designTable, '../data/04_designTable.csv');

%% Convert EEGdata to long table
nRows = nSub * nChan * nTime;

subject   = zeros(nRows, 1);
channel   = cell(nRows, 1);
time      = zeros(nRows, 1);
Amplitude = zeros(nRows, 1);

row = 1;

for s = 1:nSub
    for ch = 1:nChan
        for t = 1:nTime

            subject(row)   = s;
            channel{row}   = chanLabels_32{ch};
            time(row)      = times(t);
            Amplitude(row) = EEGdata(s, ch, t);

            row = row + 1;

        end
    end
end

EEGcsv = table( ...
    time, ...
    channel, ...
    Amplitude, ...
    subject);

%% Save EEG CSV
writetable(EEGcsv, '../data/04_EEGdata_long.csv');

disp('Saved files:');
disp('../data/04_designTable.csv');
disp('../data/04_EEGdata_long.csv');