%% ==========================================================
% Convert simulated within-subject EEG data to CSV
%
% Input:
%   EEGdata      : Rows ?? Channels ?? Time
%   designTable  : Rows ?? design variables
%
%   Each row of EEGdata corresponds to one row of designTable.
%
% Output:
%
% 1. 05_designTable.csv
%
%   Subject   CondCode   CondName
%   -------   --------   --------
%      1         -1      Control
%      1          1      Treatment
%      2         -1      Control
%      2          1      Treatment
%      ...
%
%
% 2. 05_EEGdata_long.csv
%
%   Subject   CondCode   Channel   Time   Amplitude
%   -------   --------   -------   ----   ---------
%      1         -1       Fp1      -200     ...
%      1         -1       Fp1      -196     ...
%      ...
%      1         -1       TP10      800     ...
%      1          1       Fp1      -200     ...
%      ...
%      2         -1       Fp1      -200     ...
%
% Thus, every row in designTable is expanded into
% (Channels ?? Time) rows in 05_EEGdata_long.csv.
%
%% ==========================================================

clear all; clc; close all

%% Load MAT file

load('../data/05_simulated_within_subject_EEG.mat', ...
    'EEGdata', ...
    'designTable');

%% Create output folder

if ~exist('../data','dir')
    mkdir('../data');
end

%% Time vector

times = -200:4:800;

%% Channel labels

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

%% ==========================================================
% Check dimensions
%% ==========================================================

[nObs,nChan,nTime] = size(EEGdata);

if nChan ~= numel(chanLabels_32)
    error('Number of channels does not match channel labels.');
end

if nTime ~= numel(times)
    error('Number of time points does not match the time vector.');
end

if height(designTable) ~= nObs
    error('Number of rows in designTable must equal the first dimension of EEGdata.');
end

%% ==========================================================
% Save design table
%% ==========================================================

writetable(designTable,'../data/05_designTable.csv');

%% ==========================================================
% Convert EEG data to long format
%% ==========================================================

nRows = nObs * nChan * nTime;

Subject   = zeros(nRows,1);
CondCode  = zeros(nRows,1);
Channel   = cell(nRows,1);
Time      = zeros(nRows,1);
Amplitude = zeros(nRows,1);

row = 1;

for obs = 1:nObs
    
    for ch = 1:nChan
        
        for t = 1:nTime
            
            Subject(row)   = designTable.Subject(obs);
            CondCode(row)  = designTable.CondCode(obs);
            Channel{row}   = chanLabels_32{ch};
            Time(row)      = times(t);
            Amplitude(row) = EEGdata(obs,ch,t);
            
            row = row + 1;
            
        end
    end
end

EEGlong = table( ...
    Subject, ...
    CondCode, ...
    Channel, ...
    Time, ...
    Amplitude);

%% ==========================================================
% Save EEG long table
%% ==========================================================

writetable(EEGlong,'../data/05_EEGdata_long.csv');

%% ==========================================================
% Finished
%% ==========================================================

disp('Saved files:');
disp('../data/05_designTable.csv');
disp('../data/05_EEGdata_long.csv');