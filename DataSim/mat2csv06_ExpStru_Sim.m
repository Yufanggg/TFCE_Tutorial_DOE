%% ==========================================================
% Convert simulated split-plot EEG data to CSV
%
% Input:
%   EEGdata      : Subject-condition rows ¡Á Channels ¡Á Time
%   designTable  : One row per subject-condition observation
%
%   Each row of EEGdata corresponds to one row of designTable.
%
% Output:
%
% 1. 06_designTable.csv
%
%   Subject   GroupCode   GroupName   CondCode   CondName
%
% 2. 06_EEGdata_long.csv
%
%   Subject   GroupCode   CondCode   Channel   Time   Amplitude
%
% Thus, each row of designTable is expanded into
% Channels ¡Á Time rows in 06_EEGdata_long.csv.
%% ==========================================================

clear; clc;

%% Load MAT file

load('../data/06_simulated_split_plot_EEG.mat', ...
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

%% Check dimensions

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

requiredVars = {'Subject','GroupCode','GroupName','CondCode','CondName'};

for v = 1:numel(requiredVars)
    if ~ismember(requiredVars{v}, designTable.Properties.VariableNames)
        error('designTable is missing required variable: %s', requiredVars{v});
    end
end

%% Save design table

writetable(designTable,'../data/06_designTable.csv');

%% Convert EEG data to long format

nRows = nObs * nChan * nTime;

Subject   = zeros(nRows,1);
GroupCode = zeros(nRows,1);
CondCode  = zeros(nRows,1);
Channel   = cell(nRows,1);
Time      = zeros(nRows,1);
Amplitude = zeros(nRows,1);

row = 1;

for obs = 1:nObs
    
    for ch = 1:nChan
        
        for t = 1:nTime
            
            Subject(row)   = designTable.Subject(obs);
            GroupCode(row) = designTable.GroupCode(obs);
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
    GroupCode, ...
    CondCode, ...
    Channel, ...
    Time, ...
    Amplitude);

%% Save EEG long table

writetable(EEGlong,'../data/06_EEGdata_long.csv');

%% Finished

disp('Saved files:');
disp('../data/06_designTable.csv');
disp('../data/06_EEGdata_long.csv');