%% ==========================================================
% Convert simulated nested class-student EEG data to CSV
%
% Input:
%   EEGdata      : Class-student-condition rows ?? Channels ?? Time
%   designTable  : One row per class-student-condition observation
%
%   Each row of EEGdata corresponds to one row of designTable.
%
% Output:
%
% 1. 07_designTable.csv
%
%   Class   Student   Condition   ConditionCode
%
% 2. 07_EEGdata_long.csv
%
%   Class   Student   ConditionCode   Channel   Time   Amplitude
%
% Thus, each row of designTable is expanded into
% Channels ?? Time rows in 07_EEGdata_long.csv.
%% ==========================================================

clear all; clc;close all

%% Load MAT file

load('../data/07_simulated_nested_class_student_EEG.mat', ...
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

requiredVars = {'Class','Student','Condition','ConditionCode'};

for v = 1:numel(requiredVars)
    if ~ismember(requiredVars{v}, designTable.Properties.VariableNames)
        error('designTable is missing required variable: %s', requiredVars{v});
    end
end

%% Save design table

writetable(designTable,'../data/07_designTable.csv');

%% Convert EEG data to long format

nRows = nObs * nChan * nTime;

Class         = zeros(nRows,1);
Student       = zeros(nRows,1);
ConditionCode = zeros(nRows,1);
Channel       = cell(nRows,1);
Time          = zeros(nRows,1);
Amplitude     = zeros(nRows,1);

row = 1;

for obs = 1:nObs
    
    for ch = 1:nChan
        
        for t = 1:nTime
            
            Class(row)         = designTable.Class(obs);
            Student(row)       = designTable.Student(obs);
            ConditionCode(row) = designTable.ConditionCode(obs);
            Channel{row}       = chanLabels_32{ch};
            Time(row)          = times(t);
            Amplitude(row)     = EEGdata(obs,ch,t);
            
            row = row + 1;
            
        end
    end
end

EEGlong = table( ...
    Class, ...
    Student, ...
    ConditionCode, ...
    Channel, ...
    Time, ...
    Amplitude);

%% Save EEG long table

writetable(EEGlong,'../data/07_EEGdata_long.csv');

%% Finished

disp('Saved files:');
disp('../data/07_designTable.csv');
disp('../data/07_EEGdata_long.csv');