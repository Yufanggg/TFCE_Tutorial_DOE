%% ==========================================================
% Convert simulated nested class-student EEG data to CSV
%
% Input:
%   EEGdata      : Class-student-condition rows ¡Á Channels ¡Á Time
%   designTable  : One row per class-student-condition observation
%
%   Each row of EEGdata corresponds to one row of designTable.
%
% Output:
%
% 1. 08designTable.csv
%
%  Subject, Item, Condition, ConditionCode
%
% 2. 08_EEGdata_long.csv
%
%   Subject, Item, Condition, ConditionCode, Channel, Time, Amplitude
%
% Thus, each row of designTable is expanded into
% Channels ¡Á Time rows in 08_EEGdata_long.csv.
%% ==========================================================

clear all; clc; close all

%% Load MAT file

load('../data/09_realDOE.mat');

%% Create output folder

if ~exist('../Data','dir')
    mkdir('../Data');
end

%% Check dimensions

[nObs,nChan,nTime] = size(EEGdata);

if nChan ~= numel(channelinfo)
    error('Number of channels does not match channel labels.');
end

if nTime ~= numel(time)
    error('Number of time points does not match the time vector.');
end

if height(designTable) ~= nObs
    error('Number of rows in designTable must equal the first dimension of EEGdata.');
end

requiredVars = {'SubjID', 'Target', 'JSD', 'ClassifierCongruency', ...
    'CongruencySemanticCategories', 'NumbersofStorks', ...
    'Frequency', 'LengthofDistrctor'};

for v = 1:numel(requiredVars)
    if ~ismember(requiredVars{v}, designTable.Properties.VariableNames)
        error('designTable is missing required variable: %s', requiredVars{v});
    end
end

%% Save design table

writetable(designTable,'../data/09_designTable.csv');

%% Convert EEG data to long format
%% ==========================================================
obsIndex  = kron((1:nObs).', ones(nChan * nTime, 1));
chanIndex = repmat(kron((1:nChan).', ones(nTime, 1)), nObs, 1);

channelLabels = {channelinfo.labels}.';
timeValues = time(:);

EEGlong = designTable(obsIndex, requiredVars);
EEGlong.Channel = channelLabels(chanIndex);
EEGlong.Time = repmat(timeValues, nObs * nChan, 1);
EEGlong.Amplitude = reshape(permute(EEGdata, [3 2 1]), [], 1); 
% permute(EEGdata, [3 2 1])changes the dimension order from observation ¡Á channel ¡Á time to time ¡Á channel ¡Á observation.
% reshape(..., [], 1) stacks all values into one column.

disp(designTable(1:min(20, height(designTable)), :));
disp(EEGlong(1:min(20, height(EEGlong)), :));

%% Save EEG long table

writetable(EEGlong, '../data/09_EEGdata_long.csv');

disp('Saved files:');
disp('../data/09_designTable.csv');
disp('../data/09_EEGdata_long.csv');
