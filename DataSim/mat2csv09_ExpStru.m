%% Export trial-level EEG data to CSV
%
% Input:  ../Data/09_realDOE.mat
% Output: ../Data/09_designTable.csv
%         ../Data/09_EEGdata_long.csv
%
% Each row of designTable corresponds to one trial in EEGdata.
% Marker3 is copied to every channel-time row for that trial.

clearvars;
clc;

%% Load data

scriptFolder = fileparts(mfilename('fullpath'));
dataFolder = fullfile(scriptFolder, '..', 'Data');

load(fullfile(dataFolder, '09_realDOE.mat'), ...
    'EEGdata', 'designTable', 'channelinfo', 'time');

%% Check dimensions and required columns

[nObs, nChan, nTime] = size(EEGdata);

if height(designTable) ~= nObs
    error('designTable must have one row per trial in EEGdata.');
end

if numel(channelinfo) ~= nChan
    error('Number of channels does not match channel labels.');
end

if numel(time) ~= nTime
    error('Number of time points does not match the time vector.');
end

requiredVars = {
    'SubjID', 'Marker3', 'Target', 'JSD', 'ClassifierCongruency', ...
    'CongruencySemanticCategories', 'NumbersofStorks', ...
    'Frequency', 'LengthofDistrctor'
};

missingVars = setdiff(requiredVars, designTable.Properties.VariableNames);

if ~isempty(missingVars)
    error('designTable is missing required columns: %s', ...
        strjoin(missingVars, ', '));
end

% Remove RowID, including the earlier spelling rowID, if present.
isRowID = strcmpi(designTable.Properties.VariableNames, 'RowID');
designTable(:, isRowID) = [];

%% Convert EEG data to long format

% Time varies fastest, then channel, then trial.
obsIndex = kron((1:nObs).', ones(nChan * nTime, 1));
chanIndex = repmat(kron((1:nChan).', ones(nTime, 1)), nObs, 1);

channelLabels = {channelinfo.labels}.';

EEGlong = designTable(obsIndex, requiredVars);
EEGlong.Channel = channelLabels(chanIndex);
EEGlong.Time = repmat(time(:), nObs * nChan, 1);
EEGlong.Amplitude = reshape(permute(EEGdata, [3 2 1]), [], 1);


disp(designTable(1:min(20, height(designTable)), :));
disp(EEGlong(1:min(20, height(EEGlong)), :));

%% Save EEG long table
writetable(designTable, '../data/09_designTable.csv');
writetable(EEGlong, '../data/09_EEGdata_long.csv');

disp('Saved files:');
disp('../data/09_designTable.csv');
disp('../data/09_EEGdata_long.csv');
