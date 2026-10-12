%% Convert sEEG_all and sEEG_table to CSV
clear; clc;

%% Load input files
load('sEEG_all.mat');
load('sEEG_table.mat');

%% Check dimensions and required columns
[nChan, nTime, nObs] = size(sEEG_all);

if height(sEEG_table) ~= nObs
    error('Rows in sEEG_table must match trials in sEEG_all.');
end

requiredVars = {'ID', 'Item', 'Condition'};
for k = 1:numel(requiredVars)
    if ~ismember(requiredVars{k}, sEEG_table.Properties.VariableNames)
        error('sEEG_table is missing the variable: %s', requiredVars{k});
    end
end

%% Set time values
if exist('sEEG_times', 'var') && numel(sEEG_times) == nTime
    timeValues = sEEG_times(:);
else
    timeValues = (1:nTime)';
    warning('sEEG_times not found or has the wrong length; using sample indices.');
end

%% Convert Condition labels to A/B and create numeric codes
rawCondition = sEEG_table.Condition;

if ischar(rawCondition) || iscategorical(rawCondition)
    rawCondition = cellstr(rawCondition);
elseif ~iscell(rawCondition)
    error('Unexpected data type for sEEG_table.Condition.');
end

rawCondition = cellfun(@strtrim, rawCondition(:), 'UniformOutput', false);

CON = cell(nObs, 1);
CondCode = zeros(nObs, 1);

for k = 1:nObs
    if strcmp(rawCondition{k}, 'ConditionA') || strcmp(rawCondition{k}, 'A')
        CON{k} = 'A';
        CondCode(k) = 0;
    elseif strcmp(rawCondition{k}, 'ConditionB') || strcmp(rawCondition{k}, 'B')
        CON{k} = 'B';
        CondCode(k) = 1;
    else
        error('Unexpected condition label in trial %d: %s', ...
            k, rawCondition{k});
    end
end

%% Save trial-level metadata
designTable = table( ...
    sEEG_table.ID, ...
    sEEG_table.Item, ...
    rawCondition, ...
    CON, ...
    CondCode, ...
    'VariableNames', {'ID', 'Item', 'Condition', 'CON', 'CondCode'});

writetable(designTable, 'sEEG_table.csv');

%% Create long-format EEG data
% These indices follow MATLAB's column-major ordering, matching sEEG_all(:).
nRowsEEG = nObs * nChan * nTime;

Subject       = cell(nRowsEEG, 1);
Item          = cell(nRowsEEG, 1);
Condition     = cell(nRowsEEG, 1);
ConditionCode = zeros(nRowsEEG, 1);
Channel       = zeros(nRowsEEG, 1);
Time          = zeros(nRowsEEG, 1);
Amplitude     = zeros(nRowsEEG, 1);

row = 1

for obs = 1:nObs
    for ch = 1:nChan
        for t = 1:nTime
            Subject{row}       = designTable.ID(obs);
            Item{row}          = designTable.Item(obs);
            Condition{row}     = designTable.Condition(obs);
            ConditionCode(row) = designTable.CondCode(obs);
            Channel(row)       = ch;
            Time(row)          = t;
            Amplitude(row)     = sEEG_all(ch,t,obs);
            row = row + 1;
        end
    end
    disp(obs)
end

EEGlong = table( Subject, Item, ConditionCode, ...
    Channel, Time, Amplitude);

%% ==========================================================
% Preview
%% ==========================================================

disp(designTable(1:20,:));
disp(EEGlong(1:20,:));

%% Save EEG long table

writetable(EEGlong,'EEGdata_long.csv');

%% Finished

disp('Saved files:');
disp('../designTable.csv');
disp('../EEGdata_long.csv');