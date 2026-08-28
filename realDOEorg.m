%% B_analysis 
% Perform lmeEEG on simulated EEG data 
clear all;clc
load('Exp02_chanloc_.mat');
load('Exp02_time_.mat');
load('Exp02_sEEG_1.mat');
DesignM = readtable('Exp02_DesignM.csv', 'ReadVariableNames', false);
DesignM.Properties.VariableNames = {'SubjID', 'Markers'};
DesignM.Properties.VariableNames{'Markers'} = 'Marker3';
unique_SubjID = unique(DesignM.SubjID);
Table = readtable('SemCatEStork.csv');
Table.Properties.VariableNames{'x___Target'} = 'Target';
Table2 = readtable('FreqDistra.csv');
Table = innerjoin(Table, Table2,'Keys', {'Distractor'});

repeatedTable = repmat(Table, 36, 1);
SubjList = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,...
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, ...
    45, 46];
repeatedTable.SubjID = kron(SubjList(:), ones(102, 1));

mergedTable = innerjoin(DesignM, repeatedTable,'Keys', {'SubjID', 'Marker3'});

Order = readtable('Exp02_SeqSI.csv');
Order.OrderIndex = (1:height(Order))';
repeatedOrder = innerjoin(mergedTable, Order, 'Keys', {'SubjID', 'Marker3'});


accuracyTable = readtable('Exp02_BehavioralData_R.csv');
isNA = cellfun(@(x) ischar(x) && strcmpi(strtrim(x), 'NA'), ...
               accuracyTable.Marker3);
accuracyTable(isNA, :) = [];

accuracyTable.Marker3 = cellfun(@str2double, accuracyTable.Marker3);
accuracyTable = accuracyTable(:, {'SubjID', 'Marker3', 'Accuricies'});

repeatedOrder = innerjoin(repeatedOrder, accuracyTable, 'Keys', {'SubjID', 'Marker3'});

% Restore the original Order row order
repeatedOrder = sortrows(repeatedOrder, 'OrderIndex');

rowsToRemove = repeatedOrder.SubjID == 12 | repeatedOrder.SubjID == 44 |...
    repeatedOrder.ExpTrialList == 1 | repeatedOrder.ExpTrialList == 2 |...
    repeatedOrder.ExpTrialList == 3 | repeatedOrder.ExpTrialList == 4 |...
    repeatedOrder.ExpTrialList == 74 | repeatedOrder.ExpTrialList == 75 |...
    repeatedOrder.ExpTrialList == 76 | repeatedOrder.ExpTrialList == 77;
    %repeatedOrder.Accuricies == 0 | ;

repeatedOrder(rowsToRemove, :) = [];
sEEG = sEEG(:, :, ~rowsToRemove);

channelinfo = EEG.chanlocs;

designSim = readtable('DesignStimulation_exp_.csv');

lengthTable = designSim(:, {'Distractor', 'LengthofDistrctor'});
lengthTable = unique(lengthTable);

% Each distractor must have exactly one length
assert(height(lengthTable) == ...
       numel(unique(lengthTable.Distractor)), ...
       'Some distractors have multiple LengthofDistrctor values.');

% Find each repeatedOrder distractor in lengthTable
[isFound, location] = ismember( ...
    repeatedOrder.Distractor, ...
    lengthTable.Distractor);

assert(all(isFound), ...
    'Some distractors in repeatedOrder were not found in designSim.');

% Add the variable without changing row order
repeatedOrder.LengthofDistrctor = lengthTable.LengthofDistrctor(location);

EEGdata = permute(sEEG, [3 1 2]);
designTable = repeatedOrder;

accuracyTable = readtable('Exp02_BehavioralData_R.csv');


clear DesignM sEEG Table Table2 ans designSim isFound lengthTable location repeatedOrder 
clear Order repeatedTable rowsToRemove ...
    SubjList Table unique_SubjID mergedTable EEG isNA
save('realDOE_3.mat');


disp('Data is well-organized!!!!!!!!!!');