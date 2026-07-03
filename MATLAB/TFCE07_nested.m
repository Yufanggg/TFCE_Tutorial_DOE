%% ==========================================================
% TFCE analysis
% Nested within-student design
%
% Input file contains:
%   EEGdata
%   designTable
%
% EEGdata long-format columns:
%   Class
%   Student
%   Condition
%   ConditionCode      % -1 = Control, 1 = Treatment
%   Channel
%   Time
%   Amplitude
%
% designTable columns:
%   Class
%   Student
%   Condition
%   ConditionCode
%
% Model:
%   EEG ~ Condition + (1|Class) + (1|Class:Student)
%
% Test:
%   Treatment - Control fixed effect
%
% Permutation:
%   Flip condition labels within each student nested in class
%% ==========================================================

clear; clc; close all;

fprintf('\nStarting nested TFCE analysis...\n');

%% ==========================================================
% Load data
%% ==========================================================

load('../data/07_simulated_nested_class_student_EEG.mat', ...
     'EEGdata', 'designTable');

%% ==========================================================
% Basic variables from long-format EEG table
%% ==========================================================

times = unique(EEGdata.Time, 'stable');
chanLabels = unique(EEGdata.Channel, 'stable');

nTime = numel(times);
nChan = numel(chanLabels);

%% ==========================================================
% Reconstruct EEG array:
%
% EEGarray:
%   Observation x Channel x Time
%
% Observation = one Class-Student-Condition row
%% ==========================================================

obsTable = unique( ...
    EEGdata(:, {'Class','Student','Condition','ConditionCode'}), ...
    'rows', ...
    'stable');

nRows = height(obsTable);

EEGarray = nan(nRows, nChan, nTime);

[~, obsIdx] = ismember( ...
    EEGdata(:, {'Class','Student','Condition','ConditionCode'}), ...
    obsTable, ...
    'rows');

[~, chanIdx] = ismember(EEGdata.Channel, chanLabels);
[~, timeIdx] = ismember(EEGdata.Time, times);

for i = 1:height(EEGdata)

    EEGarray(obsIdx(i), chanIdx(i), timeIdx(i)) = EEGdata.Amplitude(i);

end

fprintf('Reconstructed EEGarray: %d rows x %d channels x %d time points.\n', ...
    nRows, nChan, nTime);

%% ==========================================================
% Design variables
%% ==========================================================

Class = obsTable.Class;
Student = obsTable.Student;
CondCode = obsTable.ConditionCode;

Class = Class(:);
Student = Student(:);
CondCode = CondCode(:);

ClassLME = categorical(Class);
StudentLME = categorical(Student);

% 0 = Control, 1 = Treatment
Condition = double(CondCode == 1);

%% ==========================================================
% Check nested student blocks
%% ==========================================================

NestedID = findgroups(Class, Student);
nestedUnits = unique(NestedID);
nUnits = numel(nestedUnits);

if nRows ~= nUnits * 2
    error('Expected exactly two condition rows per nested student.');
end

for u = 1:nUnits

    idxUnit = NestedID == nestedUnits(u);

    if sum(idxUnit) ~= 2
        error('Each nested student must have exactly two rows.');
    end

    if ~all(sort(CondCode(idxUnit)) == [-1; 1])
        error('Each nested student must have one Control (-1) and one Treatment (1).');
    end

end

fprintf('Design check passed: %d nested students, 2 conditions each.\n', nUnits);

%% ==========================================================
% Load channel locations
%% ==========================================================

chanlocs_1020 = readlocs('standard_1005.elc');

allLabels = {chanlocs_1020.labels};
[tf, idx] = ismember(chanLabels, allLabels);

if any(~tf)
    error('Missing channels: %s', strjoin(chanLabels(~tf), ', '));
end

e_loc = chanlocs_1020(idx);

%% ==========================================================
% Step 1: Observed LME t-map
%% ==========================================================

fprintf('Step 1: Computing observed t-statistic map...\n');

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        EEG = double(squeeze(EEGarray(:, ch, tp)));

        tbl = table(EEG, Condition, ClassLME, StudentLME, ...
            'VariableNames', {'EEG','Condition','Class','Student'});

        lme = fitlme(tbl, ...
            'EEG ~ Condition + (1|Class) + (1|Class:Student)');

        t_Obs(ch,tp) = lme.Coefficients.tStat(2);

    end
end

fprintf('Observed t-statistic map completed.\n');

%% ==========================================================
% Step 2: Observed TFCE map
%% ==========================================================

fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');

%% ==========================================================
% Step 3: Permutation
%
% Flip condition labels within each Class:Student unit
%% ==========================================================

nPerm = 999;
TFCE_permMax = nan(nPerm,1);

fprintf('Step 3: Starting permutation test: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    perm_t = nan(nChan, nTime);

    CondCode_perm = CondCode;

    for u = 1:nUnits

        idxUnit = find(NestedID == nestedUnits(u));

        if rand > 0.5
            CondCode_perm(idxUnit) = flipud(CondCode_perm(idxUnit));
        end

    end

    Condition_perm = double(CondCode_perm == 1);

    Class_perm = categorical(Class);
    Student_perm = categorical(Student);

    for ch = 1:nChan
        for tp = 1:nTime

            EEG = double(squeeze(EEGarray(:, ch, tp)));

            tbl = table(EEG, Condition_perm, Class_perm, Student_perm, ...
                'VariableNames', {'EEG','Condition','Class','Student'});

            lme = fitlme(tbl, ...
                'EEG ~ Condition + (1|Class) + (1|Class:Student)');

            perm_t(ch,tp) = lme.Coefficients.tStat(2);

        end
    end

    TFCE_perm = ept_mex_TFCE2D(perm_t, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

    fprintf('Finished permutation %d / %d\n', p, nPerm);

end

fprintf('Permutation testing completed.\n');

%% ==========================================================
% Step 4: TFCE correction
%% ==========================================================

fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;

maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);

maxTFCEcrit = maxTFCE(round(nPerm*(1-Alpha)));

Mask = abs(TFCE_Obs) >= critTFCE;

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        P_Values(ch, tpoint) = ...
            (sum(TFCE_permMax >= abs(P_Values(ch, tpoint))) + 1) / ...
            (nPerm + 1);

    end
end


fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', maxTFCEcrit);

%% ==========================================================
% Step 5: Store results
%% ==========================================================

fprintf('Step 5: Storing results...\n');

Results = struct();

Results.Obs = t_Obs;
Results.TFCE_Obs = TFCE_Obs;
Results.TFCE_Null = TFCE_permMax;
Results.critTFCE = maxTFCEcrit;
Results.P_Values = P_Values;
Results.Mask = Mask;
Results.alpha = alpha;
Results.nPerm = nPerm;
Results.model = 'EEG ~ Condition + (1|Class) + (1|Class:Student)';
Results.test = 'Treatment - Control fixed effect';
Results.permutation = 'Condition-label flipping within Class:Student';
Results.times = times;
Results.e_loc = e_loc;
Results.chanLabels = chanLabels;

fprintf('Results stored.\n');

%% ==========================================================
% Step 6: Plot significant observed t-values
%% ==========================================================

mT = t_Obs;
mT(~Mask) = 0;

figure;

imagesc(times, 1:nChan, mT);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', chanLabels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('TFCE-corrected Nested LME Effect: Treatment - Control');
colorbar;

%% ==========================================================
% Step 7: Plot observed TFCE map
%% ==========================================================

figure;

imagesc(times, 1:nChan, TFCE_Obs);
axis xy;
xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', chanLabels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed TFCE Map: Nested LME Treatment Effect');
colorbar;

%% ==========================================================
% Step 8: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/07_TFCE_nested_class_student_results.mat', ...
     'Results');

disp('Nested class-student TFCE analysis completed and saved.');