%% ==========================================================
% TFCE analysis for a nested within-student design
%
% Required input variables:
%
%   EEGdata
%       Numeric array with dimensions:
%       Observation x Channel x Time
%
%   designTable
%       MATLAB table containing:
%           Class
%           Student
%           Condition
%           ConditionCode
%
% Condition coding:
%   Control   = -1
%   Treatment =  1
%
% Mixed-effects model:
%
%   EEG ~ Condition + (1|Class) + (1|Class:Student)
%
% Hypothesis:
%
%   H0: beta_Condition = 0
%
% With effect coding {-1, 1}:
%
%   Treatment - Control = 2 * beta_Condition
%
% Permutation strategy:
%
%   Condition labels are permuted within each student nested
%   within class. Therefore, the class structure and the
%   repeated-measures structure are both preserved.
%
% Multiple-comparison correction:
%
%   Maximum absolute TFCE statistic across all channels and
%   time points.
%% ==========================================================

clear all; close all; clc

fprintf('\nStarting nested class-student TFCE analysis...\n');

rng(123);

%% ==========================================================
% Analysis settings
%% ==========================================================

nPerm = 999;
alpha = 0.05;

inputFile = ...
    '../Data/07_simulated_nested_class_student_EEG.mat';

outputFile = ...
    '../Results/07_TFCE_nested_class_student_results.mat';



%% ==========================================================
% Load data
%% ==========================================================

S = load(inputFile);

if ~isfield(S, 'EEGdata')

    error('The input file does not contain EEGdata.');

end

if ~isfield(S, 'designTable')

    error('The input file does not contain designTable.');

end

EEGdata = S.EEGdata;
designTable = S.designTable;

[nObs, nChan, nTime] = size(EEGdata);

if height(designTable) ~= nObs

    error( ...
        ['The number of designTable rows must equal the ', ...
         'first dimension of EEGdata.']);

end

fprintf( ...
    ['EEGdata dimensions: %d observations x ', ...
     '%d channels x %d time points.\n'], ...
    nObs, ...
    nChan, ...
    nTime);

fprintf( ...
    'designTable dimensions: %d rows x %d variables.\n', ...
    height(designTable), ...
    width(designTable));


%% ==========================================================
% Extract design variables
%% ==========================================================

Class = designTable.Class(:);
Student = designTable.Student(:);

% Effect coding:
%
%   Control   = -1
%   Treatment =  1
Condition = double(designTable.ConditionCode(:));

ClassLME = categorical(Class);
StudentLME = categorical(Student);

times = -200:4:800

%% ==========================================================
% Load channel locations
%% ==========================================================
chanlocs_1020 = readlocs('standard_1005.elc');

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

allLabels = {chanlocs_1020.labels};
[tf, idx] = ismember(chanLabels_32, allLabels);

if any(~tf)
    error('Missing channels: %s', strjoin(chanLabels_32(~tf), ', '));
end

e_loc = chanlocs_1020(idx);

%% ==========================================================
% Channel-neighbour structure and TFCE parameters
%% ==========================================================

ChN = ept_ChN2(e_loc);

% TFCE parameters:
%
%   E = extent exponent
%   H = height exponent
E_H = [0.66, 2];

%% ==========================================================
% Step 1: Compute observed LME t-map
%% ==========================================================
fprintf('\nStep 1: Computing observed t-statistic map...\n');

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan

    fprintf( ...
        'Observed model: channel %d of %d\n', ...
        ch, ...
        nChan);

    for tp = 1:nTime

        EEG = double(squeeze(EEGdata(:, ch, tp)));

        tbl = table(EEG, Condition, ClassLME, ...
            StudentLME, 'VariableNames', ...
            {'EEG', 'Condition', 'Class', 'Student'});
        
        lme = fitlme( tbl, 'EEG ~ Condition + (1|Class) + (1|Class:Student)');
        t_Obs(ch, tp) = lme.Coefficients.tStat(2);
    
    end

end

fprintf('Observed t-statistic map completed.\n');

%% ==========================================================
% Step 2: Compute observed TFCE map
%% ==========================================================

fprintf('\nStep 2: Computing observed TFCE map...\n');

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');

%% ==========================================================
% Step 3: get the null maximum TFCE distribution
%% ==========================================================
%% ==========================================================
% Generate permutation instructions
%% ==========================================================

pairs = [Class, Student];

[uniquePairs, ~, unitID] = unique( pairs, 'rows', 'stable');

nUnits = size(uniquePairs, 1);

swapCondition = rand(length(uniquePairs), nPerm) < 0.5;

TFCE_permMax = nan(nPerm, 1);

%% ==========================================================
% Permutation loop
%% ==========================================================
%% ==========================================================
% Identify class-student units
%% ==========================================================

pairs = [ ...
    Class(:), ...
    Student(:)];

[uniquePairs, ~, unitID] = unique( ...
    pairs, ...
    'rows', ...
    'stable');

nUnits = size(uniquePairs,1);

%% ==========================================================
% Randomly decide which units are swapped
%% ==========================================================

swapCondition = ...
    rand(nUnits,nPerm) < 0.5;

%% ==========================================================
% Generate permutation matrices
%% ==========================================================

PermutationMatrix = cell(nPerm,1);

for p = 1:nPerm

    P = sparse( ...
        nObs, ...
        nObs);

    for u = 1:nUnits

        rows = find(unitID == u);

        if numel(rows) ~= 2

            error( ...
                'Each unit must contain exactly two observations.');

        end

        if swapCondition(u,p)

            %----------------------------------------------
            % Swap
            %
            % [0 1
            %  1 0]
            %----------------------------------------------

            P(rows(1),rows(2)) = 1;
            P(rows(2),rows(1)) = 1;

        else

            %----------------------------------------------
            % Identity
            %
            % [1 0
            %  0 1]
            %----------------------------------------------

            P(rows(1),rows(1)) = 1;
            P(rows(2),rows(2)) = 1;

        end

    end

    PermutationMatrix{p} = P;

end



parfor p = 1:nPerm

    %% ------------------------------------------------------
    % Construct the sparse permutation matrix
    %% ------------------------------------------------------

    %% ------------------------------------------------------
    % Permute responses while keeping the design fixed
    %% ------------------------------------------------------

    perm_t = zeros(nChan, nTime);

    %% ------------------------------------------------------
    % Fit permuted mixed-effects models
    %% ------------------------------------------------------

    for ch = 1:nChan

        for tp = 1:nTime

            EEG = double(squeeze(EEGdata(:, ch, tp)));
            
            EEG_perm =  PermutationMatrix{p} * EEG;
            
            tbl = table(EEG_perm, Condition, ClassLME, ...
                StudentLME, 'VariableNames', ...
                {'EEG', 'Condition', 'Class', 'Student'});
            
            lme_perm = fitlme( tbl, 'EEG ~ Condition + (1|Class) + (1|Class:Student)');
            perm_t(ch, tp) = lme_perm.Coefficients.tStat(2);

        end

    end

    %% ------------------------------------------------------
    % TFCE transformation and maximum statistic
    %% ------------------------------------------------------

    TFCE_perm = ept_mex_TFCE2D( ...
        perm_t, ...
        ChN, ...
        E_H);

    TFCE_permMax(p) = ...
        max(abs(TFCE_perm(:)));

end

fprintf('permutation completed.\n');

%% ==========================================================
% Step 4: Compute TFCE-corrected significance
%% ==========================================================

fprintf( ...
    '\nStep 4: Computing TFCE-corrected significance...\n');


%% ==========================================================
% Critical maximum-TFCE value
%
% The observed assignment is included as the identity
% permutation.
%% ==========================================================

maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);

maxTFCEcrit = maxTFCE(round(nPerm*(1-alpha)));

Mask = abs(TFCE_Obs) >= maxTFCEcrit;

P_Values = zeros(nChan, nTime);

for ch = 1:nChan

    for tp = 1:nTime

        P_Values(ch,tp) = ...
            (sum(TFCE_permMax >= abs(TFCE_Obs(ch,tp))) + 1) / ...
            (nPerm + 1);

    end

end

fprintf('TFCE correction completed.\n');

fprintf( ...
    'Critical TFCE value: %.4f\n', ...
    maxTFCEcrit);

%% ==========================================================
% Step 5: Store results
%% ==========================================================

fprintf('\nStep 5: Storing results...\n');

Results = struct();

Results.tObs         = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_Null    = TFCE_permMax;
Results.maxTFCEcrit     = maxTFCEcrit;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.alpha        = alpha;
Results.nPerm        = nPerm;


fprintf('Results stored.\n');

%% ==========================================================
% Step 11: Save results
%% ==========================================================

if ~exist('../Results', 'dir')
    mkdir('../Results');
end

save(outputFile, ...
     'Results', ...
     'nChan', ...
     'times', ...
     'e_loc');


fprintf('\nNested class-student TFCE analysis completed.\n');


% Step 6: Plot TFCE-corrected significant t-values
%% ==========================================================

sigT = Results.tObs;
sigT(~Results.Mask) = 0;

figure;

imagesc(-200:4:800, 1:nChan, sigT);
axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {e_loc.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('TFCE-corrected Significant Effects');

colorbar;

%% ==========================================================
% Step 7: Plot TFCE values
%% ==========================================================

figure;

imagesc(-200:4:800, 1:nChan, Results.TFCE_Obs);
axis xy;

%xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {e_loc.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed TFCE Map');

colorbar;


