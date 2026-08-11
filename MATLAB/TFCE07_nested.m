
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
%   Condition is manipulated within each student nested
%   within class. Each Class x Student unit contains two
%   observations, one for each condition.
%
%   Under permutation, the two responses within each unit
%   are either retained or swapped with probability 0.5.
%
%   The same permutation is applied simultaneously to all
%   Channel x Time points.
%
% Multiple-comparison correction:
%
%   Maximum absolute TFCE statistic across all channels and
%   time points.
%% ==========================================================


clear all;
close all;
clc;

fprintf('\nStarting nested class-student TFCE analysis...\n');

rng(123);


%% ==========================================================
% Analysis settings
%% ==========================================================

nPerm = 999;
alpha = 0.05;

inputFile = '../Data/07_simulated_nested_class_student_EEG.mat';

outputFile = '../Results/07_TFCE_nested_class_student_results.mat';


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


%% ==========================================================
% Basic dimensions and checks
%% ==========================================================

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

times = -200:4:800;


%% ==========================================================
% Check time dimension
%% ==========================================================

if length(times) ~= nTime

    error( ...
        ['The number of values in times does not match ', ...
         'the time dimension of EEGdata.']);

end


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

    error( ...
        'Missing channels: %s', ...
        strjoin(chanLabels_32(~tf), ', '));

end

e_loc = chanlocs_1020(idx);


%% ==========================================================
% Check number of channels
%% ==========================================================

if length(e_loc) ~= nChan

    error( ...
        ['The number of selected channel locations does not ', ...
         'match the channel dimension of EEGdata.']);

end


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


%% ----------------------------------------------------------
% Construct model table once
%% ----------------------------------------------------------

tbl_obs = table( ...
    zeros(nObs, 1), ...
    Condition, ...
    ClassLME, ...
    StudentLME, ...
    'VariableNames', ...
    {'EEG', 'Condition', 'Class', 'Student'} ...
);


%% ----------------------------------------------------------
% Fit observed mixed-effects models
%% ----------------------------------------------------------

for ch = 1:nChan

    fprintf('Observed model: channel %d of %d\n', ch, nChan);

    for tp = 1:nTime

        % Replace response only
        tbl_obs.EEG = squeeze(EEGdata(:, ch,tp));

        % Fit nested mixed-effects model
        lme = fitlme(tbl_obs, ...
            'EEG ~ Condition + (1|Class) + (1|Class:Student)');

        % Extract t statistic for Condition
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
% Step 3: Generate null maximum-TFCE distribution
%% ==========================================================

fprintf('\nStep 3: Preparing permutations...\n');


% ==========================================================
% Identify Class x Student units
% ==========================================================

pairs = [Class(:), Student(:)];

[uniquePairs, ~, unitID] = unique( ...
    pairs, ...
    'rows', ...
    'stable' ...
);

nUnits = size(uniquePairs, 1);

fprintf( ...
    'Number of Class x Student units: %d\n', ...
    nUnits);


% ==========================================================
% Check that each student has exactly two observations
% ==========================================================

for u = 1:nUnits

    rows = find(unitID == u);

    if numel(rows) ~= 2

        error( ...
            ['Each Class x Student unit must contain exactly ', ...
             'two observations. Unit %d contains %d.'], ...
            u, ...
            numel(rows));

    end

end


%% ==========================================================
% Randomly determine which units are swapped
%
% Rows:
%   Class x Student units
%
% Columns:
%   permutations
%% ==========================================================

swapCondition = rand(nUnits, nPerm) < 0.5;


%% ==========================================================
% Generate sparse permutation matrices
%
% Each permutation matrix is:
%
%   nObs x nObs
%
% Within every Class x Student unit:
%
%   no swap:
%
%       [1 0
%        0 1]
%
%   swap:
%
%       [0 1
%        1 0]
%% ==========================================================

PermutationMatrix = cell(nPerm, 1);

for p = 1:nPerm

    P = sparse(nObs, nObs);

    for u = 1:nUnits

        rows = find(unitID == u);

        if swapCondition(u, p)

            % Swap responses within the student

            P(rows(1), rows(2)) = 1;
            P(rows(2), rows(1)) = 1;

        else

            % Keep responses in their original order

            P(rows(1), rows(1)) = 1;
            P(rows(2), rows(2)) = 1;

        end

    end

    PermutationMatrix{p} = P;

end

fprintf('Permutation matrices created.\n');


%% ==========================================================
% Preallocate permutation results
%% ==========================================================

TFCE_permMax = zeros(nPerm, 1);


%% ==========================================================
% Start parallel pool
%
% Use an existing pool if one is already running.
%
% Six workers are used here as a moderate default because
% each worker repeatedly fits mixed-effects models.
%% ==========================================================

poolobj = gcp('nocreate');

if isempty(poolobj)

    fprintf('\nStarting parallel pool...\n');

    parpool('local', 6);

else

    fprintf( ...
        '\nUsing existing parallel pool with %d workers.\n', ...
        poolobj.NumWorkers);

end


%% ==========================================================
% Apply permutation matrices
%% ==========================================================

fprintf('\nStep 3: Permutation test has started...\n');


parfor p = 1:nPerm

    %% ------------------------------------------------------
    % Apply permutation to the entire EEG dataset
    %
    % This replaces:
    %
    %   PermutationMatrix{p} * EEG
    %
    % separately for every Channel x Time point.
    %
    % The permutation is now performed only once per
    % permutation.
    %% ------------------------------------------------------

    fprintf("At %d of %d permutation", p, nPerm);

    Condition_perm = PermutationMatrix{p} * Condition;


    %% ------------------------------------------------------
    % Initialize permutation t-map
    %% ------------------------------------------------------

    perm_t = zeros(nChan, nTime);


    %% ------------------------------------------------------
    % Construct mixed-model table once for this permutation
    %% ------------------------------------------------------

    tbl_perm = table( ...
        zeros(nObs, 1), ...
        Condition_perm, ...
        ClassLME, ...
        StudentLME, ...
        'VariableNames', ...
        {'EEG', 'Condition', 'Class', 'Student'} ...
    );


    %% ------------------------------------------------------
    % Fit permuted mixed-effects models
    %% ------------------------------------------------------

    for ch = 1:nChan

        for tp = 1:nTime

            % Replace response only
            tbl_perm.EEG = squeeze(EEGdata(:, ch,tp));

            % Fit nested mixed-effects model
            lme_perm = fitlme(tbl_perm, ...
                'EEG ~ Condition + (1|Class) + (1|Class:Student)');

            % Extract Condition t statistic
            perm_t(ch, tp) = lme_perm.Coefficients.tStat(2);

        end

    end


    %% ------------------------------------------------------
    % TFCE transformation
    %% ------------------------------------------------------

    TFCE_perm = ept_mex_TFCE2D( perm_t, ChN, E_H);


    %% ------------------------------------------------------
    % Maximum absolute TFCE statistic
    %% ------------------------------------------------------

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end


fprintf('Permutation testing completed.\n');


%% ==========================================================
% Step 4: Compute TFCE-corrected significance
%% ==========================================================

fprintf( ...
    '\nStep 4: Computing TFCE-corrected significance...\n');


%% ==========================================================
% Include observed assignment as the identity permutation
%
% Total reference distribution:
%
%   nPerm random permutations
%   +
%   1 observed assignment
%% ==========================================================

observedMaxTFCE = ...
    max(abs(TFCE_Obs(:)));

allMaxTFCE = sort( ...
    [observedMaxTFCE; TFCE_permMax] ...
);


%% ==========================================================
% Critical maximum-TFCE value
%% ==========================================================

nReference = nPerm + 1;

critIndex = ceil( ...
    (1 - alpha) * nReference ...
);

critIndex = min( ...
    max(critIndex, 1), ...
    nReference ...
);

maxTFCEcrit = ...
    allMaxTFCE(critIndex);


%% ==========================================================
% Significant TFCE mask
%% ==========================================================

Mask = ...
    abs(TFCE_Obs) >= maxTFCEcrit;


%% ==========================================================
% TFCE-corrected permutation p-values
%
% Monte-Carlo correction:
%
%   p = (b + 1) / (nPerm + 1)
%
% where b is the number of permutation maxima greater than
% or equal to the observed absolute TFCE statistic.
%% ==========================================================

P_Values = zeros(nChan, nTime);

for ch = 1:nChan

    for tp = 1:nTime

        P_Values(ch, tp) = ...
            ( ...
                sum( ...
                    TFCE_permMax >= ...
                    abs(TFCE_Obs(ch, tp)) ...
                ) ...
                + 1 ...
            ) ...
            / ...
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
Results.maxTFCEcrit  = maxTFCEcrit;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.alpha        = alpha;
Results.nPerm        = nPerm;

fprintf('Results stored.\n');


%% ==========================================================
% Step 6: Save results
%% ==========================================================

fprintf('\nStep 6: Saving results...\n');

if ~exist('../Results', 'dir')

    mkdir('../Results');

end

save( ...
    outputFile, ...
    'Results', ...
    'nChan', ...
    'times', ...
    'e_loc' ...
);

fprintf('Results saved to:\n%s\n', outputFile);


%% ==========================================================
% Step 7: Plot TFCE-corrected significant t-values
%% ==========================================================

sigT = Results.tObs;

sigT(~Results.Mask) = 0;

figure;

imagesc( ...
    times, ...
    1:nChan, ...
    sigT ...
);

axis xy;

xlim([-200 800]);

set( ...
    gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {e_loc.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);

xlabel('Time (ms)');
ylabel('Channel');

title( ...
    'TFCE-corrected Significant Effects' ...
);

cb = colorbar;

ylabel( ...
    cb, ...
    't-value' ...
);


%% ==========================================================
% Step 8: Plot observed TFCE values
%% ==========================================================

figure;

imagesc( ...
    times, ...
    1:nChan, ...
    Results.TFCE_Obs ...
);

axis xy;

xlim([-200 800]);

set( ...
    gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {e_loc.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);

xlabel('Time (ms)');
ylabel('Channel');

title('Observed TFCE Map');

cb = colorbar;

ylabel( ...
    cb, ...
    'TFCE value' ...
);


%% ==========================================================
% Analysis completed
%% ==========================================================

fprintf( ...
    '\nNested class-student TFCE analysis completed.\n' ...
);

