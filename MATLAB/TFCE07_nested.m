%% ==========================================================
% TFCE analysis
% Nested within-student design
%
% Input file contains:
%   EEGdata
%   designTable
%
% EEGdata dimensions:
%   Observation x Channel x Time
%
% Example:
%   120 x 32 x 351
%
% designTable columns:
%   Class
%   Student
%   Condition
%   ConditionCode      % -1 = Control, 1 = Treatment
%
% Model:
%   EEG ~ Condition + (1|Class) + (1|Class:Student)
%
% Condition coding:
%   Control   = -1
%   Treatment =  1
%
% Test:
%   Treatment - Control fixed effect
%
% With effect coding {-1, 1}, the fixed-effect coefficient is
% one-half of the Treatment - Control mean difference:
%
%   Treatment - Control = 2 * beta_Condition
%
% The t-statistic and p-value test the same null hypothesis:
%
%   H0: beta_Condition = 0
%
% Permutation:
%   Recursive multi-block permutation
%
% Hierarchy:
%   Class
%       Student nested within Class
%           Condition observations
%% ==========================================================

clear all;
clc;
close all;

fprintf('\nStarting nested TFCE analysis...\n');

rng(123);

%% ==========================================================
% Analysis settings
%% ==========================================================

nPerm = 999;
alpha = 0.05;

inputFile = ...
    '../data/07_simulated_nested_class_student_EEG.mat';

outputFile = ...
    '../results/07_TFCE_nested_class_student_results.mat';


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
% Check designTable variables
%% ==========================================================

requiredVariables = {
    'Class'
    'Student'
    'Condition'
    'ConditionCode'
    };

missingVariables = setdiff( ...
    requiredVariables, ...
    designTable.Properties.VariableNames);

if ~isempty(missingVariables)

    error( ...
        'designTable is missing variables: %s', ...
        strjoin(missingVariables, ', '));

end


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


%% ==========================================================
% Verify condition coding
%% ==========================================================

conditionValues = unique(Condition);

if ~isequal(sort(conditionValues(:)), [-1; 1])

    error( ...
        ['ConditionCode must contain exactly -1 and 1. ', ...
         'Observed values: %s'], ...
        mat2str(conditionValues'));

end


%% ==========================================================
% Construct nested hierarchy
%
% Level 1:
%   Class
%
% Level 2:
%   Student nested within Class
%
% Leaf level:
%   Condition observations
%% ==========================================================

%% ==========================================================
% Construct numeric block identifiers without findgroups
%% ==========================================================

% Class-level block ID
[~, ~, ClassID] = unique( ...
    Class, ...
    'stable');

ClassID = ClassID(:);

% Student nested within Class block ID
classStudentPairs = [
    double(Class(:)), ...
    double(Student(:))
    ];

[~, ~, NestedStudentID] = unique( ...
    classStudentPairs, ...
    'rows', ...
    'stable');

NestedStudentID = NestedStudentID(:);

blockHierarchy = [
    ClassID, ...
    NestedStudentID
    ];

nestedUnits = unique( ...
    NestedStudentID, ...
    'stable');

nUnits = numel(nestedUnits);
nClasses = numel(unique(ClassID));


%% ==========================================================
% Check nested design
%% ==========================================================

if nObs ~= nUnits * 2

    error( ...
        ['Expected exactly two condition observations for ', ...
         'each student nested within class.']);

end

for u = 1:nUnits

    rows = NestedStudentID == nestedUnits(u);

    if sum(rows) ~= 2

        error( ...
            ['Nested student unit %d does not contain ', ...
             'exactly two observations.'], ...
            u);

    end

    unitCondition = sort(Condition(rows));

    if ~isequal(unitCondition(:), [-1; 1])

        error( ...
            ['Nested student unit %d must contain one ', ...
             'Control observation (-1) and one ', ...
             'Treatment observation (1).'], ...
            u);

    end

end

fprintf( ...
    ['Design check passed: %d classes, %d nested students, ', ...
     'and two conditions per student.\n'], ...
    nClasses, ...
    nUnits);


%% ==========================================================
% Validate recursive block structure
%
% Blocks permuted at the same hierarchy level must have
% compatible subtree structures.
%% ==========================================================

% check_recursive_block_structure(blockHierarchy);
% 
% fprintf( ...
%     'Recursive hierarchy contains %d block levels.\n', ...
%     size(blockHierarchy, 2));


%% ==========================================================
% Time vector
%% ==========================================================

if isfield(S, 'times')

    times = S.times(:)';

else

    times = 1:nTime;

    warning( ...
        ['No times variable was found in the input file. ', ...
         'Using sample indices 1:%d.'], ...
        nTime);

end

if numel(times) ~= nTime

    error( ...
        ['The number of time values must equal the third ', ...
         'dimension of EEGdata.']);

end


%% ==========================================================
% Channel labels
%% ==========================================================

if isfield(S, 'chanLabels')

    chanLabels = cellstr(string(S.chanLabels));

elseif isfield(S, 'e_loc')

    chanLabels = {S.e_loc.labels};

else

    chanLabels = arrayfun( ...
        @(x) sprintf('Channel%d', x), ...
        1:nChan, ...
        'UniformOutput', ...
        false);

    warning( ...
        ['No chanLabels or e_loc variable was found. ', ...
         'Generic channel labels will be used.']);

end

if numel(chanLabels) ~= nChan

    error( ...
        ['The number of channel labels must equal the second ', ...
         'dimension of EEGdata.']);

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
% Mixed-effects model
%% ==========================================================

modelFormula = ...
    'EEG ~ Condition + (1|Class) + (1|Class:Student)';


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

        EEG = double( ...
            squeeze(EEGdata(:, ch, tp)));

        tbl = table( ...
            EEG, ...
            Condition, ...
            ClassLME, ...
            StudentLME, ...
            'VariableNames', ...
            {'EEG', 'Condition', 'Class', 'Student'});
        
        lme = fitlme( ...
            tbl, ...
            modelFormula);
        coefficientNames = ...
            lme.Coefficients.Properties.RowNames;
        
        conditionRow = strcmp( ...
                coefficientNames, ...
                'Condition');
        if ~any(conditionRow)
            error( ...
                'The Condition coefficient was not found.');
        end
        
        t_Obs(ch, tp) = ...
                lme.Coefficients.tStat(conditionRow);
    
    end

end

fprintf('Observed t-statistic map completed.\n');

%% ==========================================================
% Step 2: Compute observed TFCE map
%% ==========================================================

fprintf('\nStep 2: Computing observed TFCE map...\n');

TFCE_Obs = ept_mex_TFCE2D( ...
    double(t_Obs), ...
    ChN, ...
    E_H);

fprintf('Observed TFCE map completed.\n');


%% ==========================================================
% Step 3: Recursive multi-block permutation
%
% At every permutation:
%
%   1. Complete class blocks are permuted.
%
%   2. Student blocks are recursively permuted within matched
%      class blocks.
%
%   3. The two condition observations are permuted within
%      matched student blocks.
%
% The design table remains fixed.
%
% EEG observations are assigned to the fixed design using:
%
%   EEGdata_perm = EEGdata(permIndex,:,:)
%% ==========================================================

fprintf( ...
    ['\nStep 3: Starting %d recursive multi-block ', ...
     'permutations...\n'], ...
    nPerm);

TFCE_permMax = nan(nPerm, 1);

% Reproducible random seed for each parfor iteration
permutationSeeds = randi( ...
    2^31 - 1, ...
    nPerm, ...
    1);


%% ==========================================================
% Permutation loop
%% ==========================================================

parfor p = 1:nPerm

    stream = RandStream( ...
    'mt19937ar', ...
    'Seed', ...
    permutationSeeds(p));

    %% ------------------------------------------------------
    % Generate recursive permutation index
    %
    % permIndex(targetRow) identifies the source observation
    % assigned to that target design row.
    %% ------------------------------------------------------

    permIndex = recursive_block_permutation( ...
        blockHierarchy, ...
        stream);

    %% ------------------------------------------------------
    % Apply the permutation to EEG observation rows
    %% ------------------------------------------------------

    EEGdata_perm = EEGdata( ...
        permIndex, ...
        :, ...
        :);

    perm_t = zeros(nChan, nTime);

    %% ------------------------------------------------------
    % Fit permuted mixed-effects models
    %% ------------------------------------------------------

    for ch = 1:nChan

        for tp = 1:nTime

            EEG = double( ...
                squeeze(EEGdata_perm(:, ch, tp)));

            tbl = table( ...
                EEG, ...
                Condition, ...
                ClassLME, ...
                StudentLME, ...
                'VariableNames', ...
                {'EEG', 'Condition', 'Class', 'Student'});
            
            lmePerm = fitlme( ...
                    tbl, ...
                    modelFormula);
            
            coefficientNames = ...
                lmePerm.Coefficients.Properties.RowNames;
            
            conditionRow = strcmp( ...
                coefficientNames, ...
                'Condition');
            
            
            perm_t(ch, tp) = ...
                    lmePerm.Coefficients.tStat(conditionRow);
            
        end

    end

    %% ------------------------------------------------------
    % TFCE transformation
    %% ------------------------------------------------------

    TFCE_perm = ept_mex_TFCE2D( ...
        double(perm_t), ...
        ChN, ...
        E_H);

    %% ------------------------------------------------------
    % Maximum absolute TFCE statistic
    %% ------------------------------------------------------

    TFCE_permMax(p) = ...
        max(abs(TFCE_perm(:)));

end

fprintf('Recursive multi-block permutation completed.\n');


%% ==========================================================
% Step 4: Compute TFCE-corrected significance
%% ==========================================================

fprintf( ...
    '\nStep 4: Computing TFCE-corrected significance...\n');


%% ==========================================================
% Observed maximum absolute TFCE
%% ==========================================================

observedMaxTFCE = ...
    max(abs(TFCE_Obs(:)));


%% ==========================================================
% Critical maximum-TFCE value
%
% The observed assignment is included as the identity
% permutation.
%% ==========================================================

maxTFCEdistribution = sort( ...
    [observedMaxTFCE; TFCE_permMax]);

nReference = numel(maxTFCEdistribution);

criticalIndex = ceil( ...
    (1 - alpha) * nReference);

criticalIndex = min( ...
    max(criticalIndex, 1), ...
    nReference);

maxTFCEcrit = ...
    maxTFCEdistribution(criticalIndex);


%% ==========================================================
% Corrected significance mask
%% ==========================================================

Mask = abs(TFCE_Obs) >= maxTFCEcrit;


%% ==========================================================
% Corrected p-values
%
% For every channel-time point:
%
%                  1 + number of permutation maxima
%                      >= observed absolute TFCE
%   corrected p = --------------------------------------
%                             nPerm + 1
%% ==========================================================

P_Values = zeros(nChan, nTime);

for ch = 1:nChan

    for tp = 1:nTime

        observedTFCE = ...
            abs(TFCE_Obs(ch, tp));

        P_Values(ch, tp) = ...
            (1 + sum( ...
                TFCE_permMax >= observedTFCE)) ...
            / (nPerm + 1);

    end

end

Mask_from_P = P_Values <= alpha;

fprintf('TFCE correction completed.\n');

fprintf( ...
    'Critical TFCE value: %.4f\n', ...
    maxTFCEcrit);

fprintf( ...
    'Number of significant channel-time points: %d\n', ...
    nnz(Mask));


%% ==========================================================
% Step 5: Store results
%% ==========================================================

fprintf('\nStep 5: Storing results...\n');

Results = struct();

Results.Obs = t_Obs;

% Estimated Condition coefficient.
%
% With coding {-1,1}, this is half of the mean difference.
Results.Beta_Obs = Beta_Obs;

% Full estimated Treatment - Control mean difference.
Results.TreatmentMinusControl = Difference_Obs;

Results.TFCE_Obs = TFCE_Obs;
Results.TFCE_Null = TFCE_permMax;

Results.observedMaxTFCE = observedMaxTFCE;
Results.maxTFCEdistribution = maxTFCEdistribution;
Results.critTFCE = maxTFCEcrit;

Results.P_Values = P_Values;
Results.Mask = Mask;
Results.Mask_from_P = Mask_from_P;

Results.alpha = alpha;
Results.nPerm = nPerm;

Results.model = modelFormula;

Results.test = ...
    'Treatment - Control fixed effect';

Results.ConditionCoding = ...
    'Control = -1; Treatment = 1';

Results.CoefficientInterpretation = ...
    ['beta_Condition equals one-half of the ', ...
     'Treatment - Control mean difference'];

Results.permutation = ...
    ['Recursive multi-block permutation of Class, ', ...
     'Student-within-Class, and Condition observations'];

Results.permutationHierarchy = ...
    'Class -> Student within Class -> Condition observation';

Results.blockHierarchy = blockHierarchy;
Results.Condition = Condition;

Results.times = times;
Results.chanLabels = chanLabels;
Results.e_loc = e_loc;
Results.ChN = ChN;
Results.E_H = E_H;

fprintf('Results stored.\n');


%% ==========================================================
% Step 6: Plot significant observed t-values
%% ==========================================================

mT = t_Obs;
mT(~Mask) = 0;

figure;

imagesc( ...
    times, ...
    1:nChan, ...
    mT);

axis xy;

if min(times) <= -200 && max(times) >= 800
    xlim([-200, 800]);
end

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', chanLabels, ...
    'TickLength', [0, 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');

title( ...
    ['TFCE-corrected nested LME effect: ', ...
     'Treatment - Control']);

colorbar;


%% ==========================================================
% Step 7: Plot Treatment - Control estimate
%% ==========================================================

mDifference = Difference_Obs;
mDifference(~Mask) = 0;

figure;

imagesc( ...
    times, ...
    1:nChan, ...
    mDifference);

axis xy;

if min(times) <= -200 && max(times) >= 800
    xlim([-200, 800]);
end

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', chanLabels, ...
    'TickLength', [0, 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');

title( ...
    ['TFCE-significant estimated mean difference: ', ...
     'Treatment - Control']);

colorbar;


%% ==========================================================
% Step 8: Plot observed TFCE map
%% ==========================================================

figure;

imagesc( ...
    times, ...
    1:nChan, ...
    TFCE_Obs);

axis xy;

if min(times) <= -200 && max(times) >= 800
    xlim([-200, 800]);
end

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', chanLabels, ...
    'TickLength', [0, 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');

title( ...
    'Observed TFCE map: nested LME treatment effect');

colorbar;


%% ==========================================================
% Step 9: Plot corrected p-value map
%% ==========================================================

figure;

imagesc( ...
    times, ...
    1:nChan, ...
    P_Values);

axis xy;

if min(times) <= -200 && max(times) >= 800
    xlim([-200, 800]);
end

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', chanLabels, ...
    'TickLength', [0, 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');

title('TFCE-corrected p-value map');

colorbar;
clim([0, 1]);


%% ==========================================================
% Step 10: Plot maximum-TFCE null distribution
%% ==========================================================

figure;

histogram( ...
    TFCE_permMax, ...
    30);

hold on;

xline( ...
    maxTFCEcrit, ...
    '--', ...
    'LineWidth', ...
    2, ...
    'Label', ...
    sprintf( ...
        'Critical value = %.4f', ...
        maxTFCEcrit));

xlabel('Maximum absolute TFCE');
ylabel('Frequency');

title( ...
    'Recursive multi-block maximum-TFCE distribution');

hold off;


%% ==========================================================
% Step 11: Save results
%% ==========================================================

outputDirectory = fileparts(outputFile);

if ~exist(outputDirectory, 'dir')

    mkdir(outputDirectory);

end

save( ...
    outputFile, ...
    'Results', ...
    '-v7.3');

fprintf( ...
    '\nResults saved to:\n%s\n', ...
    outputFile);

fprintf( ...
    '\nNested class-student TFCE analysis completed.\n');

