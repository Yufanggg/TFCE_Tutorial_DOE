
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

load( '../Data/07_simulated_nested_class_student_EEG.mat');

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

    error(['The number of selected channel locations does not ', ...
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
tObs = nan(nChan, nTime);

% preparation
pairs = [Class(:), Student(:)];
[uniquePairs, ~, unitID] = unique(pairs, 'rows', 'stable');
nUnits = size(uniquePairs, 1);
D = nan(nUnits, nChan, nTime);
fprintf('Number of Class x Student units: %d\n', nUnits);


for ch = 1:nChan

    fprintf('Observed model: channel %d of %d\n', ch, nChan);

    for tp = 1:nTime
        % Calculate the within-student condition contrasts
        for u = 1:nUnits
            rowsPlus = unitID == u & Condition == 1;
            rowsMinus = unitID == u & Condition == -1;
            % Retain students observed under both conditions
            if any(rowsPlus) && any(rowsMinus)
                D(:, ch, tp) = mean(Y(rowsPlus), 'omitnan') - ...
                    mean(Y(rowsMinus), 'omitnan');
            end
        end
        tObs(ch, tp) = mean(D(:, ch, tp)) / (std(D(:, ch, tp), 0) / sqrt(nUnits));
    end
end

fprintf('Observed t-statistic map completed.\n');

%% ==========================================================
% Step 2: Compute observed TFCE map
%% ==========================================================

fprintf('\nStep 2: Computing observed TFCE map...\n');

TFCE_Obs = ept_mex_TFCE2D(tObs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');

%% ==========================================================
% Step 3: Generate null TFCE distribution
%% ==========================================================

fprintf('\n Permutation test has started...\n');
TFCE_permMaxPos = nan(nPerm, 1);
TFCE_permMinNeg = nan(nPerm, 1);

parfor p = 1:nPerm
    % One sign-randomization iteration
    signs = 2 * randi([0, 1], nUnits, 1) - 1;
    tPerm = nan(nChan, nTime);
    for ch = 1:nChan
        for tp = 1:nTime
            D_perm = D(:, ch, tp) .* signs;
            % Permuted one-sample t statistic
            tPerm(ch, tp)= mean(D_perm) / (std(D_perm, 0) / sqrt(nUnits));
        end
    end 
    
    % ------------------------------------------------------
    % TFCE transformation
    % ------------------------------------------------------

    TFCE_perm = ept_mex_TFCE2D(tPerm, ChN, E_H);
    
    % ------------------------------------------------------
    % Maximum or minmum TFCE statistic
    % ------------------------------------------------------

    TFCE_permMaxPos(p) = max(TFCE_perm(:));
    TFCE_permMinNeg(p) = min(TFCE_perm(:));
    progressbar(p/nPerm)

end

fprintf('Permutation testing completed.\n');

%% ==========================================================
% Step 4: Compute TFCE-corrected significance
%% ==========================================================

fprintf('\nStep 4: Computing TFCE-corrected significance...\n');

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime      
        pr = (1 + sum(maxTFCEPos >= TFCE_Obs(ch,tp))) / (nPerm + 1);
        pl = (1 + sum(minTFCENeg <= TFCE_Obs(ch,tp))) / (nPerm + 1);        
        P_Values(ch,tp) = min(1, 2 * min(pr, pl));
    end

end

fprintf('TFCE-corrected significance completed.\n');

%% ==========================================================
% Step 5: Store results
%% ==========================================================

fprintf('\nStep 5: Storing results...\n');

Results = struct();

Results.tObs         = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.maxTFCEPos   = maxTFCEPos;
Results.minTFCENeg   = minTFCENeg;
Results.P_Values     = P_Values;
Results.nPerm        = nPerm;

fprintf('Results stored.\n');

%% ==========================================================
% Step 6: Save results
%% ==========================================================

fprintf('\nStep 6: Saving results...\n');

if ~exist('../Results', 'dir')

    mkdir('../Results');

end

save(outputFile, 'Results', 'nChan', 'times', 'e_loc');

fprintf('Results saved to:\n%s\n', outputFile);

%% ==========================================================
% Step 7: Plot TFCE-corrected significant t-values
%% ==========================================================
clear all; clc; close all

load('../Results/07_TFCE_nested_class_student_results.mat')

sigT = Results.tObs;

sigT(~Results.P_Values > 0.05) = 0;

figure;

imagesc(times, 1:nChan, sigT );

axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {e_loc.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);

xlabel('Time (ms)');
ylabel('Channel');

title('TFCE-corrected Significant Effects');

cb = colorbar;

ylabel(cb, 't-value');

%% ==========================================================
% Step 8: Plot observed TFCE values
%% ==========================================================

figure;

imagesc(times, 1:nChan, Results.TFCE_Obs);

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

title('Observed TFCE Map');

cb = colorbar;

ylabel(cb, 'TFCE value');

%% ==========================================================
% Analysis completed
%% ==========================================================

fprintf('\nNested class-student TFCE analysis completed.\n');

