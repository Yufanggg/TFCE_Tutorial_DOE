%% ==========================================================
% TFCE analysis
% Fully crossed subject-item EEG design
%
% Input file contains:
%   EEGdata
%   designTable
%
% EEGdata long-format columns:
%   Subject
%   Item
%   Condition
%   ConditionCode      % -1 = Control, 1 = Treatment
%   Channel
%   Time
%   Amplitude
%
% designTable columns:
%   Subject
%   Item
%   Condition
%   ConditionCode
%
% Model:
%   EEG ~ Condition + (1|Subject) + (1|Item)
%
% Test:
%   Treatment - Control fixed effect
%
% Permutation:
%   Flip condition labels within each Subject × Item pair
%% ==========================================================

clear; clc; close all;

fprintf('\nStarting fully crossed subject-item TFCE analysis...\n');

rng(123);

%% ==========================================================
% Analysis settings
%% ==========================================================

nPerm = 999;
alpha = 0.05;

inputFile = ...
    '../Data/08_simulated_fully_crossed_subject_item_EEG.mat';

outputFile = ...
    '../Results/08_TFCE_fully_crossed_subject_item_results.mat';


%% ==========================================================
% Reconstruct EEG array
%
% EEGdata:
%   Observation x Channel x Time
%
% Observation = one Subject-Item-Condition row
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
Subject = designTable.Subject(:);
Item = designTable.Item(:);
CondCode = designTable.ConditionCode(:);

% Effect coding:
%
%   Control   = -1
%   Treatment =  1
Condition = double(designTable.ConditionCode(:));

SubjectLME = nominal(Subject);
ItemLME = nominal(Item);

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
% Step 0: Get mEEG
%% ==========================================================
mEEG = nan(size(EEGdata));
for ch = 1:nChan
    for tp = 1:nTime
        
        EEG = double(squeeze(EEGdata(:, ch,tp)));
        
        tbl = table(EEG, CondCode, SubjectLME, ItemLME, ...
            'VariableNames', {'EEG','Condition','Subject','Item'});
        
        lme = fitlme(tbl, ...
            'EEG ~ Condition + (1|Subject) + (1|Item)');

        mEEG(:, ch,tp) = fitted(lme,'Conditional',0) + residuals(lme);        
    end
end
disp('the marginalization stage is done!!!!!!!!!!');

clear EEG
%% ==========================================================
% Step 1: Observed LME t-map
%% ==========================================================

fprintf('Step 1: Computing observed t-statistic map...\n');

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        EEG = double(squeeze(mEEG(:, ch,tp)));

        tbl = table(EEG, CondCode, SubjectLME, ItemLME, ...
            'VariableNames', {'EEG','Condition','Subject','Item'});

        lm_local = fitlm(tbl, 'EEG ~ Condition');

        % Row 2 = Treatment - Control fixed effect
        t_Obs(ch,tp) = lm_local.Coefficients.tStat(2);

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
%% ==========================================================
TFCE_permMax = nan(nPerm,1);

fprintf('Step 3: Starting permutation test: %d permutations...\n', nPerm);

for p =1:nPerm
    permT = nan(nChan, nTime);
    
    permCondCode = CondCode(randperm(nRows),:);
    
    for ch = 1:nChan
        
        parfor tp = 1:nTime
            
            EEG = squeeze(mEEG(:, ch,tp));
            lm_perm = fitlm(permCondCode, EEG);
            
            % Coefficient 2 = permuted Group effect
            permT(ch, tp) = lm_perm.Coefficients.tStat(2);

        end
    end
    
    fprintf('At the %dth permutation\n', p);

    TFCE_perm = ept_mex_TFCE2D(permT, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));
end

fprintf('Permutation testing completed.\n');

%% ==========================================================
% Step 4: TFCE correction
%% ==========================================================

fprintf('Step 4: Computing TFCE-corrected significance...\n');

maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);

maxTFCEcrit = maxTFCE(round(nPerm*(1-Alpha)));

Mask = abs(TFCE_Obs) >= critTFCE;

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime
        P_Values(ch, tp) = ...
            (sum(TFCE_permMax >= abs(TFCE_Obs(ch, tp))) + 1) / ...
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

Results.tObs = t_Obs;
Results.TFCE_Obs = TFCE_Obs;
Results.TFCE_Null = TFCE_permMax;
Results.maxTFCEcrit = maxTFCEcrit;
Results.P_Values = P_Values;
Results.Mask = Mask;
Results.alpha = alpha;
Results.nPerm = nPerm;
Results.model = 'EEG ~ Condition + (1|Subject) + (1|Item)';
Results.test = 'Treatment - Control fixed effect';
Results.permutation = 'Condition-label flipping within Subject:Item';

fprintf('Results stored.\n');

%% ==========================================================
% Step 8: Save results
%% ==========================================================

if ~exist('../Results', 'dir')
    mkdir('../Results');
end

save(outputFile, ...
     'Results', ...
     'nChan', ...
     'times', ...
     'e_loc');

disp('Fully crossed subject-item TFCE analysis completed and saved.');

%% ==========================================================
% Step 6: Plot significant observed t-values
%% ==========================================================
clear all; clc; close all

load('../Results/08_TFCE_fully_crossed_subject_item_results.mat')

mT = Results.tObs;
mT(~Results.Mask) = 0;

figure;

imagesc(times, 1:nChan, mT);
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
title('TFCE-corrected Fully Crossed LME Effect: Treatment - Control');
colorbar;

%% ==========================================================
% Step 7: Plot observed TFCE map
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
title('Observed TFCE Map: Fully Crossed LME Treatment Effect');
colorbar;


