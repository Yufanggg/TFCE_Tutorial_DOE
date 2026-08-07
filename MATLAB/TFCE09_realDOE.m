%% ==========================================================
% lmeEEG TFCE analysis using the Freedman-Lane procedure
%
% EEGdata dimensions:
%   observations x channels x time
%
% designTable:
%   one row per observation
%
% Marginalization model:
%
%   EEG ~ CongruencySemanticCategories + Stroke + Freq +
%         JSD + Classifier + (1|Subj) + (1|Item)
%
% Test effect:
%   Classifier
%
% Freedman-Lane procedure:
%
%   1. Marginalize nuisance effects using the LME.
%   2. Fit the reduced target model without Classifier.
%   3. Permute reduced-model residuals.
%   4. Construct:
%
%        Y_perm = Yhat_reduced + permuted residuals
%
%   5. Fit the full target model to Y_perm.
%   6. Apply TFCE to observed and permutation statistics.
%% ==========================================================

clear; clc; close all;

rng(123);

fprintf('\nStarting lmeEEG Freedman-Lane analysis...\n');

%% ==========================================================
% Load prepared data
%% ==========================================================

load('../Data/realDOE.mat');

%% ==========================================================
% Check data dimensions
%% ==========================================================

[nObs, nChan, nTime] = size(EEGdata);

if height(designTable) ~= nObs
    error(['The number of rows in designTable must equal ' ...
           'the first dimension of EEGdata.']);
end

if length(channelinfo) ~= nChan
    error(['The number of channel locations does not match ' ...
           'the second dimension of EEGdata.']);
end

if exist('time', 'var') && length(time) ~= nTime
    error(['The length of time does not match the third ' ...
           'dimension of EEGdata.']);
end

fprintf('Observations: %d\n', nObs);
fprintf('Channels:     %d\n', nChan);
fprintf('Time points:  %d\n', nTime);

%% ==========================================================
% Prepare design variables
%% ==========================================================

variableNames = designTable.Properties.VariableNames;

% Subject identifier
if ismember('SubjID', variableNames)

    Subj = nominal(designTable.SubjID);

elseif ismember('SubjSubj', variableNames)

    Subj = nominal(designTable.SubjSubj);

else

    error(['designTable must contain a subject variable named ' ...
           'SubjID or SubjSubj.']);
end

% Item identifier
if ~ismember('Target', variableNames)
    error('designTable must contain the item variable Target.');
end

Item = nominal(designTable.Target);

% Predictors
% Classifier (-1 / +1 coding)
JSD_cat = categorical(designTable.JSD);

levels = categories(JSD_cat);

if numel(levels) ~= 2
    error('JSD must contain exactly two levels.');
end

JSD = -1 * ones(height(designTable),1);
JSD(JSD_cat == levels{2}) = 1;

% Classifier (-1 / +1 coding)
Classifier_cat = categorical(designTable.ClassifierCongruency);

levels = categories(Classifier_cat);

if numel(levels) ~= 2
    error('Classifier must contain exactly two levels.');
end

Classifier = -1 * ones(height(designTable),1);
Classifier(Classifier_cat == levels{2}) = 1;


Freq = log( ...
    double(designTable.Frequency) ...
);

StrokeRaw = double(designTable.NumbersofStorks);

Stroke = ...
    (StrokeRaw - mean(StrokeRaw)) ./ std(StrokeRaw);

% CongruencySemanticCategories (-1 / +1 coding)

CongruencySemanticCategories_cat = categorical( ...
    designTable.CongruencySemanticCategories ...
);

levels = categories(CongruencySemanticCategories_cat);

if numel(levels) ~= 2
    error('CongruencySemanticCategories must contain exactly two levels.');
end

CongruencySemanticCategories = -1 * ones(height(designTable),1);

CongruencySemanticCategories( ...
    CongruencySemanticCategories_cat == levels{2} ...
) = 1;

%% ==========================================================
% Check predictor lengths and missing values
%% ==========================================================

designLengths = [
    length(Subj), ...
    length(Item), ...
    length(JSD), ...
    length(Classifier), ...
    length(Freq), ...
    length(Stroke), ...
    length(CongruencySemanticCategories)
];

if any(designLengths ~= nObs)
    error('Every design variable must contain one value per observation.');
end

if any(isnan(Freq))
    error('Freq contains missing or undefined values.');
end

if any(isnan(Stroke))
    error('Stroke contains missing or undefined values.');
end

%% ==========================================================
% Create output directory
%% ==========================================================

outputDirectory = '.\Output';

if ~exist(outputDirectory, 'dir')
    mkdir(outputDirectory);
end

%% ==========================================================
% Step 1: Marginalize nuisance fixed effects
%
% Retain:
%   Intercept
%   Classifier effect
%   Marginal residuals
%
% Remove:
%   CongruencySemanticCategories
%   Stroke
%   Freq
%   JSD
%
% mEEG dimensions:
%   observations x channels x time
%% ==========================================================

fprintf('\nStep 1: Marginalizing nuisance effects...\n');

mEEG = nan(size(EEGdata));

fullMarginalFormula = [ ...
    'EEG ~ CongruencySemanticCategories + Stroke + Freq + ' ...
    'JSD + Classifier + (1|Subj) + (1|Item)' ...
];

for ch = 1:nChan

    fprintf('Marginalization: channel %d of %d\n', ch, nChan);

    for tpoint = 1:nTime

        Y = double(squeeze(EEGdata(:, ch, tpoint)));

        modelTable = table( Y, CongruencySemanticCategories, ...
            Stroke, Freq, JSD, Classifier, Subj, Item, ...
            'VariableNames', { 'EEG', 'CongruencySemanticCategories', ...
                'Stroke', 'Freq', 'JSD', 'Classifier', 'Subj', ...
                'Item' } );

        lme = fitlme(modelTable, fullMarginalFormula);

        %% --------------------------------------------------
        % Marginal fitted response plus residuals
        %% --------------------------------------------------

        marginalResponse = fitted(lme, 'Conditional', 0) + ...
            residuals(lme);

        mEEG(:, ch, tpoint) = marginalResponse;

    end
end

fprintf('Marginalization completed.\n');

%% ==========================================================
% Construct the target design matrix
%
% Full target model:
%   marginalized EEG ~ CongruencySemanticCategories + Stroke + Freq + 
%                      JSD + Classifier 
%
% Random effects were handled during marginalization.
% The following is about using Freedman-Lane procedure
%% ==========================================================
X_full = [CongruencySemanticCategories, Stroke, Freq, JSD, Classifier];

X_null = [CongruencySemanticCategories, Stroke, Freq, JSD];


%% ==========================================================
% Step 2: Observed statistics
%
% Store:
%   observed t-values
%% ==========================================================

fprintf('\nStep 2: Computing observed statistics...\n');

t_Obs = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        Y = double(mEEG(:, ch, tpoint));

        lm_local = fitlm(X_full, Y);
        
        t_Obs(ch, tpoint) = lm_local.Coefficients.tStat(6);

    end
end

fprintf('Observed t-statistic map completed.\n');

%% ==========================================================
% Step 3: Observed TFCE map
%% ==========================================================

fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(channelinfo);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');
%% ==========================================================
% Step 4: Freedman-Lane permutation
%% ==========================================================

nPerm = 999;
TFCE_permMax = nan(nPerm, 1);

fprintf('Step 3: Starting permutation testing: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    fprintf('The %dth permutation from %d started\n', p, nPerm);

    perm_t_local = nan(nChan, nTime);

    perm_idx = randperm(nObs);

    for ch = 1:nChan
        for tpoint = 1:nTime

            Y = double(mEEG(:, ch, tpoint));

            % Reduced model: EEG ~ CongruencySemanticCategories + Stroke + Freq + JSD
            lm_red = fitlm(X_null, Y);

            Y_hat_red = lm_red.Fitted;
            resid_red = lm_red.Residuals.Raw;

            % Freedman-Lane permuted response
            Y_perm = Y_hat_red + resid_red(perm_idx);

            % Full model on permuted data:
            % EEG_perm ~ Covariate + Group
            lm_perm = fitlm(X_full, Y_perm);

            % Column 3 = Group coefficient
            perm_t_local(ch, tpoint) = lm_perm.Coefficients.tStat(6);

        end
    end

    fprintf('At the %dth permutation\n', p);

    TFCE_perm = ept_mex_TFCE2D(perm_t_local, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end

fprintf('\nPermutation testing completed.\n');

%% ==========================================================
% Step 5: TFCE-corrected significance
%% ==========================================================

fprintf('Step 5: Computing TFCE-corrected significance...\n');

alpha = 0.05;
nPerm = length(TFCE_permMax);
maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);
maxTFCEcrit = maxTFCE(round(nPerm*(1-alpha)));

Mask = abs(TFCE_Obs) >= maxTFCEcrit;

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        P_Values(ch, tpoint) = ...
            (sum(TFCE_permMax >= abs(TFCE_Obs(ch, tpoint))) + 1) / ...
            (nPerm + 1);

    end
end

fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', maxTFCEcrit);

%% ==========================================================
% Step 6: Store results
%% ==========================================================

fprintf('Step 5: Storing results...\n');

Results = struct();

Results.tObs       = t_Obs;
Results.TFCE_Obs  = TFCE_Obs;
Results.TFCE_Null = TFCE_permMax;
Results.maxTFCEcrit  = maxTFCEcrit;
Results.P_Values  = P_Values;
Results.Mask      = Mask;
Results.alpha     = alpha;
Results.nPerm     = nPerm;
Results.model     = 'EEG ~ CongruencySemanticCategories + Stroke + Freq + JSD + Classifier';
Results.test      = 'Classifier effect adjusted for CongruencySemanticCategories, Stroke, Freq and JSD';

fprintf('Results stored.\n');

%% ==========================================================
% Step 7: Save TFCE results
%% ==========================================================

if ~exist('../Results', 'dir')
    mkdir('../Results');
end

save('../Results/09_realDOE_results.mat', ...
     'Results', ...
     'nChan', ...
     'time', ...
     'channelinfo');

disp('TFCE covariate-adjusted between-subject analysis completed.');
disp('Saved results: ../Results/09_realDOE_results.mat');

%% ==========================================================
% Step 8: Plot significant effects
%% ==========================================================

clear all; close all; clc

load('../Results/09_realDOE_results.mat')
sigT = Results.tObs;
sigT(~Results.Mask) = 0;

figure;

imagesc(time, 1:nChan, sigT);
axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {channelinfo.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('TFCE-corrected Significant Effects');

cb = colorbar;
ylabel(cb, 't-value', ...
    'FontSize', 15, ...
    'FontName', 'Arial');

%% ==========================================================
% Step 9: Plot TFCE values
%% ==========================================================

figure;

imagesc(time, 1:nChan, Results.TFCE_Obs);
axis xy;

xlim([-200 800]);

set(gca, ...
    'YTick', 1:nChan, ...
    'YTickLabel', {channelinfo.labels}, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

xlabel('Time (ms)');
ylabel('Channel');
title('Observed TFCE Map');

cb = colorbar;
ylabel(cb, 'TFCE-value', ...
    'FontSize', 15, ...
    'FontName', 'Arial');
