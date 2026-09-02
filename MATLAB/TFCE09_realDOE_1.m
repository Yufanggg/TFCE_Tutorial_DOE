%% ==========================================================
% lmeEEG + Freedman-Lane + TFCE
%
% REAL DATA DIAGNOSTIC
%
% Primary test:
%   Classifier congruency
%
% Stage 1:
%   Remove subject/item random-intercept contributions using LME
%
% Stage 2:
%   Test Classifier effect using ordinary linear regression
%   and restricted Freedman-Lane permutations
%
% Model:
%
% EEG ~ Stroke + Freq + LengthofDistrctor +
%       CongruencySemanticCategories +
%       JSD + Classifier +
%       (1|Subj) + (1|Item)
%
% NOTE:
%   JSD x Classifier interaction is intentionally NOT included
%   in this diagnostic analysis.
%
%% ==========================================================

clear;
clc;
close all;

rng(123);

fprintf('\nStarting Classifier lmeEEG + Freedman-Lane analysis...\n');


%% ==========================================================
% Load data
%% ==========================================================

load('../Data/realDOE_3.mat');

[nObs,nChan,nTime] = size(EEGdata);


if height(designTable) ~= nObs
    error('designTable rows do not match EEG observations.');
end

if length(channelinfo) ~= nChan
    error('Channel information does not match EEG dimensions.');
end


fprintf('Observations: %d\n',nObs);
fprintf('Channels: %d\n',nChan);
fprintf('Time points: %d\n',nTime);


%% ==========================================================
% Prepare design variables
%% ==========================================================

varNames = designTable.Properties.VariableNames;


%% ----------------------------------------------------------
% Subject
%% ----------------------------------------------------------

if ismember('SubjID',varNames)

    Subj = categorical(designTable.SubjID);

elseif ismember('SubjSubj',varNames)

    Subj = categorical(designTable.SubjSubj);

else

    error('No subject identifier found.');

end


%% ----------------------------------------------------------
% Item
%% ----------------------------------------------------------

if ~ismember('Target',varNames)
    error('Target variable missing.');
end

Item = categorical(designTable.Target);


%% ==========================================================
% JSD
%
% Low  = -1
% High = +1
%% ==========================================================

JSD = zeros(nObs,1);

tmpJSD = string(designTable.JSD);

JSD(tmpJSD == "L") = -1;
JSD(tmpJSD == "H") =  1;


fprintf('JSD Low  (L, -1): %d observations\n', ...
    sum(JSD == -1));

fprintf('JSD High (H, +1): %d observations\n', ...
    sum(JSD == 1));


if any(JSD == 0)
    error('Some JSD observations were not coded.');
end


%% ==========================================================
% Classifier Congruency
%
% Congruent   = +1
% Incongruent = -1
%% ==========================================================

Classifier = zeros(nObs,1);

tmpClassifier = ...
    string(designTable.ClassifierCongruency);

Classifier(tmpClassifier == "Congruent") = 1;

Classifier(tmpClassifier == "Incongruent") = -1;


fprintf('Classifier Congruent (+1): %d observations\n', ...
    sum(Classifier == 1));

fprintf('Classifier Incongruent (-1): %d observations\n', ...
    sum(Classifier == -1));


if any(Classifier == 0)
    error('Some Classifier observations were not coded.');
end


%% ==========================================================
% Semantic-category congruency
%% ==========================================================

tmpSemantic = ...
    string(designTable.CongruencySemanticCategories);

disp('Semantic-category labels:');
disp(unique(tmpSemantic));


CongruencySemanticCategories = zeros(nObs,1);


% ----------------------------------------------------------
% Handle common Congruent / Incongruent coding
% ----------------------------------------------------------

CongruencySemanticCategories( ...
    tmpSemantic == "Congruent") = 1;

CongruencySemanticCategories( ...
    tmpSemantic == "Incongruent") = -1;


% ----------------------------------------------------------
% Also allow 1 / 0 labels
% ----------------------------------------------------------

CongruencySemanticCategories( ...
    tmpSemantic == "1") = 1;

CongruencySemanticCategories( ...
    tmpSemantic == "0") = -1;


if any(CongruencySemanticCategories == 0)

    error(['Some CongruencySemanticCategories observations ' ...
           'were not coded. Check the labels printed above.']);

end


fprintf('Semantic congruent (+1): %d observations\n', ...
    sum(CongruencySemanticCategories == 1));

fprintf('Semantic incongruent (-1): %d observations\n', ...
    sum(CongruencySemanticCategories == -1));


%% ==========================================================
% Frequency
%
% If original frequency = 0:
% assign log-frequency = 0
%% ==========================================================

FreqOriginal = ...
    double(designTable.Frequency);


if any(FreqOriginal < 0 | isnan(FreqOriginal))
    error('Frequency contains negative or missing values.');
end


FreqRaw = zeros(nObs,1);

idxFreq = FreqOriginal > 0;

FreqRaw(idxFreq) = ...
    log(FreqOriginal(idxFreq));


if std(FreqRaw) == 0
    error('Frequency has zero variance.');
end


Freq = ...
    (FreqRaw - mean(FreqRaw)) ./ std(FreqRaw);


%% ==========================================================
% Stroke
%% ==========================================================

StrokeRaw = ...
    double(designTable.NumbersofStorks);


if any(~isfinite(StrokeRaw))
    error('Stroke contains invalid values.');
end


if std(StrokeRaw) == 0
    error('Stroke has zero variance.');
end


Stroke = ...
    (StrokeRaw - mean(StrokeRaw)) ./ std(StrokeRaw);


%% ==========================================================
% Distractor length
%% ==========================================================

LengthofDistrctorRaw = ...
    double(designTable.LengthofDistrctor);


if any(~isfinite(LengthofDistrctorRaw))
    error('LengthofDistrctor contains invalid values.');
end


if std(LengthofDistrctorRaw) == 0
    error('LengthofDistrctor has zero variance.');
end


LengthofDistrctor = ...
    (LengthofDistrctorRaw - ...
     mean(LengthofDistrctorRaw)) ./ ...
     std(LengthofDistrctorRaw);


%% ==========================================================
% STAGE 1
%
% Remove subject/item random-effect contributions
%
% Keep all fixed effects
%% ==========================================================

fprintf('\nRemoving subject/item random effects...\n');


mEEG = nan(size(EEGdata));


lmeFormula = ...
    ['EEG ~ Stroke + Freq + LengthofDistrctor + ' ...
     'CongruencySemanticCategories + JSD + Classifier + ' ...
     '(1|Subj) + (1|Item)'];


for ch = 1:nChan

    fprintf('LME channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = ...
            squeeze(double(EEGdata(:,ch,t)));


        tbl = table( ...
            Y, ...
            Stroke, ...
            Freq, ...
            LengthofDistrctor, ...
            CongruencySemanticCategories, ...
            JSD, ...
            Classifier, ...
            Subj, ...
            Item, ...
            'VariableNames', { ...
            'EEG', ...
            'Stroke', ...
            'Freq', ...
            'LengthofDistrctor', ...
            'CongruencySemanticCategories', ...
            'JSD', ...
            'Classifier', ...
            'Subj', ...
            'Item'});


        lme = ...
            fitlme(tbl,lmeFormula);


        % ----------------------------------------------
        % Random-effect contribution
        %
        % conditional fitted:
        %   fixed + random
        %
        % marginal fitted:
        %   fixed only
        % ----------------------------------------------

        randomContribution = ...
            fitted(lme,'Conditional',true) - ...
            fitted(lme,'Conditional',false);


        % ----------------------------------------------
        % Marginal EEG
        % ----------------------------------------------

        mEEG(:,ch,t) = ...
            Y - randomContribution;

    end

end


fprintf('Random-effect removal completed.\n');


%% ==========================================================
% STAGE 2
%
% Classifier effect
%
% FULL:
%
% Stroke + Freq + Length +
% SemanticCongruency + JSD + Classifier
%
% NULL:
%
% Stroke + Freq + Length +
% SemanticCongruency + JSD
%% ==========================================================

fprintf('\nPreparing Classifier models...\n');


X_full = [ ...
    Stroke, ...
    Freq, ...
    LengthofDistrctor, ...
    CongruencySemanticCategories, ...
    JSD, ...
    Classifier];


X_null = [ ...
    Stroke, ...
    Freq, ...
    LengthofDistrctor, ...
    CongruencySemanticCategories, ...
    JSD];


%% ==========================================================
% Add intercept manually
%
% This lets us use direct OLS matrix operations instead
% of repeatedly calling fitlm().
%% ==========================================================

X_full_i = ...
    [ones(nObs,1), X_full];

X_null_i = ...
    [ones(nObs,1), X_null];


% ----------------------------------------------
% Coefficient positions
%
% 1 = intercept
% 2 = Stroke
% 3 = Freq
% 4 = Length
% 5 = Semantic congruency
% 6 = JSD
% 7 = Classifier
% ----------------------------------------------

idx_Cla = 7;


%% ==========================================================
% Precompute OLS matrices
%% ==========================================================

% FULL model
XtX_full_inv = ...
    inv(X_full_i' * X_full_i);

B_full = ...
    XtX_full_inv * X_full_i';

df_full = ...
    nObs - size(X_full_i,2);


% NULL model
XtX_null_inv = ...
    inv(X_null_i' * X_null_i);

B_null = ...
    XtX_null_inv * X_null_i';


%% ==========================================================
% Observed Classifier t-map
%% ==========================================================

fprintf('\nComputing observed Classifier t-map...\n');


t_Obs_Cla = nan(nChan,nTime);


for ch = 1:nChan

    for t = 1:nTime

        Y = ...
            squeeze(mEEG(:,ch,t));


        % ------------------------------------------
        % OLS beta
        % ------------------------------------------

        beta = ...
            B_full * Y;


        % ------------------------------------------
        % Residuals
        % ------------------------------------------

        residual = ...
            Y - X_full_i * beta;


        % ------------------------------------------
        % Residual variance
        % ------------------------------------------

        sigma2 = ...
            sum(residual.^2) / df_full;


        % ------------------------------------------
        % Standard error for Classifier
        % ------------------------------------------

        se_Cla = ...
            sqrt( ...
            sigma2 * ...
            XtX_full_inv(idx_Cla,idx_Cla));


        % ------------------------------------------
        % t statistic
        % ------------------------------------------

        t_Obs_Cla(ch,t) = ...
            beta(idx_Cla) / se_Cla;

    end

end


fprintf('Observed t-map completed.\n');


%% ==========================================================
% Reduced/null model
%
% Freedman-Lane:
%
% Yperm = Yhat_null + permuted residuals
%% ==========================================================

fprintf('\nComputing reduced model...\n');


Yhat_null_Cla = ...
    nan(nObs,nChan,nTime);

Residual_null_Cla = ...
    nan(nObs,nChan,nTime);


for ch = 1:nChan

    fprintf('Reduced model channel %d/%d\n', ...
        ch,nChan);

    for t = 1:nTime

        Y = ...
            squeeze(mEEG(:,ch,t));


        beta_null = ...
            B_null * Y;


        Yhat_null_Cla(:,ch,t) = ...
            X_null_i * beta_null;


        Residual_null_Cla(:,ch,t) = ...
            Y - Yhat_null_Cla(:,ch,t);

    end

end


%% ==========================================================
% Reconstruction check
%% ==========================================================

Y_test = squeeze(mEEG(:,1,1));

reconstruction_error = ...
    norm( ...
    Y_test - ...
    (Yhat_null_Cla(:,1,1) + ...
     Residual_null_Cla(:,1,1)));


fprintf('\nReconstruction error = %.12f\n', ...
    reconstruction_error);


%% ==========================================================
% Observed TFCE
%% ==========================================================

fprintf('\nComputing observed TFCE...\n');


ChN = ...
    ept_ChN2(channelinfo);


E_H = ...
    [0.66 2];


TFCE_Obs_Cla = ...
    ept_mex_TFCE2D( ...
    t_Obs_Cla, ...
    ChN, ...
    E_H);


fprintf('Observed TFCE completed.\n');


fprintf('\nMax observed |t| = %.4f\n', ...
    max(abs(t_Obs_Cla(:))));


fprintf('Max observed |TFCE| = %.4f\n', ...
    max(abs(TFCE_Obs_Cla(:))));


%% ==========================================================
% Restricted permutation indices
%% ==========================================================

nPerm = 999;


fprintf('\nGenerating restricted permutations...\n');


rperms = ...
    lmeEEG_permutations2( ...
    nPerm, ...
    Subj, ...
    Item);


%% ==========================================================
% Permutation diagnostics
%% ==========================================================

original_idx = ...
    (1:nObs)';


changed_fraction = ...
    mean(rperms ~= original_idx,1);


fprintf('Mean proportion observations moved = %.4f\n', ...
    mean(changed_fraction));


%% ==========================================================
% Freedman-Lane permutations
%% ==========================================================

fprintf('\nStarting Freedman-Lane permutations...\n');


TFCE_permMax_Cla = ...
    zeros(nPerm,1);


permMaxT_Cla = ...
    zeros(nPerm,1);


%% ==========================================================
% Parallel pool
%% ==========================================================

delete(gcp('nocreate'));


mexFile = ...
    which('ept_mex_TFCE2D');


if isempty(mexFile)
    error('ept_mex_TFCE2D is not on the MATLAB path.');
end


pool = ...
    parpool( ...
    'Processes', ...
    4, ...
    'AttachedFiles',{mexFile});


%% ==========================================================
% Permutation loop
%% ==========================================================

parfor p = 1:nPerm


    perm_idx = ...
        rperms(:,p);


    perm_t_Cla = ...
        zeros(nChan,nTime);


    for ch = 1:nChan

        for t = 1:nTime


            % ------------------------------------------
            % Freedman-Lane response
            % ------------------------------------------

            Y_perm = ...
                Yhat_null_Cla(:,ch,t) + ...
                Residual_null_Cla( ...
                perm_idx,ch,t);


            % ------------------------------------------
            % FAST OLS
            %
            % No fitlm() here.
            % ------------------------------------------

            beta_perm = ...
                B_full * Y_perm;


            residual_perm = ...
                Y_perm - ...
                X_full_i * beta_perm;


            sigma2_perm = ...
                sum(residual_perm.^2) / ...
                df_full;


            se_Cla_perm = ...
                sqrt( ...
                sigma2_perm * ...
                XtX_full_inv( ...
                idx_Cla,idx_Cla));


            perm_t_Cla(ch,t) = ...
                beta_perm(idx_Cla) / ...
                se_Cla_perm;

        end

    end


    % ----------------------------------------------
    % Pre-TFCE diagnostic
    % ----------------------------------------------

    permMaxT_Cla(p) = ...
        max(abs(perm_t_Cla(:)));


    % ----------------------------------------------
    % TFCE
    % ----------------------------------------------

    TFCE_perm = ...
        ept_mex_TFCE2D( ...
        perm_t_Cla, ...
        ChN, ...
        E_H);


    TFCE_permMax_Cla(p) = ...
        max(abs(TFCE_perm(:)));

end


fprintf('Permutation completed.\n');


%% ==========================================================
% Corrected inference
%% ==========================================================

fprintf('\nComputing TFCE-corrected significance...\n');


alpha = 0.05;


% ----------------------------------------------------------
% Use permutation null ONLY
% ----------------------------------------------------------

maxTFCE_Cla = sort([TFCE_permMax_Cla;max(abs(TFCE_Obs_Cla(:)))]);

maxTFCEcrit_Cla = maxTFCE_Cla(round(nPerm*(1-alpha)))

fprintf('Critical TFCE value = %.4f\n', ...
    maxTFCEcrit_Cla);


Mask_Cla = ...
    abs(TFCE_Obs_Cla) >= ...
    maxTFCEcrit_Cla;


%% ==========================================================
% Corrected p-values
%% ==========================================================

P_Values_Cla = ...
    nan(nChan,nTime);


for ch = 1:nChan

    for tp = 1:nTime

        P_Values_Cla(ch,tp) = ...
            ( ...
            sum( ...
            TFCE_permMax_Cla >= ...
            abs(TFCE_Obs_Cla(ch,tp))) ...
            + 1 ...
            ) ...
            / (nPerm + 1);

    end

end


fprintf('TFCE correction completed.\n');


%% ==========================================================
% Diagnostics
%% ==========================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('CLASSIFIER DIAGNOSTIC\n');
fprintf('============================================\n');


fprintf('Max |t observed|          = %.4f\n', ...
    max(abs(t_Obs_Cla(:))));


fprintf('Permutation max-|t| median = %.4f\n', ...
    median(permMaxT_Cla));


fprintf('Permutation max-|t| 95%%   = %.4f\n', ...
    prctile(permMaxT_Cla,95));


fprintf('\n');


fprintf('Max |TFCE observed|       = %.4f\n', ...
    max(abs(TFCE_Obs_Cla(:))));


fprintf('Null TFCE median          = %.4f\n', ...
    median(TFCE_permMax_Cla));


fprintf('Null TFCE 95%%             = %.4f\n', ...
    maxTFCEcrit_Cla);


fprintf('Null TFCE maximum         = %.4f\n', ...
    max(TFCE_permMax_Cla));


fprintf('Minimum corrected p       = %.4f\n', ...
    min(P_Values_Cla(:)));


fprintf('Significant points        = %d\n', ...
    sum(Mask_Cla(:)));


%% ==========================================================
% Store results
%% ==========================================================

Results = struct();


Results.t_Obs_Cla = ...
    t_Obs_Cla;


Results.TFCE_Obs_Cla = ...
    TFCE_Obs_Cla;


Results.TFCE_Null_Cla = ...
    TFCE_permMax_Cla;


Results.maxT_Null_Cla = ...
    permMaxT_Cla;


Results.maxTFCEcrit_Cla = ...
    maxTFCEcrit_Cla;


Results.Mask_Cla = ...
    Mask_Cla;


Results.P_Values_Cla = ...
    P_Values_Cla;


Results.alpha = ...
    alpha;


Results.nPerm = ...
    nPerm;


Results.model = ...
    ['EEG ~ Stroke + Freq + LengthofDistrctor + ' ...
     'CongruencySemanticCategories + JSD + Classifier + ' ...
     '(1|Subj) + (1|Item)'];


Results.test = ...
    'Classifier main effect';


%% ==========================================================
% Save
%% ==========================================================

if ~exist('../Results','dir')
    mkdir('../Results');
end


save( ...
    '../Results/09_realDOE_classifier_diagnostic.mat', ...
    'Results', ...
    'nChan', ...
    'time', ...
    'channelinfo');


fprintf('\nResults saved.\n');


%% ==========================================================
% Plot 1
%
% Observed Classifier t-map
%% ==========================================================

figure;

imagesc( ...
    time, ...
    1:nChan, ...
    t_Obs_Cla);

axis xy;

xlim([-200 700]);

set(gca, ...
    'YTick',1:nChan, ...
    'YTickLabel',{channelinfo.labels}, ...
    'XTick',-200:150:700, ...
    'TickLength',[0 0], ...
    'FontSize',15, ...
    'FontName','Arial');

xlabel('Time (ms)');
ylabel('Channel');

title('Observed Classifier t-map');

cb = colorbar;

ylabel(cb,'t-value');


%% ==========================================================
% Plot 2
%
% Observed Classifier TFCE map
%% ==========================================================

figure;

imagesc( ...
    time, ...
    1:nChan, ...
    TFCE_Obs_Cla);

axis xy;

xlim([-200 700]);

set(gca, ...
    'YTick',1:nChan, ...
    'YTickLabel',{channelinfo.labels}, ...
    'XTick',-200:150:700, ...
    'TickLength',[0 0], ...
    'FontSize',15, ...
    'FontName','Arial');

xlabel('Time (ms)');
ylabel('Channel');

title('Observed Classifier TFCE map');

cb = colorbar;

ylabel(cb,'TFCE value');


%% ==========================================================
% Plot 3
%
% TFCE-corrected Classifier t-map
%% ==========================================================

sigT = ...
    t_Obs_Cla;


sigT(~Mask_Cla) = ...
    0;


figure;

imagesc( ...
    time, ...
    1:nChan, ...
    sigT);

axis xy;

xlim([-200 700]);

set(gca, ...
    'YTick',1:nChan, ...
    'YTickLabel',{channelinfo.labels}, ...
    'XTick',-200:150:700, ...
    'TickLength',[0 0], ...
    'FontSize',15, ...
    'FontName','Arial');

xlabel('Time (ms)');
ylabel('Channel');

title('TFCE-corrected Classifier Effect');

cb = colorbar;

ylabel(cb,'t-value');


%% ==========================================================
% Plot 4
%
% Max-|t| permutation distribution
%% ==========================================================

figure;

histogram( ...
    permMaxT_Cla, ...
    30);

hold on;

xline( ...
    max(abs(t_Obs_Cla(:))), ...
    'LineWidth',2);

xline( ...
    prctile(permMaxT_Cla,95), ...
    '--', ...
    'LineWidth',2);

xlabel('Maximum |t|');
ylabel('Permutation count');

title('Classifier max-|t| permutation distribution');

legend( ...
    'Permutation null', ...
    'Observed max |t|', ...
    '95% null threshold');


%% ==========================================================
% Plot 5
%
% Max-TFCE permutation distribution
%% ==========================================================

figure;

histogram( ...
    TFCE_permMax_Cla, ...
    30);

hold on;

xline( ...
    max(abs(TFCE_Obs_Cla(:))), ...
    'LineWidth',2);

xline( ...
    maxTFCEcrit_Cla, ...
    '--', ...
    'LineWidth',2);

xlabel('Maximum |TFCE|');
ylabel('Permutation count');

title('Classifier max-TFCE permutation distribution');

legend( ...
    'Permutation null', ...
    'Observed max TFCE', ...
    '95% null threshold');