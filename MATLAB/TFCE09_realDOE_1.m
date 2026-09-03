%% ==========================================================
% TFCE analysis using the Freedman-Lane permutation procedure
%
% Stage 1:
%   Fit a linear mixed-effects model and remove the estimated
%   subject and item random-intercept contributions from the EEG data.
%
% Stage 2:
%   Fit fixed-effects full and reduced GLMs to the marginalized EEG data.
%
% Stage 3:
%   Apply restricted Freedman-Lane residual permutation and TFCE correction.
%
% Effect tested:
%   Main effect of Classifier, while retaining the JSD-by-Classifier
%   interaction in both the full and reduced models.
%
%% ==========================================================

clear; clc; close all;

rng(123);

fprintf('\nStarting Freedman-Lane analysis...\n');

%% ==========================================================
% Load data
%% ==========================================================

load('../Data/realDOE_3.mat');

%% ==========================================================
% Dimensions
%% ==========================================================

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

%% ----------------------------
% Subject
%% ----------------------------
if ismember('SubjID',varNames)

    Subj = categorical(designTable.SubjID);

elseif ismember('SubjSubj',varNames)

    Subj = categorical(designTable.SubjSubj);

else

    error('No subject identifier found.');

end

%% ----------------------------
% Item
%% ----------------------------

if ~ismember('Target',varNames)

    error('Target variable missing.');

end

Item = categorical(designTable.Target);

%% ==========================================================
% Prepare effect-coded predictors
%% ==========================================================
%----------------------------
% JSD
%----------------------------

JSD = zeros(nObs,1);

tmpJSD = string(designTable.JSD);

JSD(tmpJSD == 'L') = -1;   % Low JSD
JSD(tmpJSD == 'H') =  1;   % High JSD

% Report the number of observations at each JSD level
fprintf('JSD Low  (L, -1): %d observations\n', sum(JSD == -1));
fprintf('JSD High (H, +1): %d observations\n', sum(JSD ==  1));

% Ensure that every JSD observation received a valid code
if any(JSD == 0)
    error('Some JSD observations were not coded. Check the JSD labels.');
end

%% ----------------------------
% Classifier (effect of interest)
%% ----------------------------

Classifier = zeros(nObs,1);

tmp = string(designTable.ClassifierCongruency);

Classifier(tmp == 'Congruent')   =  1;
Classifier(tmp == 'Incongruent') = -1;

% Report the number of observations in each Classifier condition
fprintf('Congruent (+1):   %d observations\n', sum(Classifier == 1));
fprintf('Incongruent (-1): %d observations\n', sum(Classifier == -1));

%% ----------------------------
% Frequency
%% ----------------------------
FreqRaw = log(double(designTable.Frequency));

if any(~isfinite(FreqRaw))

    error('Frequency contains invalid values.');

end

Freq = (FreqRaw-mean(FreqRaw)) ./ std(FreqRaw);

%% ----------------------------
% Stroke
%% ----------------------------
StrokeRaw = double(designTable.NumbersofStorks);

Stroke = (StrokeRaw-mean(StrokeRaw)) ./ std(StrokeRaw);

%% ----------------------------
% Semantic category congruency
%% ----------------------------

CongruencySemanticCategories = designTable.CongruencySemanticCategories;

%% ----------------------------
% Length of distractor
%% ----------------------------
% 
LengthofDistrctorRaw = designTable.LengthofDistrctor;
LengthofDistrctor = (LengthofDistrctorRaw - mean(LengthofDistrctorRaw)) ./ std(LengthofDistrctorRaw);

%%
%% ==========================================================
% Stage 1: LME-based removal of random-intercept contributions
%
% Model:
%
% EEG ~ Stroke + Freq + LengthofDistrctor +
%       CongruencySemanticCategories +
%       JSD * Classifier +
%       (1 + Classifier | Subj) +
%       (1 | Item)
%
% NOTE:
% Subject and item are modeled as crossed random intercepts.
%% ==========================================================

fprintf('\nStarting Stage 1: LME marginalization...\n');

mEEG = nan(nObs,nChan,nTime);

%% ==========================================================
% Construct LME table
%% ==========================================================

LMEtable = table( ...
    Subj, ...
    Item, ...
    Stroke, ...
    Freq, ...
    LengthofDistrctor, ...
    CongruencySemanticCategories, ...
    JSD, ...
    Classifier, ...
    'VariableNames', { ...
    'Subj', ...
    'Item', ...
    'Stroke', ...
    'Freq', ...
    'LengthofDistrctor', ...
    'CongruencySemanticCategories', ...
    'JSD', ...
    'Classifier'});


%% ==========================================================
% Fit LME at every channel x time point
%% ==========================================================

for ch = 1:nChan

    fprintf('LME channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = squeeze(double(EEGdata(:,ch,t)));

        LMEtable.EEG = Y;

        %% --------------------------------------------------
        % Mixed-effects model
        %% --------------------------------------------------
        lme = fitlme( ...
            LMEtable, ...
            ['EEG ~ Stroke + Freq + LengthofDistrctor + ' ...
             'CongruencySemanticCategories + ' ...
             'JSD * Classifier + ' ...
             '(1 | Subj) + (1 | Item)']);

        %% --------------------------------------------------
        % Conditional fitted values
        %
        % Fixed effects
        % +
        % Subject random intercept
        % +
        % Item random intercept
        %% --------------------------------------------------
        fittedConditional = fitted(lme,'Conditional',true);

        %% --------------------------------------------------
        % Marginal fitted values
        %
        % Fixed effects only
        %% --------------------------------------------------
        fittedMarginal = fitted(lme,'Conditional',false);

        %% --------------------------------------------------
        % Total random-effect contribution
        %% --------------------------------------------------
        randomContribution = fittedConditional - fittedMarginal;

        %% --------------------------------------------------
        % Marginalized EEG
        %
        % Removes:
        %
        %   subject random intercept
%   item random intercept
        %
        % Retains:
        %
        %   fixed Classifier effect
        %   fixed JSD effect
        %   nuisance fixed effects
        %   residual error
        %% --------------------------------------------------
        mEEG(:,ch,t) = Y - randomContribution;

    end

end

fprintf('\nStage 1 marginalization completed.\n');

%% ==========================================================
% Stage 2: Fixed-effects GLM
%
% Subject and Item are NOT included here because their
% modeled random contributions were handled in Stage 1.
%% ==========================================================
X_full = [ ...
    Stroke, ...
    Freq, ...
    LengthofDistrctor, ...
    CongruencySemanticCategories, ...
    JSD, ...
    Classifier, JSD .* Classifier];

X_null_Cla = [ ...
    Stroke, ...
    Freq, ...
    LengthofDistrctor, ...
    CongruencySemanticCategories, ...
    JSD, JSD .* Classifier];
%% ==========================================================
% Observed Classifier t-map
%% ==========================================================
fprintf('\nComputing observed Classifier t-map...\n');

t_Obs_Cla = nan(nChan,nTime);

for ch = 1:nChan

    fprintf('Observed GLM channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = squeeze(mEEG(:,ch,t));

        lm_full = fitlm(X_full,Y);

        % Classifier is the second-to-last predictor; the final predictor
        % is the JSD-by-Classifier interaction.
        t_Obs_Cla(ch,t) = lm_full.Coefficients.tStat(end-1);
        
    end

end

%% ==========================================================
% Reduced model for Classifier Freedman-Lane test
%% ==========================================================
fprintf('\nPrecomputing reduced Classifier models...\n');

Yhat_null_Cla = nan(nObs,nChan,nTime);
Residual_null_Cla = nan(nObs,nChan,nTime);

for ch = 1:nChan

    fprintf('Reduced GLM channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = squeeze(mEEG(:,ch,t));

        lm_null = fitlm(X_null_Cla,Y);

        Yhat_null_Cla(:,ch,t) = lm_null.Fitted;

        Residual_null_Cla(:,ch,t) = lm_null.Residuals.Raw;

    end

end

%% ==========================================================
% Observed TFCE
%% ==========================================================
ChN = ept_ChN2(channelinfo);

E_H = [0.66 2];

TFCE_Obs_Cla = ept_mex_TFCE2D(t_Obs_Cla, ChN, E_H);

%% ==========================================================
% Generate restricted permutations that respect the subject-item structure
%% ==========================================================
nPerm = 999;

rperms = lmeEEG_permutations2(nPerm, Subj, Item);

TFCE_permMax_Cla = zeros(nPerm,1);

%% ==========================================================
% Freedman-Lane permutation loop for the Classifier main effect
%% ==========================================================
fprintf('\nStarting %d permutations...\n',nPerm);

for p = 1:nPerm

    perm_idx = rperms(:,p);

    t_perm_Cla = nan(nChan,nTime);

    for ch = 1:nChan

        for t = 1:nTime
            %% ------------------------------------------
            % Freedman-Lane pseudo-response
            %% ------------------------------------------
            Y_perm = Yhat_null_Cla(:,ch,t) + Residual_null_Cla(perm_idx,ch,t);

            %% ------------------------------------------
            % Full GLM
            %% ------------------------------------------
            lm_perm = fitlm(X_full,Y_perm);

            %% ------------------------------------------
            % Classifier t-statistic
            %% ------------------------------------------
            t_perm_Cla(ch,t) = lm_perm.Coefficients.tStat(end-1);

        end

    end
    %% ----------------------------------------------
    % TFCE
    %% ----------------------------------------------
    TFCE_perm = ept_mex_TFCE2D(t_perm_Cla, ChN, E_H);
    %% ----------------------------------------------
    % Maximum absolute TFCE
    %% ----------------------------------------------
    TFCE_permMax_Cla(p) = max(abs(TFCE_perm(:)));
    
    fprintf('Permutation %d/%d\n',p,nPerm);

end

%% ==========================================================
fprintf('\nComputing TFCE-corrected significance...\n');

alpha = 0.05;

% Critical value from the permutation distribution
maxTFCEcrit = quantile(TFCE_permMax_Cla, 1-alpha);

fprintf('Critical TFCE value = %.4f\n', maxTFCEcrit);

% Two-sided TFCE significance mask
Mask_Cla = abs(TFCE_Obs_Cla) >= maxTFCEcrit;

% Permutation-corrected p-values
P_Values_Cla = nan(nChan,nTime);

for ch = 1:nChan
    for tp = 1:nTime
        P_Values_Cla(ch,tp) = ...
            (sum(TFCE_permMax_Cla >= abs(TFCE_Obs_Cla(ch,tp))) + 1) ...
            / (nPerm + 1);
    end
end

fprintf('TFCE correction completed.\n');

%% ==========================================================
% Store results
%% ==========================================================
Results = struct();

Results.t_Obs_Cla = t_Obs_Cla;

Results.TFCE_Obs_Cla = TFCE_Obs_Cla;

Results.TFCE_permMax_Cla = TFCE_permMax_Cla;

Results.maxTFCEcrit = maxTFCEcrit;

Results.Mask_Cla = Mask_Cla;

Results.P_Values_Cla = P_Values_Cla;

%% ==========================================================
% Save
%% ==========================================================
if ~exist('../Results','dir')

    mkdir('../Results');

end

save('../Results/09_realDOE_results_1.mat',...
    'Results',...
    'nChan',...
    'time',...
    'channelinfo');

fprintf('Saved results.\n')

%% ==========================================================
% Plot the pointwise significance mask using the selected display threshold
%% ==========================================================

% clear all; clc; close all
% 
% load('../Results/09_realDOE_results_1.mat')

figure;

sigT = Results.t_Obs_Cla;

sigT(~Results.Mask_Cla) = 0;

imagesc(time, 1:nChan, sigT);

axis xy;

xlim([-200 700]);

set(gca,...
    'YTick',1:nChan,...
    'YTickLabel',{channelinfo.labels},...
    'XTick',-200:150:700,...
    'TickLength',[0 0],...
    'FontSize',15,...
    'FontName','Arial');

xlabel('Time (ms)');

ylabel('Channel');

title('TFCE-corrected Classifier Effect');

cb=colorbar;

ylabel(cb,'t-value',...
    'FontSize',15,...
    'FontName','Arial');

%% ==========================================================
% Plot the observed TFCE map
%% ==========================================================
figure;

imagesc(time,1:nChan,Results.TFCE_Obs_Cla);

axis xy;

xlim([-200 700]);

set(gca,...
    'YTick',1:nChan,...
    'YTickLabel',{channelinfo.labels},...
    'XTick',-200:150:700,...
    'TickLength',[0 0],...
    'FontSize',15,...
    'FontName','Arial');

xlabel('Time (ms)');

ylabel('Channel');

title('Observed TFCE Map');

cb=colorbar;

ylabel(cb,'TFCE value',...
    'FontSize',15,...
    'FontName','Arial');
