%% ==========================================================
% TFCE analysis using Ordinary Freedman-Lane procedure
%
% Stage 1:
%  subject/item + other nuisance variables for a GLM
%
% Stage 2:
%   reduced GLM
%
% Stage 3:
%   Freedman-lane residualization
%
% Test:
%   Classifier effect
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
% Binary predictors
%% ==========================================================
%----------------------------
% JSD
%----------------------------

JSD = zeros(nObs,1);

tmpJSD = string(designTable.JSD);

JSD(tmpJSD == 'L') = -1;   % Low JSD
JSD(tmpJSD == 'H') =  1;   % High JSD

% Check coding
fprintf('JSD Low  (L, -1): %d observations\n', sum(JSD == -1));
fprintf('JSD High (H, +1): %d observations\n', sum(JSD ==  1));

% Check for unexpected/unmatched values
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

% Check coding
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

% %% ----------------------------
% % Length of distractor
% %% ----------------------------
% 
LengthofDistrctorRaw = designTable.LengthofDistrctor;
LengthofDistrctor = (LengthofDistrctorRaw - mean(LengthofDistrctorRaw)) ./ std(LengthofDistrctorRaw);

%% ==========================================================
% Construct nuisance and full design matrices
%
% Subject + Item are treated as nuisance fixed effects.
%
% Test:
% H0: t_Classifier = 0
%% ==========================================================
% Reduced model: nuisance only
X_null = [ ...
    ones(nObs,1), ...
    SubjDummy, ...
    ItemDummy, ...
    Stroke, ...
    Freq, ...
    LengthofDistrctor, ...
    CongruencySemanticCategories, ...
    JSD];

% Full model: nuisance + effect of interest
X_full = [ ...
    X_null, ...
    Classifier];

idx_Classifier = size(X_full,2);

%% ==========================================================
% Observed t-map
%% ==========================================================

t_Obs = nan(nChan,nTime);

for ch = 1:nChan
    for t = 1:nTime

        Y = squeeze(double(EEGdata(:,ch,t)));

        lm_full = fitlm(X_full(:,2:end),Y);

        % last coefficient = Classifier
        t_Obs(ch,t) = ...
            lm_full.Coefficients.tStat(end);

    end
end

%% ==========================================================
% Observed TFCE
%% ==========================================================

ChN = ept_ChN2(channelinfo);
E_H = [0.66 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs,ChN,E_H);

%% ==========================================================
% Reduced model
%% ==========================================================

Yhat_null = nan(nObs,nChan,nTime);
Residual_null = nan(nObs,nChan,nTime);

for ch = 1:nChan
    for t = 1:nTime

        Y = squeeze(double(EEGdata(:,ch,t)));

        lm_null = fitlm(X_null(:,2:end),Y);

        Yhat_null(:,ch,t) = lm_null.Fitted;

        Residual_null(:,ch,t) = lm_null.Residuals.Raw;

    end
end

%% ==========================================================
% Freedman-Lane permutation
%% ==========================================================

nPerm = 999;

TFCE_permMax = zeros(nPerm,1);

delete(gcp('nocreate'));

mexFile = which('ept_mex_TFCE2D');

if isempty(mexFile)
    error('ept_mex_TFCE2D is not on the MATLAB path.');
end

pool = parpool('Processes',6, 'AttachedFiles',{mexFile});

parfor p = 1:nPerm

    % unrestricted residual permutation
    perm_idx = randperm(nObs);

    perm_t = nan(nChan,nTime);

    for ch = 1:nChan
        for t = 1:nTime

            Y_perm = Yhat_null(:,ch,t) + ...
                Residual_null(perm_idx,ch,t);

            lm_perm = fitlm(X_full(:,2:end),Y_perm);

            perm_t(ch,t) = lm_perm.Coefficients.tStat(end);

        end
    end

    TFCE_perm = ept_mex_TFCE2D(perm_t,ChN,E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end

fprintf('Permutation completed.\n');

%% ==========================================================
% Part 3
%
% TFCE correction
%
%% ==========================================================

fprintf('\nComputing TFCE corrected significance...\n');

alpha = 0.05;


maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);

maxTFCEcrit = maxTFCE(round(nPerm*(1-alpha)));

fprintf('Critical TFCE value = %.4f\n',maxTFCEcrit);

Mask = abs(TFCE_Obs) >= TFCEcrit;

P_Values = nan(nChan,nTime); 

for ch=1:nChan

    for tp=1:nTime

        P_Values(ch,tp)= ...
            (sum(TFCE_permMax >= abs(TFCE_Obs(ch,tp)))+1) /(nPerm+1);

    end
    
end

fprintf('TFCE correction completed.\n');

%% ==========================================================
% Store results
%% ==========================================================

Results=struct();

Results.t_Obs = t_Obs;

Results.TFCE_Obs = TFCE_Obs;

Results.TFCE_Null = TFCE_permMax;

Results.maxTFCEcrit = maxTFCEcrit;

Results.Mask = Mask;

Results.P_Values = P_Values;


fprintf('Results stored.\n');

%% ==========================================================
% Save
%% ==========================================================

if ~exist('../Results','dir')

    mkdir('../Results');

end

save('../Results/09_realDOE_results_0.mat',...
    'Results',...
    'nChan',...
    'time',...
    'channelinfo');

fprintf('Saved results.\n')

%% ==========================================================
% Plot corrected t-map
%% ==========================================================

% clear all; clc; close all
% 
% load('../Results/09_realDOE_results_4.mat')

figure;

sigT = Results.t_Obs;

sigT(~Results.Mask)=0;

imagesc(time,1:nChan,Results.P_Values < 0.2);

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
% Plot observed TFCE map
%% ==========================================================

figure;

imagesc(time,1:nChan,Results.TFCE_Obs_Int);

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
