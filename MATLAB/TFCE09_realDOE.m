%% ==========================================================
% lmeEEG TFCE analysis using Freedman-Lane procedure
%
% Stage 1:
%   Remove subject/item random intercepts using LME
%
% Stage 2:
%   Freedman-Lane permutation on adjusted EEG
%
% Model:
%
% EEG ~ CongruencySemanticCategories + Stroke + Freq +
%       JSD + Classifier + (1|Subj) + (1|Item)
%
% Test:
%   Classifier effect
%
%% ==========================================================

clear; clc; close all;

rng(123);

fprintf('\nStarting lmeEEG Freedman-Lane analysis...\n');

%% ==========================================================
% Load data
%% ==========================================================

load('../Data/realDOE.mat');

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
Freq = log(double(designTable.Frequency));

if any(~isfinite(Freq))

    error('Frequency contains invalid values.');

end

%% ----------------------------
% Stroke
%% ----------------------------
StrokeRaw = double(designTable.NumbersofStorks);

Stroke = (StrokeRaw-mean(StrokeRaw)) ./ std(StrokeRaw);

%% ----------------------------
% Semantic category congruency
%% ----------------------------

tmp = categorical(designTable.CongruencySemanticCategories);

lev = categories(tmp);

if length(lev)~=2

    error('CongruencySemanticCategories must have two levels.');

end

CongruencySemanticCategories=zeros(nObs,1);

CongruencySemanticCategories(tmp==lev{1})=-1;
CongruencySemanticCategories(tmp==lev{2})=1;

%% ==========================================================
% Create adjusted EEG
%
% Remove:
%   Subject random intercept
%   Item random intercept
%
% Keep:
%   Fixed effects
%% ==========================================================
fprintf('\nRemoving subject/item random effects...\n');

mEEG = nan(size(EEGdata));

lmeFormula = ...
    'EEG ~ CongruencySemanticCategories + Stroke + Freq + JSD * Classifier + (1|Subj)+(1|Item)';

for ch = 1:nChan

    for t = 1:nTime

        Y = squeeze(double(EEGdata(:,ch,t)));

        tbl = table( ...
            Y,...
            Stroke,...
            Freq,...
            CongruencySemanticCategories,...
            JSD,...
            Classifier,...
            Subj,...
            Item,...
            'VariableNames',...
            {'EEG',...
             'Stroke',...
             'Freq',...
             'CongruencySemanticCategories',...
             'JSD',...
             'Classifier',...
             'Subj',...
             'Item'});

        lme = fitlme(tbl,lmeFormula);

        % random-effect contribution:
        %
        % (fixed + random) - fixed
        %
        randomContribution = fitted(lme,'Conditional',true) - ...
            fitted(lme,'Conditional',false);

        % remove only random effects

        mEEG(:,ch,t)=Y-randomContribution;
        
    end
end

fprintf('Random effects removed.\n');

%% ==========================================================
% Part 2
%
% Construct target models
%
% Full model:
%
% EEG ~ Stroke + Freq + CongruencySemanticCategories + JSD * Classifier
%
% Null model:
%
% EEG ~ Stroke + Freq + CongruencySemanticCategories + JSD + JSD .* Classifier
%
%% ==========================================================

fprintf('\nPreparing target design matrices...\n');

X_full = [Stroke, Freq, CongruencySemanticCategories,...
    JSD, JSD.*Classifier, Classifier];

X_null = [Stroke, Freq, CongruencySemanticCategories,...
    JSD, JSD.*Classifier];

%% ==========================================================
% Observed statistics from the full model
%% ==========================================================
fprintf('\nComputing observed t-map...\n');

t_Obs = nan(nChan,nTime);

for ch = 1:nChan

    for t = 1:nTime

        Y=squeeze(mEEG(:,ch,t));

        lm_full = fitlm(X_full,Y);

        coefNames = lm_full.Coefficients.Properties.RowNames;

        idx = contains(coefNames,'x6');

        if sum(idx)~=1
            
            error('Classifier coefficient not found.');

        end

        t_Obs(ch,t)= lm_full.Coefficients.tStat(idx);

    end

end

fprintf('Observed t-map completed.\n');

%% ==========================================================
% Precompute reduced model
%
% Freedman-Lane:
%
% Y_perm = Y_hat_null + permuted residuals
%
%% ==========================================================
fprintf('\nPrecomputing reduced models...\n');

Yhat_null = nan(nObs,nChan,nTime);

Residual_null = nan(nObs,nChan,nTime);

for ch = 1:nChan

    fprintf('Reduced model channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = squeeze(mEEG(:,ch,t));

        lm_null = fitlm(X_null,Y);

        Yhat_null(:,ch,t)=lm_null.Fitted;

        Residual_null(:,ch,t)=lm_null.Residuals.Raw;

    end
end

fprintf('Reduced models completed.\n');

%% ==========================================================
% TFCE observed map
%% ==========================================================
fprintf('\nComputing observed TFCE...\n');

ChN = ept_ChN2(channelinfo);

E_H = [0.66 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE completed.\n');

%% ==========================================================
% Freedman-Lane permutation
%% ==========================================================
nPerm=999;

TFCE_permMax=zeros(nPerm,1);

fprintf('\nStarting Freedman-Lane permutations...\n');

parfor p=1:nPerm
    
    fprintf('At %d/%d permutation\n', p, nPerm);

    % ------------------------------------
    % One permutation for all EEG points
    % ------------------------------------

    perm_idx=randperm(nObs);

    perm_t=zeros(nChan,nTime);

    for ch=1:nChan

        for t=1:nTime

            % original reduced-model fitted values

            Y0 = Yhat_null(:,ch,t);

            % permuted residuals

            e_perm = Residual_null(perm_idx,ch,t);

            % Freedman-Lane response

            Y_perm = Y0 + e_perm;

            % full model

            lm_perm = fitlm(X_full,Y_perm);
            
            coefNames = lm_perm.Coefficients.Properties.RowNames;

            idx=contains(coefNames,'x6');

            perm_t(ch,t)= lm_perm.Coefficients.tStat(idx);
            
        end

    end

    % TFCE on this permutation

    TFCE_perm = ept_mex_TFCE2D(perm_t, ChN, E_H);

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

Mask = abs(TFCE_Obs) >= maxTFCEcrit;

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

Results.tObs = t_Obs;

Results.TFCE_Obs = TFCE_Obs;

Results.TFCE_Null = TFCE_permMax;

Results.maxTFCEcrit = maxTFCEcrit;

Results.Mask = Mask;

Results.P_Values = P_Values;

Results.alpha = alpha;

Results.nPerm = nPerm;

Results.model = ...
    ['EEG ~ CongruencySemanticCategories + Stroke + ' ...
     'Freq + JSD + Classifier'];


Results.test = ...
    'Classifier effect after removing subject/item random intercepts';

fprintf('Results stored.\n');

%% ==========================================================
% Save
%% ==========================================================

if ~exist('../Results','dir')

    mkdir('../Results');

end

save('../Results/09_realDOE_results.mat',...
    'Results',...
    'nChan',...
    'time',...
    'channelinfo');

fprintf('Saved results.\n')

%% ==========================================================
% Plot corrected t-map
%% ==========================================================

clear all; clc; close all

load('../Results/09_realDOE_results.mat')

figure;

sigT = Results.tObs;

sigT(~Results.Mask)=0;

imagesc(time,1:nChan,Results.tObs);

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

imagesc(time,1:nChan,Results.TFCE_Obs);

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
