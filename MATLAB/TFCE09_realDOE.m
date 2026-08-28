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
%       SC + SH + (1|Subj) + (1|Item)
%
% Test:
%   SH effect
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
% SemanticCategory
%----------------------------

SemanticCategory = zeros(nObs,1);

tmpSemanticCategory = string(designTable.SemanticCategory);

SemanticCategory(tmpSemanticCategory == 0) = -1;   % SC-
SemanticCategory(tmpSemanticCategory == 1) =  1;   % SC+

% Check coding
fprintf('SC-, -1): %d observations\n', sum(SemanticCategory == -1));
fprintf('SC+, +1): %d observations\n', sum(SemanticCategory ==  1));

% Check for unexpected/unmatched values
if any(SemanticCategory == 0)
    error('Some SC observations were not coded. Check the SC labels.');
end

%% ----------------------------
% Shape
%% ----------------------------

Shape = zeros(nObs,1);

tmpShape = string(designTable.SemanticCategory);

Shape(tmpShape == 0) = -1;   % SC-
Shape(tmpShape == 1) =  1;   % SC+

% Check coding
fprintf('SH-, -1): %d observations\n', sum(Shape == -1));
fprintf('SH+, +1): %d observations\n', sum(Shape ==  1));

% Check for unexpected/unmatched values
if any(Shape == 0)
    error('Some SH observations were not coded. Check the SC labels.');
end

%% ----------------------------
% Frequency
%% ----------------------------
FreqOriginal = double(designTable.Frequency);

FreqRaw = zeros(nObs,1);

idxFreq = FreqOriginal > 0;

FreqRaw(idxFreq) = log(FreqOriginal(idxFreq));

if any(FreqOriginal < 0 | isnan(FreqOriginal))
    error('Frequency contains invalid values.');
end

Freq = (FreqRaw - mean(FreqRaw)) ./ std(FreqRaw);
%% ----------------------------
% Stroke
%% ----------------------------
StrokeRaw = double(designTable.NumbersofStrokes);
Stroke = (StrokeRaw-mean(StrokeRaw)) ./ std(StrokeRaw);
% 
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
    'EEG ~ Freq + Stroke + SemanticCategory * Shape + (1|Subj)+(1|Item)';

for ch = 1:nChan

    for t = 1:nTime

        Y = squeeze(double(EEGdata(:,ch,t)));

        tbl = table( ...
            Y,...
            Freq,...
            Stroke,...
            SemanticCategory,...
            Shape, ...
            Subj,...
            Item,...
            'VariableNames',...
            {'EEG',...
             'Freq',...
             'Stroke',...
             'SemanticCategory',...
             'Shape',...
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
% EEG ~ Freq + Stroke + SemanticCategory * Shape
%
% Null model:
%
% EEG ~ Freq + Stroke + Shape + SemanticCategory .* Shape
%
%% ==========================================================

fprintf('\nPreparing target design matrices...\n');

X_full = [Freq, Stroke, SemanticCategory, Shape, SemanticCategory .* Shape];

X_null_SC = [Freq, Stroke, Shape, SemanticCategory .* Shape];

X_null_SH = [Freq, Stroke, SemanticCategory, SemanticCategory .* Shape];

X_null_Int = [Freq, Stroke, SemanticCategory, Shape];


%% ==========================================================
% Observed statistics from the full model
%% ==========================================================
fprintf('\nComputing observed t-map...\n');

t_Obs_SC = nan(nChan,nTime);

t_Obs_SH = nan(nChan,nTime);

t_Obs_Int = nan(nChan,nTime);

for ch = 1:nChan

    for t = 1:nTime

        Y=squeeze(mEEG(:,ch,t));

        lm_full = fitlm(X_full,Y);

        coefNames = lm_full.Coefficients.Properties.RowNames;

        idx_SC = contains(coefNames,'x3');

        idx_SH = contains(coefNames,'x4');

        idx_Int = contains(coefNames,'x5');



        t_Obs_SC(ch,t)= lm_full.Coefficients.tStat(idx_SC);

        t_Obs_SH(ch,t)= lm_full.Coefficients.tStat(t_Obs_SH);

        t_Obs_Int(ch,t)= lm_full.Coefficients.tStat(idx_Int);

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

Yhat_null_SC= nan(nObs,nChan,nTime); Residual_null_SC = nan(nObs,nChan,nTime);

Yhat_null_SH = nan(nObs,nChan,nTime); Residual_null_SH = nan(nObs,nChan,nTime);

Yhat_null_Int = nan(nObs,nChan,nTime); Residual_null_Int = nan(nObs,nChan,nTime);


for ch = 1:nChan

    fprintf('Reduced model channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = squeeze(mEEG(:,ch,t));

        lm_null_SC = fitlm(X_null_SC,Y);

        Yhat_null_SC(:,ch,t) = lm_null_SC.Fitted;

        Residual_null_SC(:,ch,t) = lm_null_SC.Residuals.Raw;

        
        lm_null_SH = fitlm(X_null_SH,Y);

        Yhat_null_SH(:,ch,t) = lm_null_SH.Fitted;

        Residual_null_SH(:,ch,t) = lm_null_SH.Residuals.Raw;

                
        lm_null_Int = fitlm(X_null_Int,Y);

        Yhat_null_Int(:,ch,t) = lm_null_Int.Fitted;

        Residual_null_Int(:,ch,t) = lm_null_Int.Residuals.Raw;

    end
end

fprintf('Reduced models completed.\n');

%% ==========================================================
% TFCE observed map
%% ==========================================================
fprintf('\nComputing observed TFCE...\n');

ChN = ept_ChN2(channelinfo);

E_H = [0.66 2];


TFCE_Obs_SC = ept_mex_TFCE2D(t_Obs_SC, ChN, E_H);

TFCE_Obs_SH = ept_mex_TFCE2D(t_Obs_SH, ChN, E_H);

TFCE_Obs_Int = ept_mex_TFCE2D(t_Obs_Int, ChN, E_H);

fprintf('Observed TFCE completed.\n');

%% ==========================================================
% Freedman-Lane permutation
%% ==========================================================
nPerm=999;

TFCE_permMax_SC = zeros(nPerm,1);

TFCE_permMax_SH = zeros(nPerm,1);

TFCE_permMax_Int = zeros(nPerm,1);

fprintf('\nStarting Freedman-Lane permutations...\n');

delete(gcp('nocreate'));

mexFile = which('ept_mex_TFCE2D');

if isempty(mexFile)
    error('ept_mex_TFCE2D is not on the MATLAB path.');
end

pool = parpool('Processes',4, ...
    'AttachedFiles',{mexFile});


parfor p=1:nPerm
    
    fprintf('At %d/%d permutation\n', p, nPerm);

    % ------------------------------------
    % One permutation for all EEG points
    % ------------------------------------

    perm_idx = randperm(nObs);


    perm_t_SC = zeros(nChan,nTime);

    perm_t_SH = zeros(nChan,nTime);

    perm_t_Int = zeros(nChan,nTime);

    for ch=1:nChan

        for t=1:nTime

            % original reduced-model fitted values

            Y0_SC = Yhat_null_SC(:,ch,t);

            % permuted residuals

            e_perm_SC = Residual_null_SC(perm_idx,ch,t);

            % Freedman-Lane response

            Y_perm_SC = Y0_SC + e_perm_SC;

            % full model

            lm_perm_SC = fitlm(X_full,Y_perm_SC);

            coefNames_SC = lm_perm_SC.Coefficients.Properties.RowNames;


            idx_SC = contains(coefNames_SC,'x3');

            perm_t_SC(ch,t)= lm_perm_SC.Coefficients.tStat(idx_SC);


            %--- original reduced-model fitted values

            Y0_SH = Yhat_null_SH(:,ch,t);

            % permuted residuals

            e_perm_SH = Residual_null_SH(perm_idx,ch,t);

            % Freedman-Lane response

            Y_perm_SH = Y0_SH + e_perm_SH;

            % full model

            lm_perm_SH = fitlm(X_full,Y_perm_SH);

            coefNames_SH = lm_perm_SH.Coefficients.Properties.RowNames;

            idx_SH = contains(coefNames_SH,'x4');

            perm_t_SH(ch,t)= lm_perm_SH.Coefficients.tStat(idx_SH);


            % %--- original reduced-model fitted values

            Y0_Int = Yhat_null_Int(:,ch,t);

            % permuted residuals

            e_perm_Int = Residual_null_Int(perm_idx,ch,t);


            % Freedman-Lane response

            Y_perm_Int = Y0_Int + e_perm_Int;

            % full model

            lm_perm_Int = fitlm(X_full,Y_perm_Int);

            coefNames_Int = lm_perm_Int.Coefficients.Properties.RowNames;

            idx_Int = contains(coefNames,'x5');

            perm_t_Int(ch,t)= lm_perm_Int.Coefficients.tStat(idx_Int);

        end

    end

    % TFCE on this permutation

    TFCE_perm_SC = ept_mex_TFCE2D(perm_t_SC, ChN, E_H);

    TFCE_permMax_SC(p) = max(abs(TFCE_perm_SC(:)));



    TFCE_perm_SH = ept_mex_TFCE2D(perm_t_SH, ChN, E_H);

    TFCE_permMax_SH(p) = max(abs(TFCE_perm_SH(:)));
    % 


    TFCE_perm_Int = ept_mex_TFCE2D(perm_t_Int, ChN, E_H);

    TFCE_permMax_Int(p) = max(abs(TFCE_perm_Int(:)));
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


maxTFCE_SC = sort([TFCE_permMax_SC;max(abs(TFCE_Obs_SC(:)))]);

maxTFCEcrit_SC = maxTFCE_SC(round(nPerm*(1-alpha)));

fprintf('Critical TFCE value = %.4f\n',maxTFCEcrit_SC);

Mask_SC = abs(TFCE_Obs_SC) >= maxTFCEcrit_SC;


maxTFCE_SH = sort([TFCE_permMax_SH;max(abs(TFCE_Obs_SH(:)))]);

maxTFCEcrit_SH = maxTFCE_SH(round(nPerm*(1-alpha)));

fprintf('Critical TFCE value = %.4f\n',maxTFCEcrit_SH);

Mask_SH = abs(TFCE_Obs_SH) >= maxTFCEcrit_SH;


maxTFCE_Int = sort([TFCE_permMax_Int;max(abs(TFCE_Obs_Int(:)))]);

maxTFCEcrit_Int = maxTFCE_Int(round(nPerm*(1-alpha)));

fprintf('Critical TFCE value = %.4f\n',maxTFCEcrit_Int);

Mask_Int = abs(TFCE_Obs_Int) >= maxTFCEcrit_Int;


P_Values_SC = nan(nChan,nTime); P_Values_SH = nan(nChan,nTime); P_Values_Int = nan(nChan,nTime);

for ch=1:nChan

    for tp=1:nTime

        P_Values_SC(ch,tp)= ...
            (sum(TFCE_permMax_SC >= abs(TFCE_Obs_SC(ch,tp)))+1) /(nPerm+1);

        P_Values_SH(ch,tp)= ...
            (sum(TFCE_permMax_SH >= abs(TFCE_Obs_SH(ch,tp)))+1) /(nPerm+1);

        P_Values_Int(ch,tp)= ...
            (sum(TFCE_permMax_Int >= abs(TFCE_Obs_Int(ch,tp)))+1) /(nPerm+1);

    end
    
end

fprintf('TFCE correction completed.\n');

%% ==========================================================
% Store results
%% ==========================================================

Results=struct();

Results.t_Obs_SC = t_Obs_SC;

Results.TFCE_Obs_SC = TFCE_Obs_SC;

Results.TFCE_Null_SC = TFCE_permMax_SC;

Results.maxTFCEcrit_SC = maxTFCEcrit_SC;

Results.Mask_SC = Mask_SC;

Results.P_Values_SC = P_Values_SC;



Results.t_Obs_SH = t_Obs_SH;

Results.TFCE_Obs_SH = TFCE_Obs_SH;

Results.TFCE_Null_SH = TFCE_permMax_SH;

Results.maxTFCEcrit_SH = maxTFCEcrit_SH;

Results.Mask_SH = Mask_SH;

Results.P_Values_SH = P_Values_SH;



Results.t_Obs_Int = t_Obs_Int;

Results.TFCE_Obs_Int = TFCE_Obs_Int;

Results.TFCE_Null_Int = TFCE_permMax_Int;

Results.maxTFCEcrit_Int = maxTFCEcrit_Int;

Results.Mask_Int = Mask_Int;

Results.P_Values_Int = P_Values_Int;


Results.alpha = alpha;

Results.nPerm = nPerm;

Results.model = ...
    ['EEG ~ Stroke + ' ...
     'Freq + SC * SH'];

Results.test = ...
    'SH effect after removing subject/item random intercepts';

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

sigT = Results.t_Obs_Cla;

sigT(~Results.Mask_Cla)=0;

imagesc(time,1:nChan,Results.P_Values_Int < 0.05);

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

title('TFCE-corrected SH Effect');

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
