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

% lev = categories(tmp);
% 
% if length(lev)~=2
% 
%     error('CongruencySemanticCategories must have two levels.');
% 
% end
% 
% CongruencySemanticCategories=zeros(nObs,1);
% 
% CongruencySemanticCategories(tmp==lev{1})=-1;
% CongruencySemanticCategories(tmp==lev{2})=1;
% 
% %% ----------------------------
% % Length of distractor
% %% ----------------------------
% 
LengthofDistrctorRaw = designTable.LengthofDistrctor;
LengthofDistrctor = (LengthofDistrctorRaw - mean(LengthofDistrctorRaw)) ./ std(LengthofDistrctorRaw);
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
    'EEG ~ Stroke + Freq + CongruencySemanticCategories + LengthofDistrctor + JSD * Classifier + (1|Subj)+(1|Item)';

for ch = 1:nChan

    for t = 1:nTime

        Y = squeeze(double(EEGdata(:,ch,t)));

        tbl = table( ...
            Y,...
            Stroke,...
            Freq,...
            CongruencySemanticCategories,...
            LengthofDistrctor, ...
            JSD,...
            Classifier,...
            Subj,...
            Item,...
            'VariableNames',...
            {'EEG',...
             'Stroke',...
             'Freq',...
             'CongruencySemanticCategories',...
             'LengthofDistrctor',...
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

X_full = [Stroke, Freq, CongruencySemanticCategories, LengthofDistrctor, JSD, Classifier, JSD .* Classifier];

X_null_JSD = [Stroke, Freq, CongruencySemanticCategories, LengthofDistrctor, Classifier, JSD .* Classifier];

X_null_Cla = [Stroke, Freq, CongruencySemanticCategories, LengthofDistrctor, JSD, JSD .* Classifier];

X_null_Int = [Stroke, Freq, CongruencySemanticCategories, LengthofDistrctor, JSD, Classifier];

%% ==========================================================
% Observed statistics from the full model
%% ==========================================================
fprintf('\nComputing observed t-map...\n');

t_Obs_JSD = nan(nChan,nTime);

t_Obs_Cla = nan(nChan,nTime);

t_Obs_Int = nan(nChan,nTime);

for ch = 1:nChan

    for t = 1:nTime

        Y=squeeze(mEEG(:,ch,t));

        lm_full = fitlm(X_full,Y);

        coefNames = lm_full.Coefficients.Properties.RowNames;

        idx_JSD = contains(coefNames,'x5');

        idx_Cla = contains(coefNames,'x6');

        idx_Int = contains(coefNames,'x7');


        t_Obs_JSD(ch,t)= lm_full.Coefficients.tStat(idx_JSD);

        t_Obs_Cla(ch,t)= lm_full.Coefficients.tStat(idx_Cla);

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

Yhat_null_JSD = nan(nObs,nChan,nTime); Residual_null_JSD = nan(nObs,nChan,nTime);

Yhat_null_Cla = nan(nObs,nChan,nTime); Residual_null_Cla = nan(nObs,nChan,nTime);

Yhat_null_Int = nan(nObs,nChan,nTime); Residual_null_Int = nan(nObs,nChan,nTime);


for ch = 1:nChan

    fprintf('Reduced model channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = squeeze(mEEG(:,ch,t));

        lm_null_JSD = fitlm(X_null_JSD,Y);

        Yhat_null_JSD(:,ch,t) = lm_null_JSD.Fitted;

        Residual_null_JSD(:,ch,t) = lm_null_JSD.Residuals.Raw;

        
        lm_null_Cla = fitlm(X_null_Cla,Y);

        Yhat_null_Cla(:,ch,t) = lm_null_Cla.Fitted;

        Residual_null_Cla(:,ch,t) = lm_null_Cla.Residuals.Raw;

                
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


TFCE_Obs_JSD = ept_mex_TFCE2D(t_Obs_JSD, ChN, E_H);

TFCE_Obs_Cla = ept_mex_TFCE2D(t_Obs_Cla, ChN, E_H);

TFCE_Obs_Int = ept_mex_TFCE2D(t_Obs_Int, ChN, E_H);

fprintf('Observed TFCE completed.\n');

%% ==========================================================
% Freedman-Lane permutation
%% ==========================================================
nPerm=999;

TFCE_permMax_JSD = zeros(nPerm,1);

TFCE_permMax_Cla = zeros(nPerm,1);

% TFCE_permMax_Int = zeros(nPerm,1);

fprintf('\nStarting Freedman-Lane permutations...\n');

delete(gcp('nocreate'));
parpool('Processes', 4);


parfor p=1:nPerm
    
    fprintf('At %d/%d permutation\n', p, nPerm);

    % ------------------------------------
    % One permutation for all EEG points
    % ------------------------------------

    perm_idx = randperm(nObs);


    perm_t_JSD = zeros(nChan,nTime);

    perm_t_Cla = zeros(nChan,nTime);

    % perm_t_Int = zeros(nChan,nTime);

    for ch=1:nChan

        for t=1:nTime

            % original reduced-model fitted values

            Y0_JSD = Yhat_null_JSD(:,ch,t);

            % permuted residuals

            e_perm_JSD = Residual_null_JSD(perm_idx,ch,t);

            % Freedman-Lane response

            Y_perm_JSD = Y0_JSD + e_perm_JSD;

            % full model

            lm_perm_JSD = fitlm(X_full,Y_perm_JSD);

            coefNames_JSD = lm_perm_JSD.Coefficients.Properties.RowNames;

            idx_JSD = contains(coefNames_JSD,'x5');

            perm_t_JSD(ch,t)= lm_perm_JSD.Coefficients.tStat(idx_JSD);


            %--- original reduced-model fitted values

            Y0_Cla = Yhat_null_Cla(:,ch,t);

            % permuted residuals

            e_perm_Cla = Residual_null_Cla(perm_idx,ch,t);

            % Freedman-Lane response

            Y_perm_Cla = Y0_Cla + e_perm_Cla;

            % full model

            lm_perm_Cla = fitlm(X_full,Y_perm_Cla);

            coefNames_Cla = lm_perm_Cla.Coefficients.Properties.RowNames;

            idx_Cla = contains(coefNames_Cla,'x6');

            perm_t_Cla(ch,t)= lm_perm_Cla.Coefficients.tStat(idx_Cla);


            % %--- original reduced-model fitted values

            Y0_Int = Yhat_null_Int(:,ch,t);

            % permuted residuals

            e_perm_Int = Residual_null_Int(perm_idx_Int,ch,t);

            % Freedman-Lane response

            Y_perm_Int = Y0_Int + e_perm_Int;

            % full model

            lm_perm_Int = fitlm(X_full,Y_perm_Int);

            coefNames_Int = lm_perm_Int.Coefficients.Properties.RowNames;

            idx_Int = contains(coefNames,'x7');

            perm_t_Int(ch,t)= lm_perm_Int.Coefficients.tStat(idx_Int);

        end

    end

    TFCE on this permutation

    TFCE_perm_JSD = ept_mex_TFCE2D(perm_t_JSD, ChN, E_H);

    TFCE_permMax_JSD(p) = max(abs(TFCE_perm_JSD(:)));



    TFCE_perm_Cla = ept_mex_TFCE2D(perm_t_Cla, ChN, E_H);

    TFCE_permMax_Cla(p) = max(abs(TFCE_perm_Cla(:)));
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


maxTFCE_JSD = sort([TFCE_permMax_JSD;max(abs(TFCE_Obs_JSD(:)))]);

maxTFCEcrit_JSD = maxTFCE_JSD(round(nPerm*(1-alpha)));

fprintf('Critical TFCE value = %.4f\n',maxTFCEcrit_JSD);

Mask_JSD = abs(TFCE_Obs_JSD) >= maxTFCEcrit_JSD;


maxTFCE_Cla = sort([TFCE_permMax_Cla;max(abs(TFCE_Obs_Cla(:)))]);

maxTFCEcrit_Cla = maxTFCE_Cla(round(nPerm*(1-alpha)));

fprintf('Critical TFCE value = %.4f\n',maxTFCEcrit_Cla);

Mask_Cla = abs(TFCE_Obs_Cla) >= maxTFCEcrit_Cla;


maxTFCE_Int = sort([TFCE_permMax_Int;max(abs(TFCE_Obs_Int(:)))]);

maxTFCEcrit_Int = maxTFCE_Int(round(nPerm*(1-alpha)));

fprintf('Critical TFCE value = %.4f\n',maxTFCEcrit_Int);

Mask_Int = abs(TFCE_Obs_Int) >= maxTFCEcrit_Int;


P_Values_JSD = nan(nChan,nTime); P_Values_Cla = nan(nChan,nTime); P_Values_Int = nan(nChan,nTime);

for ch=1:nChan

    for tp=1:nTime

        P_Values_JSD(ch,tp)= ...
            (sum(TFCE_permMax_JSD >= abs(TFCE_Obs_JSD(ch,tp)))+1) /(nPerm+1);

        P_Values_Cla(ch,tp)= ...
            (sum(TFCE_permMax_Cla >= abs(TFCE_Obs_Cla(ch,tp)))+1) /(nPerm+1);

        P_Values_Int(ch,tp)= ...
            (sum(TFCE_permMax_Int >= abs(TFCE_Obs_Int(ch,tp)))+1) /(nPerm+1);

    end
    
end

fprintf('TFCE correction completed.\n');

%% ==========================================================
% Store results
%% ==========================================================

Results=struct();

Results.t_Obs_JSD = t_Obs_JSD;

Results.TFCE_Obs_JSD = TFCE_Obs_JSD;

Results.TFCE_Null_JSD = TFCE_permMax_JSD;

Results.maxTFCEcrit_JSD = maxTFCEcrit_JSD;

Results.Mask_JSD = Mask_JSD;

Results.P_Values_JSD = P_Values_JSD;



Results.t_Obs_Cla = t_Obs_Cla;

Results.TFCE_Obs_Cla = TFCE_Obs_Cla;

Results.TFCE_Null_Cla = TFCE_permMax_Cla;

Results.maxTFCEcrit_Cla = maxTFCEcrit_Cla;

Results.Mask_Cla = Mask_Cla;

Results.P_Values_Cla = P_Values_Cla;



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
     'Freq + CongruencySemanticCategories + LengthofDistrctor + JSD * Classifier'];



Results.test = ...
    'Classifier effect after removing subject/item random intercepts';

fprintf('Results stored.\n');

%% ==========================================================
% Save
%% ==========================================================

if ~exist('../Results','dir')

    mkdir('../Results');

end

save('../Results/09_realDOE_results_4.mat',...
    'Results',...
    'nChan',...
    'time',...
    'channelinfo');

fprintf('Saved results.\n')

%% ==========================================================
% Plot corrected t-map
%% ==========================================================

clear all; clc; close all

load('../Results/09_realDOE_results_4.mat')

figure;

sigT = Results.tObs;

sigT(~Results.Mask)=0;

imagesc(time,1:nChan,Results.P_Values<0.2);

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
