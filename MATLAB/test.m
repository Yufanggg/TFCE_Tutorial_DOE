%% ==========================================================
% Minimal lmeEEG + Freedman-Lane + TFCE
% Test only: SemanticCategory (SC) effect
%% ==========================================================

clear; clc; close all;

rng(123);

fprintf('\nStarting minimal SC analysis...\n');

%% ==========================================================
% Load data
%% ==========================================================

load('../Data/09_realDOE.mat');

[nObs,nChan,nTime] = size(EEGdata);

%% ==========================================================
% Prepare variables
%% ==========================================================

varNames = designTable.Properties.VariableNames;

% Subject
if ismember('SubjID',varNames)
    Subj = categorical(designTable.SubjID);
else
    error('No subject identifier found.');
end

% Item
Item = categorical(designTable.Target);

% Semantic Category
SemanticCategory = zeros(nObs,1);
tmpSC = string(designTable.SemanticCategory);

SemanticCategory(tmpSC == "0") = -1;
SemanticCategory(tmpSC == "1") =  1;

if any(SemanticCategory == 0)
    error('Some SemanticCategory observations were not coded.');
end

% Shape
Shape = zeros(nObs,1);
tmpSH = string(designTable.Shape);

Shape(tmpSH == "0") = -1;
Shape(tmpSH == "1") =  1;

if any(Shape == 0)
    error('Some Shape observations were not coded.');
end

%% ==========================================================
% Stage 1
% Remove Subject + Item random-effect contribution
%% ==========================================================

fprintf('\nRemoving random effects...\n');

mEEG = nan(size(EEGdata));

lmeFormula = ...
    ['EEG ~ SemanticCategory*Shape ' ...
     '+ (1|Subj) + (1|Item)'];

for ch = 1:nChan

    fprintf('LME channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = squeeze(double(EEGdata(:,ch,t)));

        tbl = table( ...
            Y, ...
            SemanticCategory, ...
            Shape, ...
            Subj, ...
            Item, ...
            'VariableNames', { ...
                'EEG', ...
                'SemanticCategory', ...
                'Shape', ...
                'Subj', ...
                'Item'});

        lme = fitlme(tbl,lmeFormula);

        randomContribution = ...
            fitted(lme,'Conditional',true) - ...
            fitted(lme,'Conditional',false);

        mEEG(:,ch,t) = Y - randomContribution;

    end
end

fprintf('Random-effect removal completed.\n');


%% ==========================================================
% Stage 2
% Define full and reduced models for SC
%% ==========================================================

% Full:
% EEG ~ Freq + Stroke + SC + SH + SC*SH
X_full_SC = [ ...
    SemanticCategory, ...
    Shape, ...
    SemanticCategory .* Shape];

% Reduced:
% EEG ~ Freq + Stroke + SH + SC*SH
X_null_SC = [ ...
    Shape, ...
    SemanticCategory .* Shape];


%% ==========================================================
% Observed SC t-map
%% ==========================================================

fprintf('\nComputing observed SC t-map...\n');

t_Obs_SC = nan(nChan,nTime);

for ch = 1:nChan

    for t = 1:nTime

        Y = squeeze(mEEG(:,ch,t));

        lm_full = fitlm(X_full_SC,Y);

        % fitlm adds intercept:
        %
        % row 1 = intercept
        % row 2 = SemanticCategory
        % row 3 = Shape
        % row 4 = interaction

        t_Obs_SC(ch,t) = ...
            lm_full.Coefficients.tStat(2);

    end
end

fprintf('Observed SC t-map completed.\n');


%% ==========================================================
% Reduced model for Freedman-Lane
%% ==========================================================

fprintf('\nComputing reduced-model fitted values and residuals...\n');

Yhat_null_SC = nan(nObs,nChan,nTime);
Residual_null_SC = nan(nObs,nChan,nTime);

for ch = 1:nChan

    fprintf('Reduced LM channel %d/%d\n',ch,nChan);

    for t = 1:nTime

        Y = squeeze(mEEG(:,ch,t));

        lm_null = fitlm(X_null_SC,Y);

        Yhat_null_SC(:,ch,t) = ...
            lm_null.Fitted;

        Residual_null_SC(:,ch,t) = ...
            lm_null.Residuals.Raw;

    end
end


%% ==========================================================
% Sanity check
%% ==========================================================

Y_test = squeeze(mEEG(:,1,1));

reconstruction_error = norm( ...
    Y_test - ...
    (Yhat_null_SC(:,1,1) + Residual_null_SC(:,1,1)));

fprintf('\nReconstruction error = %.12f\n', ...
    reconstruction_error);


%% ==========================================================
% Observed TFCE
%% ==========================================================

ChN = ept_ChN2(channelinfo);
E_H = [0.66 2];

TFCE_Obs_SC = ...
    ept_mex_TFCE2D(t_Obs_SC,ChN,E_H);

fprintf('\nMax observed |t|    = %.4f\n', ...
    max(abs(t_Obs_SC(:))));

fprintf('Max observed |TFCE| = %.4f\n', ...
    max(abs(TFCE_Obs_SC(:))));


%% ==========================================================
% Generate restricted permutations
%% ==========================================================

nPerm = 10;

fprintf('\nGenerating restricted permutations...\n');

rperms = ...
    lmeEEG_permutations2(nPerm,Subj,Item);

original_idx = (1:nObs)';

changed_fraction = ...
    mean(rperms ~= original_idx,1);

fprintf('Mean proportion observations moved = %.4f\n', ...
    mean(changed_fraction));


%% ==========================================================
% Freedman-Lane permutation
%% ==========================================================

fprintf('\nStarting permutations...\n');

TFCE_permMax_SC = zeros(nPerm,1);

for p = 1:nPerm

    fprintf('Permutation %d/%d\n',p,nPerm);

    perm_idx = rperms(:,p);

    perm_t_SC = zeros(nChan,nTime);

    for ch = 1:nChan

        for t = 1:nTime

            % Reduced-model fitted component
            Y0 = Yhat_null_SC(:,ch,t);

            % Restricted permutation of residuals
            e_perm = ...
                Residual_null_SC(perm_idx,ch,t);

            % Freedman-Lane pseudo-response
            Y_perm = Y0 + e_perm;

            % Fit full ordinary linear model
            lm_perm = fitlm(X_full_SC,Y_perm);

            % SemanticCategory coefficient
            perm_t_SC(ch,t) = ...
                lm_perm.Coefficients.tStat(2);

        end
    end

    % TFCE
    TFCE_perm = ...
        ept_mex_TFCE2D(perm_t_SC,ChN,E_H);

    % Max statistic
    TFCE_permMax_SC(p) = ...
        max(abs(TFCE_perm(:)));

end


%% ==========================================================
% Corrected inference
%% ==========================================================

fprintf('\nComputing corrected inference...\n');

alpha = 0.05;

TFCEcrit_SC = ...
    prctile(TFCE_permMax_SC,100*(1-alpha));

Mask_SC = ...
    abs(TFCE_Obs_SC) >= TFCEcrit_SC;

P_Values_SC = nan(nChan,nTime);

for ch = 1:nChan

    for t = 1:nTime

        P_Values_SC(ch,t) = ...
            (sum(TFCE_permMax_SC >= ...
            abs(TFCE_Obs_SC(ch,t))) + 1) ...
            / (nPerm + 1);

    end
end


%% ==========================================================
% Diagnostics
%% ==========================================================

fprintf('\n===== SC DIAGNOSTIC =====\n');

fprintf('Max |t observed|      = %.4f\n', ...
    max(abs(t_Obs_SC(:))));

fprintf('Max |TFCE observed|   = %.4f\n', ...
    max(abs(TFCE_Obs_SC(:))));

fprintf('Null TFCE median      = %.4f\n', ...
    median(TFCE_permMax_SC));

fprintf('Null TFCE 95%%         = %.4f\n', ...
    TFCEcrit_SC);

fprintf('Null TFCE maximum     = %.4f\n', ...
    max(TFCE_permMax_SC));

fprintf('Minimum corrected p   = %.4f\n', ...
    min(P_Values_SC(:)));

fprintf('Significant points    = %d\n', ...
    sum(Mask_SC(:)));


%% ==========================================================
% Plot observed t-map
%% ==========================================================

figure;

imagesc(time,1:nChan,t_Obs_SC);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('Observed Semantic Category t-map');

colorbar;


%% ==========================================================
% Plot observed TFCE map
%% ==========================================================

figure;

imagesc(time,1:nChan,TFCE_Obs_SC);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('Observed Semantic Category TFCE map');

colorbar;


%% ==========================================================
% Plot permutation null distribution
%% ==========================================================

figure;

histogram(TFCE_permMax_SC,30);

hold on;

xline( ...
    max(abs(TFCE_Obs_SC(:))), ...
    'LineWidth',2);

xline( ...
    TFCEcrit_SC, ...
    '--', ...
    'LineWidth',2);

xlabel('Maximum |TFCE|');
ylabel('Permutation count');

title('SC TFCE null distribution');

legend( ...
    'Permutation null', ...
    'Observed max TFCE', ...
    '95% critical value');


%% ==========================================================
% Plot corrected SC t-map
%% ==========================================================

sigT = t_Obs_SC;

sigT(~Mask_SC) = 0;

figure;

imagesc(time,1:nChan,t_Obs_SC);

axis xy;

xlabel('Time (ms)');
ylabel('Channel');

title('TFCE-corrected Semantic Category effect');

colorbar;