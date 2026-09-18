%% Crossed LME + restricted Freedman-Lane + TFCE
% Tests: Classifier congruency and JSD x Classifier interaction
%
% Stage 1: At every channel/time point, fit:
%          (a) the full crossed LME to obtain the observed Classifier t; and
%          (b) effect-specific reduced crossed LMEs to obtain conditional
%              fitted values and conditional residuals.
%
% Stage 2: Generate each null outcome with restricted Freedman-Lane:
%          Y* = fitted_reduced_conditional + permuted_residual_reduced.
%          Refit the full crossed LME to Y* at every channel/time point,
%          apply TFCE, and retain the maximum absolute TFCE value.
%
% IMPORTANT ASSUMPTION:
% Classifier must vary within Subject x Item cells, and observations must be
% exchangeable within those cells under H0 after adjustment for covariates.

clear;
clc;
close all;

rng(123, 'twister');

fprintf(['\nStarting Classifier and JSD x Classifier crossed-LME ' ...
         '+ Freedman-Lane analysis...\n']);

%% Load data
load('../Data/realDOE_3.mat');

[nObs, nChan, nTime] = size(EEGdata);

if height(designTable) ~= nObs
    error('designTable rows do not match EEG observations.');
end

if numel(channelinfo) ~= nChan
    error('Channel information does not match EEG dimensions.');
end

fprintf('Observations: %d\n', nObs);
fprintf('Channels: %d\n', nChan);
fprintf('Time points: %d\n', nTime);

%% Prepare design variables
varNames = designTable.Properties.VariableNames;

% Subject
if ismember('SubjID', varNames)
    Subj = categorical(designTable.SubjID);
elseif ismember('SubjSubj', varNames)
    Subj = categorical(designTable.SubjSubj);
else
    error('No subject identifier found.');
end

% Item
if ~ismember('Target', varNames)
    error('Target variable missing.');
end
Item = categorical(designTable.Target);

% JSD: Low = -1; High = +1
tmpJSD = string(designTable.JSD);
JSD = nan(nObs, 1);
JSD(tmpJSD == "L") = -1;
JSD(tmpJSD == "H") =  1;
if any(isnan(JSD))
    error('Some JSD observations were not coded.');
end

% Classifier congruency: Incongruent = -1; Congruent = +1
tmpClassifier = string(designTable.ClassifierCongruency);
Classifier = nan(nObs, 1);
Classifier(tmpClassifier == "Incongruent") = -1;
Classifier(tmpClassifier == "Congruent")   =  1;
if any(isnan(Classifier))
    error('Some Classifier observations were not coded.');
end

% Semantic-category congruency: Incongruent/0 = -1; Congruent/1 = +1
tmpSemantic = string(designTable.CongruencySemanticCategories);
CongruencySemanticCategories = nan(nObs, 1);
CongruencySemanticCategories(tmpSemantic == "Incongruent" | ...
                             tmpSemantic == "0") = -1;
CongruencySemanticCategories(tmpSemantic == "Congruent" | ...
                             tmpSemantic == "1") =  1;
if any(isnan(CongruencySemanticCategories))
    disp('Unrecognised semantic-category labels:');
    disp(unique(tmpSemantic(isnan(CongruencySemanticCategories))));
    error('Some semantic-category observations were not coded.');
end

% Log frequency, preserving original zero as zero before standardisation
FreqOriginal = double(designTable.Frequency);
if any(FreqOriginal < 0 | ~isfinite(FreqOriginal))
    error('Frequency contains negative or invalid values.');
end
FreqRaw = zeros(nObs, 1);
positiveFreq = FreqOriginal > 0;
FreqRaw(positiveFreq) = log(FreqOriginal(positiveFreq));
Freq = zscore_checked(FreqRaw, 'Frequency');

% Stroke count
StrokeRaw = double(designTable.NumbersofStorks);
Stroke = zscore_checked(StrokeRaw, 'Stroke');

% Distractor length
LengthofDistrctorRaw = double(designTable.LengthofDistrctor);
LengthofDistrctor = zscore_checked(LengthofDistrctorRaw, ...
    'LengthofDistrctor');

fprintf('Classifier Congruent (+1): %d\n', sum(Classifier == 1));
fprintf('Classifier Incongruent (-1): %d\n', sum(Classifier == -1));

%% Validate the restricted exchangeability blocks
% Each block is one Subject x Item combination. Both Classifier levels must
% occur within a block for a within-cell Classifier test to be identified.
[cellID, cellSubj, cellItem] = findgroups(Subj, Item); %#ok<ASGLU>
% nCells = numel(cellSubj);
% cellSize = splitapply(@numel, Classifier, cellID);
% cellMin = splitapply(@min, Classifier, cellID);
% cellMax = splitapply(@max, Classifier, cellID);
% informativeCell = cellMin == -1 & cellMax == 1;
% 
% fprintf('Subject x Item cells: %d\n', nCells);
% fprintf('Cell-size range: %d to %d observations\n', ...
%     min(cellSize), max(cellSize));
% fprintf('Cells containing both Classifier levels: %d/%d\n', ...
%     sum(informativeCell), nCells);
% 
% if any(cellSize < 2)
%     error(['At least one Subject x Item cell has fewer than two trials. ' ...
%            'Residuals cannot be permuted within that cell.']);
% end
% 
% if any(~informativeCell)
%     warning(['Some Subject x Item cells do not contain both Classifier ' ...
%              'levels. Verify that the intended randomisation really was ' ...
%              'within every Subject x Item cell.']);
% end

%% Stage 1: observed full-LME statistic and reduced-LME components
fprintf('\nStage 1: fitting observed full and reduced crossed LMEs...\n');

fullFormula = [ ...
    'EEG ~ Stroke + Freq + LengthofDistrctor + ' ...
    'CongruencySemanticCategories + JSD*Classifier + ' ...
    '(1|Subj) + (1|Item)'];

% Coefficient-specific null for the Classifier main effect. The interaction
% is retained, so this tests beta_Classifier = 0 at centred JSD = 0.
nullFormulaCla = [ ...
    'EEG ~ Stroke + Freq + LengthofDistrctor + ' ...
    'CongruencySemanticCategories + JSD + JSD:Classifier + ' ...
    '(1|Subj) + (1|Item)'];

% Null for the interaction: retain both constituent main effects.
nullFormulaInt = [ ...
    'EEG ~ Stroke + Freq + LengthofDistrctor + ' ...
    'CongruencySemanticCategories + JSD + Classifier + ' ...
    '(1|Subj) + (1|Item)'];

t_Obs_Cla = nan(nChan, nTime);


YhatNullCla = nan(nObs, nChan, nTime);
ResidualNullCla = nan(nObs, nChan, nTime);


for ch = 1:nChan
    fprintf('Observed/reduced LME channel %d/%d\n', ch, nChan);

    for t = 1:nTime
        Y = squeeze(double(EEGdata(:, ch, t)));

        tbl = table( ...
            Y, Stroke, Freq, LengthofDistrctor, ...
            CongruencySemanticCategories, JSD, Classifier, Subj, Item, ...
            'VariableNames', { ...
                'EEG', 'Stroke', 'Freq', 'LengthofDistrctor', ...
                'CongruencySemanticCategories', 'JSD', 'Classifier', ...
                'Subj', 'Item'});

        % Observed statistic from the full crossed mixed model.
        lmeFull = fitlme(tbl, fullFormula, 'FitMethod', 'REML');
        coefNames = string(lmeFull.CoefficientNames);
        idxClassifier = coefNames == "Classifier";

        if nnz(idxClassifier) ~= 1
            error('Could not uniquely identify the Classifier coefficient.');
        end

        t_Obs_Cla(ch, t) = lmeFull.Coefficients.tStat(idxClassifier);

        % Effect-specific reduced models.
        lmeNullCla = fitlme(tbl, nullFormulaCla, 'FitMethod', 'REML');

        % Preserve both nuisance fixed effects and estimated subject/item
        % random-intercept contributions. Only conditional level-1
        % residuals are subsequently permuted.
        YhatNullCla(:, ch, t) = fitted(lmeNullCla, 'Conditional', false);
        ResidualNullCla(:, ch, t) = residuals(lmeNullCla, 'Conditional', false);
    end
end

%% Stage 2: restricted Freedman-Lane, full-LME refitting, and TFCE
nPerm = 999;
tfceParameters = [0.66 2];
rperms = lmeEEG_permutations2(nPerm, Subj, Item);

% ept_TFCE channel-neighbour matrix.
ChN = ept_ChN2(channelinfo);

TFCE_Obs_Cla = ept_mex_TFCE2D(t_Obs_Cla, ChN, tfceParameters);

maxTFCE_Cla = nan(nPerm, 1);


fprintf(['\nStage 2: %d restricted Freedman-Lane permutations; ' ...
         'refitting the full LME at every sample...\n'], nPerm);

for p = 1:nPerm
    % One common permutation for every channel/time point. Residual rows are
    % rearranged only within the same Subject x Item cell.
    permIdx = rperms(:,p);

    tPermClassifier = nan(nChan, nTime);

    for ch = 1:nChan
        for t = 1:nTime
            % Effect-specific Freedman-Lane pseudo-outcomes under H0.
            YpermCla = YhatNullCla(:, ch, t) + ...
                ResidualNullCla(permIdx, ch, t);

            tblPermCla = table( ...
                YpermCla, Stroke, Freq, LengthofDistrctor, ...
                CongruencySemanticCategories, JSD, Classifier, ...
                Subj, Item, ...
                'VariableNames', { ...
                    'EEG', 'Stroke', 'Freq', 'LengthofDistrctor', ...
                    'CongruencySemanticCategories', 'JSD', ...
                    'Classifier', 'Subj', 'Item'});


            % The same full LME and the same t-statistic as observed.
            lmePermCla = fitlme(tblPermCla, fullFormula, 'FitMethod', 'REML');

            permCoefNamesCla = string(lmePermCla.CoefficientNames);

            permClassifierIdx = permCoefNamesCla == "Classifier"; 

            if nnz(permClassifierIdx) ~= 1
                error(['Could not uniquely identify Classifier in ' ...
                       'permutation %d, channel %d, time %d.'], ...
                      p, ch, t);
            end

            tPermClassifier(ch, t) = lmePermCla.Coefficients.tStat(permClassifierIdx);

        end
    end

    TFCE_permMax_Cla = ept_mex_TFCE2D( ...
        tPermClassifier, ChN, tfceParameters);

    maxTFCE_Cla(p) = max(abs(TFCE_permMax_Cla), [], 'all');


    if mod(p, 25) == 0 || p == nPerm
        fprintf('Permutation %d/%d\n', p, nPerm);
    end
end

%% ==========================================================
fprintf('\nComputing TFCE-corrected significance...\n');

alpha = 0.05;

% Critical value from the permutation distribution
maxTFCE = sort([maxTFCE_Cla;max(abs(TFCE_Obs_Cla(:)))]);
maxTFCEcrit = maxTFCE(round(nPerm*(1-alpha)));

fprintf('Critical TFCE value = %.4f\n', maxTFCEcrit);

% Two-sided TFCE significance mask
Mask_Cla = abs(TFCE_Obs_Cla) >= maxTFCEcrit;

% Permutation-corrected p-values
P_Values_Cla = nan(nChan,nTime);

for ch = 1:nChan
    for tp = 1:nTime
        P_Values_Cla(ch,tp) = ...
            (sum(maxTFCE_Cla >= abs(TFCE_Obs_Cla(ch,tp))) + 1) ...
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

Results.TFCE_permMax_Cla = maxTFCE_Cla;

Results.maxTFCEcrit = maxTFCEcrit;

Results.Mask_Cla = Mask_Cla;

Results.P_Values_Cla = P_Values_Cla;

%% ==========================================================
% Save
%% ==========================================================
if ~exist('../Results','dir')

    mkdir('../Results');

end

save('../Results/09_realDOE_results_5.mat',...
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
% load('../Results/09_realDOE_results_5.mat')

figure;

sigT = Results.t_Obs_Cla;

sigT(~Results.Mask_Cla) = 0;

imagesc(time, 1:nChan, Results.P_Values_Cla < 0.3);

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
