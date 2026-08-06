%% ==========================================================
% TFCE analysis
% Unbalanced between-subject 2 x 2 factorial design
%
% Input file contains:
%   EEGdata
%   designTable
%
% EEGdata:
%   Subjects x Channels x Time
%
% Full model:
%   EEG ~ FactorA + FactorB + FactorA:FactorB
%
% Tests:
%   1. Main effect of FactorA, adjusted for FactorB
%      and the FactorA x FactorB interaction
%
%   2. Main effect of FactorB, adjusted for FactorA
%      and the FactorA x FactorB interaction
%
%   3. FactorA x FactorB interaction, adjusted for
%      the two main effects
%
% Permutation:
%   Freedman-Lane residual permutation separately
%   for each tested effect
%% ==========================================================

clear all; clc; close all;

fprintf('\nStarting TFCE factorial analysis...\n');

rng(123);

%% ==========================================================
% Load data
%% ==========================================================

load('../Data/04_simulated_between_subject_2by2_EEG_unbalanced.mat');

var1 = double(designTable.FactorA(:));
var2 = double(designTable.FactorB(:));

times = -200:4:800;

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
    error( ...
        'Missing channels: %s', ...
        strjoin(chanLabels_32(~tf), ', ') ...
    );
end

e_loc = chanlocs_1020(idx);

%% ==========================================================
% Basic checks
%% ==========================================================

[nSubj, nChan, nTime] = size(EEGdata);

if height(designTable) ~= nSubj
    error( ...
        ['The number of rows in designTable does not match ' ...
         'the number of subjects in EEGdata.'] ...
    );
end

if length(var1) ~= nSubj
    error( ...
        'Length of FactorA does not match the number of subjects.' ...
    );
end

if length(var2) ~= nSubj
    error( ...
        'Length of FactorB does not match the number of subjects.' ...
    );
end

if length(times) ~= nTime
    error( ...
        'Length of times does not match the number of time points.' ...
    );
end

if length(e_loc) ~= nChan
    error( ...
        ['The number of channel locations does not match ' ...
         'the number of EEG channels.'] ...
    );
end

if any(isnan(var1)) || any(isnan(var2))
    error('FactorA and FactorB must not contain missing values.');
end

var1_levels = unique(var1);
var2_levels = unique(var2);

if numel(var1_levels) ~= 2
    error('FactorA must have exactly two levels.');
end

if numel(var2_levels) ~= 2
    error('FactorB must have exactly two levels.');
end

%% ==========================================================
% Display cell sample sizes
%% ==========================================================

fprintf('\nCell sample sizes:\n');

for a = 1:numel(var1_levels)
    for b = 1:numel(var2_levels)

        n_cell = sum( ...
            var1 == var1_levels(a) & ...
            var2 == var2_levels(b) ...
        );

        fprintf( ...
            'FactorA = %g, FactorB = %g: n = %d\n', ...
            var1_levels(a), ...
            var2_levels(b), ...
            n_cell ...
        );

    end
end

%% ==========================================================
% Design matrices
%
% fitlm automatically includes an intercept.
%
% Full model:
%   EEG ~ FactorA + FactorB + FactorA:FactorB
%
% Coefficient rows:
%   1 = Intercept
%   2 = FactorA
%   3 = FactorB
%   4 = FactorA x FactorB
%% ==========================================================

interaction = var1 .* var2;

X_full = [var1, var2, interaction];

%% Reduced model for testing FactorA
%
% Removes the FactorA column while retaining FactorB and
% the interaction term.
%
% This tests the FactorA coefficient in the full model.

X_red_var1 = [var2, interaction];

%% Reduced model for testing FactorB
%
% Removes the FactorB column while retaining FactorA and
% the interaction term.

X_red_var2 = [var1, interaction];

%% Reduced model for testing the interaction
%
% Retains both main effects and removes only the interaction.

X_red_Int = [var1, var2];

coefRow_var1 = 2;
coefRow_var2 = 3;
coefRow_Int  = 4;

%% ==========================================================
% TFCE settings
%% ==========================================================

nPerm = 999;
alpha = 0.05;

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

%% ==========================================================
% Step 1: Observed t-statistic maps
%% ==========================================================

fprintf('\nStep 1: Computing observed t-statistic maps...\n');

t_Obs_var1 = zeros(nChan, nTime);
t_Obs_var2 = zeros(nChan, nTime);
t_Obs_Int  = zeros(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        Y = double(squeeze(EEGdata(:, ch, tpoint)));

        lm_full = fitlm(X_full, Y);

        t_Obs_var1(ch, tpoint) = ...
            lm_full.Coefficients.tStat(coefRow_var1);

        t_Obs_var2(ch, tpoint) = ...
            lm_full.Coefficients.tStat(coefRow_var2);

        t_Obs_Int(ch, tpoint) = ...
            lm_full.Coefficients.tStat(coefRow_Int);

    end
end

fprintf('Observed t-statistic maps completed.\n');

%% ==========================================================
% Step 2: Observed TFCE maps
%% ==========================================================

fprintf('Step 2: Computing observed TFCE maps...\n');

TFCE_Obs_var1 = ept_mex_TFCE2D(t_Obs_var1, ChN, E_H );

TFCE_Obs_var2 = ept_mex_TFCE2D(t_Obs_var2, ChN, E_H );

TFCE_Obs_Int = ept_mex_TFCE2D(t_Obs_Int, ChN, E_H );

fprintf('Observed TFCE maps completed.\n');

%% ==========================================================
% Step 3: Freedman-Lane permutation testing
%% ==========================================================

TFCE_permMax_var1 = nan(nPerm, 1);
TFCE_permMax_var2 = nan(nPerm, 1);
TFCE_permMax_Int  = nan(nPerm, 1);

fprintf( ...
    ['Step 3: Starting Freedman-Lane permutation testing: ' ...
     '%d permutations...\n'], ...
    nPerm ...
);

parfor p = 1:nPerm

    
    perm_t_var1 = zeros(nChan, nTime);
    perm_t_var2 = zeros(nChan, nTime);
    perm_t_Int  = zeros(nChan, nTime);

    %% ------------------------------------------------------
    % Generate one subject-level permutation for each effect
    %
    % The same permutation is used across all channels and
    % time points for a given effect. This preserves the
    % spatial and temporal dependence structure.
    %% ------------------------------------------------------

    perm_idx_var1 = randperm(nSubj);
    perm_idx_var2 = randperm(nSubj);
    perm_idx_Int  = randperm(nSubj);

    %% ------------------------------------------------------
    % Mass-univariate permutation regression
    %% ------------------------------------------------------

    for ch = 1:nChan
        for tpoint = 1:nTime

            Y = double(squeeze(EEGdata(:, ch, tpoint)));

            %% ==================================================
            % FactorA main effect
            %
            % Reduced model:
            %   EEG ~ FactorB + FactorA:FactorB
            %
            % Full model:
            %   EEG ~ FactorA + FactorB + FactorA:FactorB
            %% ==================================================

            lm_red_var1 = fitlm(X_red_var1, Y);

            Y_hat_red_var1 = lm_red_var1.Fitted;
            resid_red_var1 = lm_red_var1.Residuals.Raw;

            Y_perm_var1 = ...
                Y_hat_red_var1 + ...
                resid_red_var1(perm_idx_var1);

            lm_perm_var1 = fitlm(X_full, Y_perm_var1);

            perm_t_var1(ch, tpoint) = ...
                lm_perm_var1.Coefficients.tStat(coefRow_var1);

            %% ==================================================
            % FactorB main effect
            %
            % Reduced model:
            %   EEG ~ FactorA + FactorA:FactorB
            %
            % Full model:
            %   EEG ~ FactorA + FactorB + FactorA:FactorB
            %% ==================================================

            lm_red_var2 = fitlm(X_red_var2, Y);

            Y_hat_red_var2 = lm_red_var2.Fitted;
            resid_red_var2 = lm_red_var2.Residuals.Raw;

            Y_perm_var2 = ...
                Y_hat_red_var2 + ...
                resid_red_var2(perm_idx_var2);

            lm_perm_var2 = fitlm(X_full, Y_perm_var2);

            perm_t_var2(ch, tpoint) = ...
                lm_perm_var2.Coefficients.tStat(coefRow_var2);

            %% ==================================================
            % FactorA x FactorB interaction
            %
            % Reduced model:
            %   EEG ~ FactorA + FactorB
            %
            % Full model:
            %   EEG ~ FactorA + FactorB + FactorA:FactorB
            %% ==================================================

            lm_red_Int = fitlm(X_red_Int, Y);

            Y_hat_red_Int = lm_red_Int.Fitted;
            resid_red_Int = lm_red_Int.Residuals.Raw;

            Y_perm_Int = ...
                Y_hat_red_Int + ...
                resid_red_Int(perm_idx_Int);

            lm_perm_Int = fitlm(X_full, Y_perm_Int);

            perm_t_Int(ch, tpoint) = ...
                lm_perm_Int.Coefficients.tStat(coefRow_Int);

        end
    end

    fprintf('Permutation %d of %d completed.\n', p, nPerm);

    %% ------------------------------------------------------
    % Apply TFCE
    %% ------------------------------------------------------

    TFCE_perm_var1 = ept_mex_TFCE2D( ...
        perm_t_var1, ...
        ChN, ...
        E_H ...
    );

    TFCE_perm_var2 = ept_mex_TFCE2D( ...
        perm_t_var2, ...
        ChN, ...
        E_H ...
    );

    TFCE_perm_Int = ept_mex_TFCE2D( ...
        perm_t_Int, ...
        ChN, ...
        E_H ...
    );

    %% ------------------------------------------------------
    % Store maximum absolute TFCE values
    %% ------------------------------------------------------

    TFCE_permMax_var1(p) = ...
        max(abs(TFCE_perm_var1(:)));

    TFCE_permMax_var2(p) = ...
        max(abs(TFCE_perm_var2(:)));

    TFCE_permMax_Int(p) = ...
        max(abs(TFCE_perm_Int(:)));

end

fprintf('\nPermutation testing completed.\n');

%% ==========================================================
% Step 4: TFCE-corrected significance
%% ==========================================================

fprintf('Step 4: Computing TFCE-corrected significance...\n');

%% ----------------------------------------------------------
% Add the observed maximum to each permutation distribution
%% ----------------------------------------------------------

maxTFCE_var1 = sort([
    TFCE_permMax_var1;
    max(abs(TFCE_Obs_var1(:)))
]);

maxTFCE_var2 = sort([
    TFCE_permMax_var2;
    max(abs(TFCE_Obs_var2(:)))
]);

maxTFCE_Int = sort([
    TFCE_permMax_Int;
    max(abs(TFCE_Obs_Int(:)))
]);

%% ----------------------------------------------------------
% Determine the upper alpha-level critical value
%% ----------------------------------------------------------

criticalIndex = ceil((nPerm + 1) * (1 - alpha));

criticalIndex = min( ...
    criticalIndex, ...
    nPerm + 1 ...
);

maxTFCEcrit_var1 = maxTFCE_var1(criticalIndex);
maxTFCEcrit_var2 = maxTFCE_var2(criticalIndex);
maxTFCEcrit_Int  = maxTFCE_Int(criticalIndex);

%% ----------------------------------------------------------
% Corrected significance masks
%% ----------------------------------------------------------

Mask_var1 = ...
    abs(TFCE_Obs_var1) >= maxTFCEcrit_var1;

Mask_var2 = ...
    abs(TFCE_Obs_var2) >= maxTFCEcrit_var2;

Mask_Int = ...
    abs(TFCE_Obs_Int) >= maxTFCEcrit_Int;

%% ----------------------------------------------------------
% TFCE-corrected p-values
%% ----------------------------------------------------------

P_Values_var1 = nan(nChan, nTime);
P_Values_var2 = nan(nChan, nTime);
P_Values_Int  = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime

        P_Values_var1(ch, tpoint) = ...
            ( ...
                sum( ...
                    TFCE_permMax_var1 >= ...
                    abs(TFCE_Obs_var1(ch, tpoint)) ...
                ) + 1 ...
            ) / (nPerm + 1);

        P_Values_var2(ch, tpoint) = ...
            ( ...
                sum( ...
                    TFCE_permMax_var2 >= ...
                    abs(TFCE_Obs_var2(ch, tpoint)) ...
                ) + 1 ...
            ) / (nPerm + 1);

        P_Values_Int(ch, tpoint) = ...
            ( ...
                sum( ...
                    TFCE_permMax_Int >= ...
                    abs(TFCE_Obs_Int(ch, tpoint)) ...
                ) + 1 ...
            ) / (nPerm + 1);

    end
end

fprintf('TFCE-corrected significance completed.\n');

fprintf( ...
    'Critical TFCE value for FactorA = %.4f\n', ...
    maxTFCEcrit_var1 ...
);

fprintf( ...
    'Critical TFCE value for FactorB = %.4f\n', ...
    maxTFCEcrit_var2 ...
);

fprintf( ...
    'Critical TFCE value for interaction = %.4f\n', ...
    maxTFCEcrit_Int ...
);

%% ==========================================================
% Step 5: Store results
%% ==========================================================

fprintf('Step 5: Storing results...\n');

Results = struct();

%% FactorA main effect

Results.tObs_var1 = t_Obs_var1;
Results.TFCE_Obs_var1 = TFCE_Obs_var1;
Results.TFCE_Null_var1 = TFCE_permMax_var1;
Results.maxTFCEcrit_var1 = maxTFCEcrit_var1;
Results.P_Values_var1 = P_Values_var1;
Results.Mask_var1 = Mask_var1;

%% FactorB main effect

Results.tObs_var2 = t_Obs_var2;
Results.TFCE_Obs_var2 = TFCE_Obs_var2;
Results.TFCE_Null_var2 = TFCE_permMax_var2;
Results.maxTFCEcrit_var2 = maxTFCEcrit_var2;
Results.P_Values_var2 = P_Values_var2;
Results.Mask_var2 = Mask_var2;

%% FactorA x FactorB interaction

Results.tObs_Int = t_Obs_Int;
Results.TFCE_Obs_Int = TFCE_Obs_Int;
Results.TFCE_Null_Int = TFCE_permMax_Int;
Results.maxTFCEcrit_Int = maxTFCEcrit_Int;
Results.P_Values_Int = P_Values_Int;
Results.Mask_Int = Mask_Int;

%% Analysis information

Results.alpha = alpha;
Results.nPerm = nPerm;

Results.model = ...
    'EEG ~ FactorA + FactorB + FactorA:FactorB';

Results.test_var1 = ...
    ['FactorA main effect adjusted for FactorB ' ...
     'and FactorA:FactorB'];

Results.test_var2 = ...
    ['FactorB main effect adjusted for FactorA ' ...
     'and FactorA:FactorB'];

Results.test_interaction = ...
    ['FactorA:FactorB interaction adjusted for ' ...
     'FactorA and FactorB'];

Results.permutation_var1 = ...
    ['Freedman-Lane residual permutation under reduced model: ' ...
     'EEG ~ FactorB + FactorA:FactorB'];

Results.permutation_var2 = ...
    ['Freedman-Lane residual permutation under reduced model: ' ...
     'EEG ~ FactorA + FactorA:FactorB'];

Results.permutation_interaction = ...
    ['Freedman-Lane residual permutation under reduced model: ' ...
     'EEG ~ FactorA + FactorB'];

fprintf('Results stored.\n');

%% ==========================================================
% Step 6: Save results
%% ==========================================================

if ~exist('../Results', 'dir')
    mkdir('../Results');
end

save( ...
    '../Results/04_TFCE_between_subject_2by2_unbalanced_results.mat', ...
    'Results', ...
    'nChan', ...
    'times', ...
    'e_loc' ...
);

disp('TFCE factorial analysis completed and saved.');

%% ==========================================================
% Step 7: Plot TFCE-corrected significant t-statistic maps
%% ==========================================================

clear all; close all; clc;

load( ...
    '../Results/04_TFCE_between_subject_2by2_unbalanced_results.mat', ...
    'Results', ...
    'nChan', ...
    'times', ...
    'e_loc' ...
);

%% ----------------------------------------------------------
% FactorA main effect
%% ----------------------------------------------------------

mT_var1 = Results.tObs_var1;
mT_var1(~Results.Mask_var1) = 0;

figure;

imagesc(times, 1:nChan, mT_var1);
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

title( ...
    'TFCE-corrected Main Effect: Factor A' ...
);

hColorbar = colorbar;

set( ...
    get(hColorbar, 'YLabel'), ...
    'String', 't-value', ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);

%% ----------------------------------------------------------
% FactorB main effect
%% ----------------------------------------------------------

mT_var2 = Results.tObs_var2;
mT_var2(~Results.Mask_var2) = 0;

figure;

imagesc(times, 1:nChan, mT_var2);
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

title( ...
    'TFCE-corrected Main Effect: Factor B' ...
);

hColorbar = colorbar;

set( ...
    get(hColorbar, 'YLabel'), ...
    'String', 't-value', ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);

%% ----------------------------------------------------------
% FactorA x FactorB interaction
%% ----------------------------------------------------------

mT_Int = Results.tObs_Int;
mT_Int(~Results.Mask_Int) = 0;

figure;

imagesc(times, 1:nChan, mT_Int);
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

title( ...
    'TFCE-corrected Interaction: Factor A x Factor B' ...
);

hColorbar = colorbar;

set( ...
    get(hColorbar, 'YLabel'), ...
    'String', 't-value', ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);

%% ==========================================================
% Step 8: Plot observed TFCE maps
%% ==========================================================

%% FactorA TFCE map

figure;

imagesc(times, 1:nChan, Results.TFCE_Obs_var1);
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

title( ...
    'Observed TFCE Map: Factor A Main Effect' ...
);

hColorbar = colorbar;

set( ...
    get(hColorbar, 'YLabel'), ...
    'String', 'TFCE value', ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);

%% FactorB TFCE map

figure;

imagesc(times, 1:nChan, Results.TFCE_Obs_var2);
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

title( ...
    'Observed TFCE Map: Factor B Main Effect' ...
);

hColorbar = colorbar;

set( ...
    get(hColorbar, 'YLabel'), ...
    'String', 'TFCE value', ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);

%% Interaction TFCE map

figure;

imagesc(times, 1:nChan, Results.TFCE_Obs_Int);
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

title( ...
    'Observed TFCE Map: Factor A x Factor B Interaction' ...
);

hColorbar = colorbar;

set( ...
    get(hColorbar, 'YLabel'), ...
    'String', 'TFCE value', ...
    'FontSize', 15, ...
    'FontName', 'Arial' ...
);


