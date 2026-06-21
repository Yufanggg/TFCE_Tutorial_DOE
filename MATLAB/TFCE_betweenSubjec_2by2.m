
clear all; clc; close all;

load('data/02_simulated_between_subject_2by2_EEG.mat')
chanlocs_1020 = readlocs('standard_1005.elc');

chanLabels_32 = {
'Fp1','Fp2',...
'F7','F3','Fz','F4','F8',...
'FC5','FC1','FC2','FC6',...
'T7','C3','Cz','C4','T8',...
'CP5','CP1','CP2','CP6',...
'P7','P3','Pz','P4','P8',...
'PO9','O1','Oz','O2','PO10',...
'TP9','TP10'
};
allLabels = {chanlocs_1020.labels};
[tf, idx] = ismember(chanLabels_32, allLabels);

if any(~tf)
    error('Missing channels: %s', strjoin(chanLabels_32(~tf), ', '));
end

%% Setup
e_loc = chanlocs_1020(idx);

[nSubj, nChan, nTime] = size(data);

X_obs = [var1(:), var2(:)];

t_Obs_var1 = zeros(nChan, nTime);
t_Obs_var2 = zeros(nChan, nTime);

% Step 1: Observed regression t-values
for ch = 1:nChan
    for tp = 1:nTime
        EEG_local = double(data(:, ch, tp));

        lm = fitlm(X_obs, EEG_local(:));

        t_Obs_var1(ch, tp) = lm.Coefficients.tStat(2);
        t_Obs_var2(ch, tp) = lm.Coefficients.tStat(3);
    end
end

% Step 2: TFCE on observed t-maps
ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs_var1 = ept_mex_TFCE2D(t_Obs_var1, ChN, E_H);
TFCE_Obs_var2 = ept_mex_TFCE2D(t_Obs_var2, ChN, E_H);

% Step 3: Synchronized permutation test
nPerms = 999;

TFCE_permMax_var1 = nan(nPerms, 1);
TFCE_permMax_var2 = nan(nPerms, 1);

idx_Bminus = find(var2 == -1);
idx_Bplus  = find(var2 ==  1);

idx_Aminus = find(var1 == -1);
idx_Aplus  = find(var1 ==  1);

parfor p = 1:nPerms

    % Permute Factor A within each level of Factor B
    var1_perm = var1(:);

    var1_perm(idx_Bminus) = var1_perm(idx_Bminus(randperm(numel(idx_Bminus))));
    var1_perm(idx_Bplus)  = var1_perm(idx_Bplus(randperm(numel(idx_Bplus))));

    X_perm_A = [var1_perm(:), var2(:)];

    % Permute Factor B within each level of Factor A
    var2_perm = var2(:);

    var2_perm(idx_Aminus) = var2_perm(idx_Aminus(randperm(numel(idx_Aminus))));
    var2_perm(idx_Aplus)  = var2_perm(idx_Aplus(randperm(numel(idx_Aplus))));

    X_perm_B = [var1(:), var2_perm(:)];

    % Fit permutation models
    perm_t_var1 = zeros(nChan, nTime);
    perm_t_var2 = zeros(nChan, nTime);

    for ch = 1:nChan
        for tp = 1:nTime
            EEG_local = double(data(:, ch, tp));

            lm_A = fitlm(X_perm_A, EEG_local(:));
            lm_B = fitlm(X_perm_B, EEG_local(:));

            perm_t_var1(ch, tp) = lm_A.Coefficients.tStat(2);
            perm_t_var2(ch, tp) = lm_B.Coefficients.tStat(3);
        end
    end

    % TFCE permutation max statistics
    TFCE_perm_var1 = ept_mex_TFCE2D(perm_t_var1, ChN, E_H);
    TFCE_perm_var2 = ept_mex_TFCE2D(perm_t_var2, ChN, E_H);

    TFCE_permMax_var1(p) = max(abs(TFCE_perm_var1(:)));
    TFCE_permMax_var2(p) = max(abs(TFCE_perm_var2(:)));
end

% Step 4: TFCE-corrected p-values and masks
Alpha = 0.05;

maxTFCE_var1 = sort([TFCE_permMax_var1; max(abs(TFCE_Obs_var1(:)))]);
maxTFCE_var2 = sort([TFCE_permMax_var2; max(abs(TFCE_Obs_var2(:)))]);

critIdx = round((nPerms + 1) * (1 - Alpha));

maxTFCEcrit_var1 = maxTFCE_var1(critIdx);
maxTFCEcrit_var2 = maxTFCE_var2(critIdx);

Mask_var1 = abs(TFCE_Obs_var1) >= maxTFCEcrit_var1;
Mask_var2 = abs(TFCE_Obs_var2) >= maxTFCEcrit_var2;

P_Values_var1 = nan(nChan, nTime);
P_Values_var2 = nan(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime
        P_Values_var1(ch, tp) = ...
            sum(abs(TFCE_Obs_var1(ch, tp)) <= maxTFCE_var1) / (nPerms + 1);

        P_Values_var2(ch, tp) = ...
            sum(abs(TFCE_Obs_var2(ch, tp)) <= maxTFCE_var2) / (nPerms + 1);
    end
end

% Store results
Results.Obs_var1       = t_Obs_var1;
Results.TFCE_Obs_var1  = TFCE_Obs_var1;
Results.maxTFCE_var1   = maxTFCE_var1;
Results.P_Values_var1  = P_Values_var1;
Results.Mask_var1      = Mask_var1;

Results.Obs_var2       = t_Obs_var2;
Results.TFCE_Obs_var2  = TFCE_Obs_var2;
Results.maxTFCE_var2   = maxTFCE_var2;
Results.P_Values_var2  = P_Values_var2;
Results.Mask_var2      = Mask_var2;

% Step 5: Plot ignificant observed effects for var1 and var2
plot_tfce_results(Results.Obs_var1, Results.Mask_var1, ...
                  times, e_loc, 'Significant Observed Effects: var1');

plot_tfce_results(Results.Obs_var2, Results.Mask_var2, ...
                  times, e_loc, 'Significant Observed Effects: var2');
              
                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local functions must be at the end of the script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_tfce_results(Obs, Mask, times, e_loc, plot_title)

    mT = Obs;
    mT(~Mask) = 0;

    figure;
    imagesc(times, 1:size(mT,1), mT);
    axis xy;

    xlim([-200 800]);

    set(gca, ...
        'YTick', 1:size(mT,1), ...
        'YTickLabel', {e_loc.labels}, ...
        'XTick', -200:200:800, ...
        'TickLength', [0 0], ...
        'FontSize', 15, ...
        'FontName', 'Arial');

    xlabel('Time (ms)');
    ylabel('Channel');
    title(plot_title);

    colorbar;

end
