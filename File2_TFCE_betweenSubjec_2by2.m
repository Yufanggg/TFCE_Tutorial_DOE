
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

e_loc = chanlocs_1020(idx);
%% Step-1
[nSubj, nChan, nTime] = size(data);

t_Obs_var1 = zeros(nChan, nTime);
t_Obs_var2 = zeros(nChan, nTime);

for ch = 1:nChan

    for tpoint = 1:nTime

        EEG_local = double(data(:, ch, tpoint));
        tic;
        X = [var1(:) var2(:)];
        lm_local = fitlm(X, EEG_local(:));
        toc;
        t_Obs_var1(ch,tpoint) = lm_local.Coefficients.tStat(2);
        t_Obs_var2(ch,tpoint) = lm_local.Coefficients.tStat(3);

    end

end

% Step-2
ChN = ept_ChN2(e_loc); E_H = [0.66, 2];
TFCE_Obs_var1 = ept_mex_TFCE2D(t_Obs_var1, ChN, E_H);
TFCE_Obs_var2 = ept_mex_TFCE2D(t_Obs_var2, ChN, E_H);

% Step-3: Synchronized permutation test
nperms=999;
num_rows = size(var1,1);

TFCE_permMax_var1 = nan(nperms,1);
TFCE_permMax_var2 = nan(nperms,1);

idx_Bminus = find(var2 == -1);
idx_Bplus  = find(var2 ==  1);

idx_Aminus = find(var1 == -1);
idx_Aplus  = find(var1 ==  1);

parfor p = 1:nperms
    %Permute Factor A within each level of Factor B
    var1_perm = var1(:);

    var1_perm(idx_Bminus) = var1_perm(idx_Bminus(randperm(length(idx_Bminus))));
    var1_perm(idx_Bplus)  = var1_perm(idx_Bplus(randperm(length(idx_Bplus))));

    X_perm_A = [var1_perm(:), var2(:)];
    
    perm_t_local_var1 = nan(size(data,2),size(data,3));
    perm_t_local_var2 = nan(size(data,2),size(data,3));
    
    % Permute Factor B within each level of Factor A
    var2_perm = var2(:);

    var2_perm(idx_Aminus) = var2_perm(idx_Aminus(randperm(length(idx_Aminus))));
    var2_perm(idx_Aplus)  = var2_perm(idx_Aplus(randperm(length(idx_Aplus))));

    X_perm_B = [var1(:), var2_perm(:)];
    
    % Fit models
    perm_t_var1 = zeros(nChan,nTime);
    perm_t_var2 = zeros(nChan,nTime);
    
    for ch = 1:nChan
        for tpoint = 1:nTime
            EEG_local = double(data(:, ch, tpoint));
            tic;
            lm_A = fitlm(X_perm_A, EEG_local(:));
            lm_B = fitlm(X_perm_B, EEG_local(:));

            perm_t_var1(ch,tpoint) = lm_A.Coefficients.tStat(2);
            perm_t_var2(ch,tpoint) = lm_B.Coefficients.tStat(3);
            
            toc;
            
        end
    end
    TFCE_perm_var1 = ept_mex_TFCE2D(perm_t_var1, ChN, E_H);
    TFCE_perm_var2 = ept_mex_TFCE2D(perm_t_var2, ChN, E_H);

    TFCE_permMax_var1(p) = max(abs(TFCE_perm_var1(:)));
    TFCE_permMax_var2(p) = max(abs(TFCE_perm_var2(:)));
end

% Step-4: TFCE-corrected p-values and masks
Alpha = .05;
nPerm = length(TFCE_permMax_var1);

P_Values_var1 = nan(size(TFCE_Obs_var1));
P_Values_var2 = nan(size(TFCE_Obs_var2));

maxTFCE_var1 = sort([TFCE_permMax_var1;max(abs(TFCE_Obs_var1(:)))]);
maxTFCEcrit_var1 = maxTFCE_var1(round(nPerm*(1-Alpha)));
Mask_var1 = abs(TFCE_Obs_var1)>=maxTFCEcrit_var1;
P_Values_var1 = NaN(size(TFCE_Obs_var1,1),size(TFCE_Obs_var1,2));
for ch = 1:nChan
    for tpoint = 1:size(nTime,2)
        P_Values_var1(ch,tpoint) = sum(abs(TFCE_Obs_var1(ch,tpoint))<=maxTFCE_var1)/(nPerm+1);
    end
end

Results.Obs_var1                 = t_Obs_var1;
Results.TFCE_Obs_var1            = TFCE_Obs_var1;
Results.maxTFCE_var1             = maxTFCE_var1;
Results.P_Values_var1            = P_Values_var1;
Results.Mask_var1               = Mask_var1;

% Step-5
% Mask non-significant values
mT = Results.Obs;
mT(~Results.Mask) = 0;

% Channel labels
tick_labels = {e_loc.labels};

% Plot significant effects
figure;
imagesc(times, 1:32, mT);
axis xy;

% Axes formatting
xlim([-200 800]);
set(gca, ...
    'YTick', 1:32, ...
    'YTickLabel', tick_labels, ...
    'XTick', -200:200:800, ...
    'TickLength', [0 0], ...
    'FontSize', 15, ...
    'FontName', 'Arial');

% Labels and title
xlabel('Time (ms)');
ylabel('Channel');
title('Significant Observed Effects');

% Colorbar
hc = colorbar;
