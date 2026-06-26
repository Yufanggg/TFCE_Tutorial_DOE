
clear all; clc; close all;
load('data/03_simulated_between_subject_2by2Int_EEG.mat')
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


t_Obs_Int = zeros(nChan, nTime);

for ch = 1:nChan

    for tpoint = 1:nTime

        EEG_local = double(data(:, ch, tpoint));
        tic;
        lm_local = fitlm(X, EEG_local(:));
        t_Obs_Int(ch,tpoint) = lm_local.Coefficients.tStat(4);
        toc;
    end

end

% Step-2
ChN = ept_ChN2(e_loc); E_H = [0.66, 2];

TFCE_Obs_Int = ept_mex_TFCE2D(t_Obs_Int, ChN, E_H);


% Step-3
nPerm=1;
num_rows = size(group,1);

TFCE_permMax_Int = nan(nPerm,1);

% Design matrices (only focus on the interaction effects)
X_full = X;
X_red = [X(:, 1), X(:, 2)];

parfor p = 1:nPerm
    
    perm_t_local = nan(nChan, nTime);
    
    for ch = 1:nChan
        for tpoint = 1:nTime
            
            % EEG data at this channel and time point 
            Y = double(data(:, ch, tpoint));
            
            % Reduced model 
            beta_red = X_red \ Y; 
            Y_hat_red = X_red * beta_red; 
            resid_red = Y - Y_hat_red;
            
            % Freedman-Lane permutation 
            perm_idx = randperm(num_rows); 
            Y_perm = Y_hat_red + resid_red(perm_idx);
            
            % Full model fitted to permuted data 
            lm_local = fitlm(X_full, Y_perm);
            perm_t_local(ch,tpoint) = lm_local.Coefficients.tStat(4);
        end
    end

   
    TFCE_perm = ept_mex_TFCE2D(perm_t_local, ChN, E_H);
    TFCE_permMax_Int(p) = max(abs(TFCE_perm(:)));
    
end 

% Step-4
Alpha = .05;
maxTFCE_Int = sort([TFCE_permMax_Int;max(abs(TFCE_Obs_Int(:)))]);
maxTFCEcrit_Int = maxTFCE_Int(round(nPerm*(1-Alpha)));
Mask_Int = abs(TFCE_Obs_Int)>=maxTFCEcrit_Int;
P_Values_Int = NaN(size(TFCE_Obs_Int,1),size(TFCE_Obs_Int,2));
for ch = 1:nChan
    for tpoint = 1:nTime
        P_Values_Int(ch,tpoint) = sum(abs(TFCE_Obs_Int(ch,tpoint))<=maxTFCE_Int)/(nPerm+1);
    end
end

Results.Obs_Int                 = t_Obs_Int;
Results.TFCE_Obs_Int            = TFCE_Obs_Int;
Results.maxTFCE_Int             = maxTFCE_Int;
Results.P_Values_Int            = P_Values_Int;
Results.Mask_Int               = Mask_Int;

% Step-5
% Mask non-significant values
mT = Results.Obs_Int;
mT(~Results.Mask_Int) = 0;

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
