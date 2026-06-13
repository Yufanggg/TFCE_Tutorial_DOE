
clear all; clc; close all;
load('data/04_simulated_between_subject_covariate_EEG.mat')
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
X = [group, covariate];
t_Obs = zeros(nChan, nTime);

for ch = 1:nChan

    for tpoint = 1:nTime
        
        EEG_local = double(data(:, ch, tpoint));  
        tic;
        lm_local = fitlm(X, EEG_local(:));
        toc;
        % t-statistic for Group effect
        t_Obs(ch, tpoint) = lm_local.Coefficients.tStat(2);

    end

end

% Step-2
ChN = ept_ChN2(e_loc); E_H = [0.66, 2];
TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

% Step-3
nperms=999;
num_rows = size(group,1);
TFCE_permMax = nan(nperms,1);
ChN = ept_ChN2(e_loc); E_H = [0.66, 2];
perm_t = nan(size(data,2),size(data,3));

% Design matrices 
X_full = [group, covariate]; 
X_red = covariate; 

parfor p = 1:nperms
    
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
            tic;
            % Full model fitted to permuted data 
            lm_local = fitlm(X_full, Y_perm)
            perm_t_local(ch,tpoint) = lm_local.Coefficients.tStat(2);
            toc;
        end
    end
   
    TFCE_perm = ept_mex_TFCE2D(perm_t_local, ChN, E_H);
    TFCE_permMax(p) = max(abs(TFCE_perm(:)));
end

% Step-4
Alpha = .05;
nPerm = length(TFCE_permMax);
maxTFCE = sort([TFCE_permMax;max(abs(TFCE_Obs(:)))]);
maxTFCEcrit = maxTFCE(round(nPerm*(1-Alpha)));
Mask = abs(TFCE_Obs)>=maxTFCEcrit;
P_Values = NaN(size(TFCE_Obs,1),size(TFCE_Obs,2));
for ch = 1:nChan
    for tpoint = 1:nTime
        P_Values(ch,tpoint) = sum(abs(TFCE_Obs(ch,tpoint))<=maxTFCE)/(nPerm+1);
    end
end
Results.Obs                 = t_Obs;
Results.TFCE_Obs            = TFCE_Obs;
Results.maxTFCE             = maxTFCE;
Results.P_Values            = P_Values;
Results.Mask                = Mask;

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