
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

t_Obs_var1 = zeros(nChan, nTime);
t_Obs_var2 = zeros(nChan, nTime);
t_Obs_Int = zeros(nChan, nTime);

for ch = 1:nChan

    for tpoint = 1:nTime

        EEG_local = double(data(:, ch, tpoint));
        tic;
        lm_local = fitlm(X, EEG_local(:));
        t_Obs_var1(ch,tpoint) = lm_local.Coefficients.tStat(2);
        t_Obs_var2(ch,tpoint) = lm_local.Coefficients.tStat(3);
        t_Obs_Int(ch,tpoint) = lm_local.Coefficients.tStat(4);
        toc;
    end

end

% Step-2
ChN = ept_ChN2(e_loc); E_H = [0.66, 2];
TFCE_Obs_var1 = ept_mex_TFCE2D(t_Obs_var1, ChN, E_H);
TFCE_Obs_var2 = ept_mex_TFCE2D(t_Obs_var2, ChN, E_H);
TFCE_Obs_Int = ept_mex_TFCE2D(t_Obs_Int, ChN, E_H);


% Step-3
nperms=999;
num_rows = size(var1,1);

TFCE_permMax_var1 = nan(nperms,1);
TFCE_permMax_var2 = nan(nperms,1);
TFCE_permMax_Int = nan(nperms,1);

parfor p = 1:nperms
    XX = group(randperm(num_rows),:);
    perm_t_local_var1 = nan(size(data,2),size(data,3));
    perm_t_local_var2 = nan(size(data,2),size(data,3));
    
    for ch = 1:nChan
        for tpoint = 1:nTime
            EEG_local = double(data(:, ch, tpoint));
            tic;
            % Test var1, controlling for var2
            Xred1 = [ones(N,1), var2];
            b_red1 = Xred1 \ y;
            yhat_red1 = Xred1*b_red1;
            res_red1 = y - yhat_red1;

            lm_local = fitlm(XX, EEG_local);
            perm_t_local_var1(ch,tpoint) = lm_local.Coefficients.Estimate(2);
            code 
            % Test var2, controlling for var1
            Xred2 = [ones(N,1), var1];
            b_red2 = Xred2 \ y;
            yhat_red2 = Xred2*b_red2;
            res_red2 = y - yhat_red2;
            
            %Test the interaction
            XredInt = [ones(N,1), var1, var2];
            
            toc;
            
        end
    end
    TFCE_perm_var1 = ept_mex_TFCE2D(perm_t_local_var1, ChN, E_H);
    TFCE_permMax_var1(p) = max(abs(TFCE_perm_var1(:)));
    
    TFCE_perm_var2 = ept_mex_TFCE2D(perm_t_local_var2, ChN, E_H);
    TFCE_permMax_var2(p) = max(abs(TFCE_perm_var2(:)));
end

% Step-4
Alpha = .05;
nPerm = length(TFCE_permMax_var1);
maxTFCE_var1 = sort([TFCE_permMax_var1;max(abs(TFCE_Obs_var1(:)))]);
maxTFCEcrit_var1 = maxTFCE_var1(round(nPerm*(1-Alpha)));
Mask_var1 = abs(TFCE_Obs_var1)>=maxTFCEcrit_var1;
P_Values_var1 = NaN(size(TFCE_Obs_var1,1),size(TFCE_Obs_var1,2));
for idx = 1:size(TFCE_Obs_var1,1)
    for jdx = 1:size(TFCE_Obs_var1,2)
        P_Values_var1(idx,jdx) = sum(abs(TFCE_Obs_var1(idx,jdx))<=maxTFCE_var1)/(nPerm+1);
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
