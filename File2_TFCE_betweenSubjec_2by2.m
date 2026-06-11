
clear all; clc; close all;
load('data/01_simulated_between_subject_EEG.mat')
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

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan

    for tpoint = 1:nTime

        EEG_local = double(squeeze(data(:, ch, tpoint)));
        tic;
        lm_local = fitlm(group, EEG_local);
        toc;
        t_Obs(ch,tpoint) = lm_local.Coefficients.Estimate(2);

    end

end

% Step-2
ChN = ept_ChN2(e_loc); E_H = [0.66, 2];
TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

% Step-3
nperms=999;
num_rows = size(group,1);

TFCE_permMax = nan(nperms,1);
perm_t = nan(size(data,2),size(data,3));
parfor p = 1:nperms
    XX = group(randperm(num_rows),:);
    perm_t_local = nan(size(data,2),size(data,3));
    for ch = 1:size(data,2)
        for tpoint = 1:size(data,3)
            EEG_local = double(squeeze(data(:, ch, tpoint)));

            tic;
            lm_local = fitlm(XX, EEG_local);
            toc;
            perm_t_local(ch,tpoint) = lm_local.Coefficients.Estimate(2);%stats.tstat; 
        end
    end
    ChN = ept_ChN2(e_loc); E_H = [0.66, 2];
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
for idx = 1:size(TFCE_Obs,1)
    for jdx = 1:size(TFCE_Obs,2)
        P_Values(idx,jdx) = sum(abs(TFCE_Obs(idx,jdx))<=maxTFCE)/(nPerm+1);
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

% mT = Results.Obs;
% mT(not(Results.Mask))=0;
% tick_labels = reshape({e_loc.labels}, 32, 1);
% 
% 
% figure,
% imagesc(mT)
% xlim([-200 800])
% set(gca,'ytick',1:32,'FontSize',15,'FontName','Arial');
% set(gca,'TickLength',[0 0]);
% set(gca,'XTick',linspace(-200, 800, 4),'XTickLabel',-200:100:700,'FontSize',15,'FontName','Arial');
% yticklabels(tick_labels);
% hc=colorbar;