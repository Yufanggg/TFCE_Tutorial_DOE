%% ==========================================================
% TFCE analysis
% Nested within-student design
%
% Input file contains only:
%   EEGdata
%   designTable
%
% EEGdata:
%   Student-condition rows x Channels x Time
%   120 x 32 x 251
%
% designTable:
%   Class
%   Student
%   CondCode    % -1 = Control, 1 = Treatment
%   CondName
%
% Model:
%   EEG ~ CondCode + (1|Class) + (1|Class:Student)
%
% Test:
%   Treatment - Control fixed effect
%
% Permutation:
%   Flip condition labels within each student nested in class
%% ==========================================================

clear; clc; close all;
fprintf('\nStarting nested TFCE analysis...\n');

%% ==========================================================
% Load data
%% ==========================================================

load('../data/07_simulated_nested_class_student_EEG.mat', ...
     'EEGdata', 'designTable');

times = -200:4:800;

%% ==========================================================
% Load channel locations
%% ==========================================================

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

%% ==========================================================
% Basic checks
%% ==========================================================

[nRows, nChan, nTime] = size(EEGdata);

Class    = designTable.Class;
Student  = designTable.Student;
CondCode = designTable.CondCode;

if height(designTable) ~= nRows
    error('Rows in designTable must match rows in EEGdata.');
end

if length(times) ~= nTime
    error('Length of times does not match number of time points.');
end

if length(e_loc) ~= nChan
    error('Number of channel locations does not match number of channels.');
end

Class    = Class(:);
Student  = Student(:);
CondCode = CondCode(:);

% % Unique nested student ID for checking and permutation
% nClass = unique(Class); nStudentPerClass = unique(Student)
% nestedID = sub2ind([nClass, nStudentPerClass], Class, Student);
% 
% nUnits = length(unique(nestedID));
% 
% if nRows ~= nUnits * 2
%     error('Expected exactly two condition rows per nested student.');
% end

% for u = 1:nUnits
% 
%     idxUnit = NestedID == nestedUnits(u);
% 
%     if sum(idxUnit) ~= 2
%         error('Each nested student must have exactly two rows.');
%     end
% 
%     if ~all(sort(CondCode(idxUnit)) == [-1; 1])
%         error('Each nested student must have one Control (-1) and one Treatment (1).');
%     end
% 
% end

%% ==========================================================
% Variables for LME
%% ==========================================================

ClassLME   = categorical(Class);
StudentLME = categorical(Student);

% 0 = Control, 1 = Treatment
Condition = double(CondCode == 1);

%% ==========================================================
% Step 1: Observed LME t-map
%% ==========================================================

fprintf('Step 1: Computing observed t-statistic map...\n');

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        EEG = double(squeeze(EEGdata(:, ch, tp)));

        tbl = table(EEG, Condition, ClassLME, StudentLME, ...
            'VariableNames', {'EEG','Condition','Class','Student'});

        lme = fitlme(tbl, ...
            'EEG ~ Condition + (1|Class) + (1|Class:Student)');

        % Row 2 = Treatment - Control fixed effect
        t_Obs(ch,tp) = lme.Coefficients.tStat(2);

    end
end

fprintf('Observed t-statistic map completed.\n');

%% ==========================================================
% Step 2: Observed TFCE map
%% ==========================================================

fprintf('Step 2: Computing observed TFCE map...\n');

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');

%% ==========================================================
% Step 3: Permutation by flipping condition labels within nested student
%% ==========================================================

nPerm = 999;
TFCE_permMax = nan(nPerm,1);

fprintf('Step 3: Starting within-student condition-label flipping: %d permutations...\n', nPerm);

parfor p = 1:nPerm

    perm_t = nan(nChan, nTime);

    CondCode_perm = CondCode;

    for u = 1:nUnits

        idxUnit = find(NestedID == nestedUnits(u));

        if rand > 0.5
            CondCode_perm(idxUnit) = flipud(CondCode_perm(idxUnit));
        end

    end

    Condition_perm = double(CondCode_perm == 1);

    Class_perm   = categorical(Class);
    Student_perm = categorical(Student);

    for ch = 1:nChan
        for tp = 1:nTime

            EEG = double(squeeze(EEGdata(:, ch, tp)));

            tbl = table(EEG, Condition_perm, Class_perm, Student_perm, ...
                'VariableNames', {'EEG','Condition','Class','Student'});

            lme = fitlme(tbl, ...
                'EEG ~ Condition + (1|Class) + (1|Class:Student)');

            perm_t(ch,tp) = lme.Coefficients.tStat(2);

        end
    end

    fprintf('At the %dth permutation\n', p);

    TFCE_perm = ept_mex_TFCE2D(perm_t, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

end

fprintf('\nPermutation testing completed.\n');

%% ==========================================================
% Step 4: TFCE correction
%% ==========================================================

fprintf('Step 4: Computing TFCE-corrected significance...\n');

alpha = 0.05;

critTFCE = prctile(TFCE_permMax, 100 * (1 - alpha));

Mask = abs(TFCE_Obs) >= critTFCE;

P_Values = nan(nChan, nTime);

for i = 1:numel(TFCE_Obs)

    P_Values(i) = ...
        (sum(TFCE_permMax >= abs(TFCE_Obs(i))) + 1) / ...
        (nPerm + 1);

end

fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', critTFCE);

%% ==========================================================
% Step 5: Store results
%% ==========================================================

fprintf('Step 5: Storing results...\n');

Results = struct();

Results.Obs          = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_Null    = TFCE_permMax;
Results.critTFCE     = critTFCE;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.alpha        = alpha;
Results.nPerm        = nPerm;
Results.model        = 'EEG ~ Condition + (1|Class) + (1|Class:Student)';
Results.test         = 'Treatment - Control fixed effect';
Results.permutation  = 'Condition-label flipping within Class:Student';
Results.times        = times;
Results.e_loc        = e_loc;

fprintf('Results stored.\n');

%% ==========================================================
% Step 6: Plot significant observed t-values
%% ==========================================================

mT = t_Obs;
mT(~Mask) = 0;

figure;

imagesc(times, 1:nChan, mT);
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
title('TFCE-corrected Nested LME Effect: Treatment - Control');

colorbar;

%% ==========================================================
% Step 7: Plot observed TFCE map
%% ==========================================================

figure;

imagesc(times, 1:nChan, TFCE_Obs);
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
title('Observed TFCE Map: Nested LME Treatment Effect');

colorbar;

%% ==========================================================
% Step 8: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/07_TFCE_nested_class_student_results.mat', ...
     'Results');

disp('Nested class-student TFCE analysis completed and saved.');


%% ==========================================================
% TFCE analysis for nested class-student EEG design
%
% Model idea:
%   Y ~ Condition + (1|Class) + (1|Class:Student)
%
% Exchangeability structure:
%   Class = higher-level block, kept fixed
%   Student = nested block within Class, kept fixed
%   Condition = within-student repeated factor, permuted within student
%
% Permutation:
%   Randomly swap Control/Treatment within each student
%   Equivalent to sign-flipping Treatment - Control difference waves
%% ==========================================================

clear; clc; close all;

fprintf('\nStarting nested TFCE analysis...\n');

%% Load data

load('../data/07_simulated_nested_class_student_EEG.mat');

[nSub, nCond, nChan, nTime] = size(data);

if nCond ~= 2
    error('Expected data format: Subjects x 2 Conditions x Channels x Time');
end

%% ==========================================================
% Long-format design variables for fitlme
%% ==========================================================
% For old MATLAB compatibility, use kron instead of repelem
nClass = 6;
nStudentPerClass = 10;
nSub = nClass * nStudentPerClass
nCond = 2;

% Original subject-level IDs: 60 x 1
studentID = studentID(:);
classID   = classID(:);

% Long-format IDs: 120 x 1
Student = kron(studentID, ones(nCond,1));
Class   = kron(classID, ones(nCond,1));

% Condition: 120 x 1
% -1 = Control, 1 = Treatment
Condition = repmat([-1; 1], nSub, 1);

% Convert grouping variables to categorical
Student = categorical(Student);
Class   = categorical(Class);
%% ==========================================================
% Step 1: Observed LME t-map
%% ==========================================================

fprintf('Computing observed t map...\n');

t_Obs = zeros(nChan, nTime);

for ch = 1:nChan
    for tp = 1:nTime

        controlVals   = squeeze(data(:,1,ch,tpoint));
        treatmentVals = squeeze(data(:,2,ch,tpoint));

        EEG = reshape([controlVals treatmentVals]', [], 1);
        
        tbl = table(EEG, Condition, Student, Class, ...
            'VariableNames', {'EEG','Condition','Student', 'Class'});

        lme = fitlme(tbl, 'EEG ~ Condition + (1|Student) + (1|Class:Student)');

        % Row 2 = Treatment - Control fixed effect
        t_Obs(ch,tpoint) = lme.Coefficients.tStat(2);

    end
end

fprintf('Observed t map completed.\n');

%% ==========================================================
% Step 2: Observed TFCE map
%% ==========================================================
%% TFCE settings

e_loc = chanlocs_EEG;

ChN = ept_ChN2(e_loc);
E_H = [0.66, 2];

fprintf('Computing observed TFCE map...\n');

TFCE_Obs = ept_mex_TFCE2D(t_Obs, ChN, E_H);

fprintf('Observed TFCE map completed.\n');

%% ==========================================================
% Step 3: Multi-level block permutation
%
% Higher-level block:
%   Class is fixed
%
% Nested block:
%   Student is fixed within Class
%
% Within-student repeated factor:
%   Condition labels are swapped within each student
%
% This is equivalent to sign-flipping each student's
% Treatment - Control difference wave.
%% ==========================================================
nPerms = 999;

fprintf('Starting block-restricted permutation testing: %d permutations...\n', nPerms);

TFCE_permMax = nan(nPerms, 1);

parfor p = 1:nPerms

    perm_t_local = zeros(nChan, nTime);

    %% ------------------------------------------------------
    % Permute within exchangeability blocks
    %
    % Class is fixed.
    % Student is fixed within Class.
    % Control/Treatment are randomly swapped within each student.
    %% ------------------------------------------------------

    data_perm = data;
    
    for s = 1:nSub

        if rand < 0.5

            % Swap the whole EEG image:
            % condition x channel x time
            data_perm(s,[1 2],:,:) = data_perm(s,[2 1],:,:);

        end

    end

    %% ------------------------------------------------------
    % Compute permuted t map
    %% ------------------------------------------------------

    for ch = 1:nChan
        for tp = 1:nTime
            
            controlVals   = squeeze(data_perm(:,1,ch,tpoint));
            treatmentVals = squeeze(data_perm(:,2,ch,tpoint));

            EEG = reshape([controlVals treatmentVals]', [], 1);
        
            tbl = table(EEG, Condition, studentID, classID, ...
            'VariableNames', {'EEG','Condition','Student', 'Class'});

            lme = fitlme(tbl, 'EEG ~ Condition + (1|Student) + (1|Class:Student)');

            % Row 2 = Treatment - Control fixed effect
            perm_t_local(ch,tpoint) = lme.Coefficients.tStat(2);

        end
    end

    %% ------------------------------------------------------
    % TFCE transform and max statistic
    %% ------------------------------------------------------

    TFCE_perm = ept_mex_TFCE2D(perm_t_local, ChN, E_H);

    TFCE_permMax(p) = max(abs(TFCE_perm(:)));

    fprintf('Finished permutation %d\n', p);

end

fprintf('Permutation testing completed.\n');

%% ==========================================================
% Step 4: TFCE-corrected p-values
%% ==========================================================

fprintf('Computing TFCE-corrected p-value map...\n');
alpha = 0.05;

critTFCE = prctile(TFCE_permMax, 100 * (1 - alpha));

Mask = abs(TFCE_Obs) >= critTFCE;

P_Values = nan(nChan, nTime);

for ch = 1:nChan
    for tpoint = 1:nTime
        P_Values(ch, tpoint) = (sum(TFCE_permMax >= abs(TFCE_Obs(ch, tpoint))) + 1) / ...
        (nPerm + 1);
    end
end

fprintf('TFCE-corrected significance completed.\n');
fprintf('Critical TFCE value = %.4f\n', critTFCE);

%% ==========================================================
% Step 5: Store results
%% ==========================================================
fprintf('Step 5: Storing results...\n');

Results = struct();

Results.Obs          = t_Obs;
Results.TFCE_Obs     = TFCE_Obs;
Results.TFCE_Null    = TFCE_permMax;
Results.critTFCE     = critTFCE;
Results.P_Values     = P_Values;
Results.Mask         = Mask;
Results.alpha        = alpha;
Results.nPerm        = nPerm;
Results.model        = 'EEG ~ Condition + (1|Subject)';
Results.test         = 'Treatment - Control fixed effect';

fprintf('Results stored.\n');
%% ==========================================================
% Step 6: Plot significant observed t-values
%% ==========================================================

mT = t_Obs;
mT(~Mask) = 0;

figure;

imagesc(times, 1:nChan, mT);
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
title('TFCE-corrected LME Effect: Treatment - Control');

colorbar;

%% ==========================================================
% Step 7: Plot observed TFCE map
%% ==========================================================

figure;

imagesc(times, 1:nChan, TFCE_Obs);
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
title('Observed TFCE Map: LME Treatment Effect');

colorbar;

%% ==========================================================
% Step 8: Save results
%% ==========================================================

if ~exist('../results', 'dir')
    mkdir('../results');
end

save('../results/07_TFCE_nested_LME_results.mat', ...
     'Results', ...
     't_Obs', ...
     'TFCE_Obs', ...
     'TFCE_permMax', ...
     'critTFCE', ...
     'P_Values', ...
     'Mask', ...
     'times', ...
     'e_loc');

disp('nested LME TFCE analysis completed and saved.');
