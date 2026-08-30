% function [rperms] = lmeEEG_permutations2(nperms,SS,Item)
% % lmeEEG_permutations: Compute unique permutations within subjects and items
% % This function is for fully-crossed designs  
% % [Inputs]
% % - SS: Nominal variable of subjects' indeces
% % - Item: Nominal variable of items' indeces
% % - nperms: number of permutations
% % [Output]
% % -rperms: matrix with unique permutations
% disp('Compute permutation indeces')
% subj =unique(SS);
% it = unique(Item);
% rperms = nan(length(SS),nperms);
% 
% for id = 1:length(subj)
%     fprintf("for the %d subj\n", id)
%     for itx = 1:length(it)
%         fprintf("for the %d item\n", itx)
%         idx = find(SS==subj(id)&Item==it(itx));
%         idx2 = repmat(idx,1,nperms);
%         for i =1:nperms
%             goon = true;
%             while goon
%                 tmp = idx2(randperm(size(idx2,1)),i);
%                 if ~any(all(repmat(tmp,1,size(idx2,2)) == idx2, 1))%isempty(find(sum(tmp==idx2)==size(idx2,1)))
%                     idx2(:,i)=tmp;
%                     goon = false;
%                 end
%             end
%         end
%         %progressbar(id/length(subj));
%         rperms(idx,:)=idx2;
% 
%     end
% end



function rperms = lmeEEG_permutations2(nperms, SS, Item)

subj = unique(SS);
it   = unique(Item);

nObs = length(SS);
rperms = nan(nObs, nperms);

for id = 1:length(subj)

    for itx = 1:length(it)

        idx = find(SS == subj(id) & Item == it(itx));

        nIdx = length(idx);

        if nIdx == 0
            continue
        elseif nIdx == 1
            rperms(idx,:) = repmat(idx, 1, nperms);
        else
            [~, permOrder] = sort(rand(nIdx, nperms), 1);
            rperms(idx,:) = idx(permOrder);
        end

    end
end

if any(isnan(rperms(:)))
    error('Some observations were not assigned a permutation index.');
end

end
