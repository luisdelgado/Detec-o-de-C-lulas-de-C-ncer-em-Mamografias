function [configs_top] = get_top(configs, n)
%GET_TOP Summary of this function goes here
%   Detailed explanation goes here

configs_ordered = {}
for i = 1:length(configs)
    configs{i}('visited') = 0;
end

for i = 1:length(configs)
    max_config = containers.Map({'auc_mean'}, {-1}) 
    max_j = -1
    for j = 1:length(configs)
        if (~configs{j}('visited') && ...
                (configs{j}('auc_mean') > max_config('auc_mean')))
            max_config = configs{j};
            max_j = j
        end
    end
    configs_ordered{i} = max_config;
    configs{max_j}('visited') = 1
    
    configs_top = configs_ordered(1:n);
end
            


end

