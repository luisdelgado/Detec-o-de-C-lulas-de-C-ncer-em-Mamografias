echo on
clear

%   Variaveis da automatização
times = 5;
neurons = [2, 6, 14];
learn_rate = [0.4, 0.1, 0.001];
activation = ['logsig','tansig'];
trainning = ['trainoss','traingdm','trainlm'];

configs = {};
config_count = 1;

for neuron = 1:length(neurons)
    for lr = 1:length(learn_rate)
        % Primeiramente, é criado o objeto de configuração, contendo
        % todos os argumentos necessários pra rodar a rede
        config = containers.Map( ...
            {'neurons', 'activation', 'trainning', 'learning_rate', 'times'}, ...
            {neurons(neuron), 'logsig', 'traingdm', learn_rate(lr), times});

        config_out = run_nn(config, false)
        configs{config_count} = config_out;
        config_count = config_count + 1;
    end
end

configs_top = get_top(configs, 3);

for c = 1:length(configs_top)
    for af = 1:length(activation)
        for tf = 1:length(trainning)
            top_config = configs_top{c};
            activation_func = activation(af);
            trainning_func = trainning(tf);
            
            config = containers.Map( ...
                {'neurons', 'activation', 'trainning', 'learning_rate', 'times'}, ...
                {top_config('neurons'), activation_func, trainning_func, ...
                top_config('learning_rate'), 1});
            
            config_out2 = run_nn(config, true);
        end
    end
end
   
            
            
            