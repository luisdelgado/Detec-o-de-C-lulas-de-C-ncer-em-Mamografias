function [config_out] = run_nn(config, print_flag)
%    Informacoes sobre a rede e os dados
numEntradas   = 6;     % Numero de nodos de entrada
numSaidas     = 2;     % Numero de nodos de saida
numTr         = 7596;   % Numero de padroes de treinamento
numVal        = 3798;    % Numero de padroes de validacao
numTeste      = 1961;    % Numero de padroes de teste

echo off

%    Abrindo arquivos 
arquivoTreinamento = fopen('treinamento.txt','r');  
arquivoValidacao   = fopen('validacao.txt','r');    
arquivoTeste       = fopen('teste.txt','r');        

%    Lendo arquivos e armazenando dados em matrizes
dadosTreinamento    = fscanf(arquivoTreinamento,'%f',[(numEntradas + 1), numTr]);   % Lendo arquivo de treinamento
entradasTreinamento = dadosTreinamento(1:numEntradas, 1:numTr);
saidasTreinamento   = dadosTreinamento((numEntradas + 1):(numEntradas + 1), 1:numTr);

novaSaidasTr        = abs(saidasTreinamento - 1);
saidasTreinamento   = [novaSaidasTr; saidasTreinamento];

dadosValidacao      = fscanf(arquivoValidacao,'%f',[(numEntradas + 1), numVal]);    % Mesmo processo para validacao
entradasValidacao   = dadosValidacao(1:numEntradas, 1:numVal);
saidasValidacao     = dadosValidacao((numEntradas + 1):(numEntradas + 1), 1:numVal);

novaSaidasVal       = abs(saidasValidacao - 1);
saidasValidacao     = [novaSaidasVal; saidasValidacao];

dadosTeste          = fscanf(arquivoTeste,'%f',[(numEntradas + 1), numTeste]);      % Mesmo processo para teste
entradasTeste       = dadosTeste(1:numEntradas, 1:numTeste);
saidasTeste         = dadosTeste((numEntradas + 1):(numEntradas + 1), 1:numTeste);

novaSaidasTest      = abs(saidasTeste - 1);
saidasTeste         = [novaSaidasTest; saidasTeste];

%    Fechando arquivos
fclose(arquivoTreinamento);
fclose(arquivoValidacao);
fclose(arquivoTeste);

numEscondidos = config('neurons');     % Numero de nodos escondidos
activation = config('activation');
trainning = config('trainning');
learning_rate = config('learning_rate');
times = config('times');

results_auc = zeros(1, times); % Cria um array de zeros para savar os resultados (AUC)

for iter = 1:times
    %   Criando a rede (para ajuda, digite 'help newff')

    for entrada = 1 : numEntradas;  % Cria 'matrizFaixa', que possui 'numEntradas' linhas, cada uma sendo igual a [0 1].
         matrizFaixa(entrada,:) = [0 1];  
    end

    rede = newff(matrizFaixa,[numEscondidos numSaidas],{activation, activation}, ...
        trainning, 'learngdm','mse');
    % matrizFaixa                    : indica que todas as entradas possuem valores na faixa entre 0 e 1
    % [numEscondidos numSaidas]      : indica a quantidade de nodos escondidos e de saida da rede
    % {'logsig','logsig'}            : indica que os nodos das camadas escondida e de saida terao funcao de ativacao sigmoide logistica
    % 'traingdm','learngdm'          : indica que o treinamento vai ser feito com gradiente descendente (backpropagation)
    % 'sse'                          : indica que o erro a ser utilizado vai ser SSE (soma dos erros quadraticos)

    % Inicializa os pesos da rede criada (para ajuda, digite 'help init')
    rede = init(rede);
    echo on
    %   Parametros do treinamento (para ajuda, digite 'help traingd')
    rede.trainParam.epochs   = 10000;    % Maximo numero de iteracoes
    rede.trainParam.lr       = learning_rate;
    rede.trainParam.goal     = 0;      % Criterio de minimo erro de treinamento
    rede.trainParam.max_fail = 20;      % Criterio de quantidade maxima de falhas na validacao
    rede.trainParam.min_grad = 0;      % Criterio de gradiente minimo
    rede.trainParam.show     = 10;     % Iteracoes entre exibicoes na tela (preenchendo com 'NaN', nao exibe na tela)
    rede.trainParam.time     = inf;    % Tempo maximo (em segundos) para o treinamento
    echo off
    fprintf('\nTreinando ...\n')

    conjuntoValidacao.P = entradasValidacao; % Entradas da validacao
    conjuntoValidacao.T = saidasValidacao;   % Saidas desejadas da validacao

    %   Treinando a rede
    [redeNova,desempenho,saidasRede,erros] = train(rede,entradasTreinamento,saidasTreinamento,[],[],conjuntoValidacao);
    % redeNova   : rede apos treinamento
    % desempenho : apresenta os seguintes resultados
    %              desempenho.perf  - vetor com os erros de treinamento de todas as iteracoes (neste exemplo, escolheu-se erro SSE)
    %              desempenho.vperf - vetor com os erros de validacao de todas as iteracoes (idem)
    %              desempenho.epoch - vetor com as iteracoes efetuadas
    % saidasRede : matriz contendo as saidas da rede para cada padrao de treinamento
    % erros      : matriz contendo os erros para cada padrao de treinamento
    %             (para cada padrao: erro = saida desejada - saida da rede)
    % Obs.       : Os dois argumentos de 'train' preenchidos com [] apenas sao utilizados quando se usam delays
    %             (para ajuda, digitar 'help train')

    fprintf('\nTestando ...\n');

    %    Testando a rede
    [saidasRedeTeste,Pf,Af,errosTeste,desempenhoTeste] = privatesim(redeNova,entradasTeste,[],[],saidasTeste);
    % saidasRedeTeste : matriz contendo as saidas da rede para cada padrao de teste
    % Pf,Af           : matrizes nao usadas neste exemplo (apenas quando se usam delays)
    % errosTeste      : matriz contendo os erros para cada padrao de teste
    %                  (para cada padrao: erro = saida desejada - saida da rede)
    % desempenhoTeste : erro de teste (neste exemplo, escolheu-se erro SSE)

    fprintf('MSE para o conjunto de treinamento: %6.5f \n',desempenho.perf(length(desempenho.perf)));
    fprintf('MSE para o conjunto de validacao: %6.5f \n',desempenho.vperf(length(desempenho.vperf)));
    fprintf('MSE para o conjunto de teste: %6.5f \n',desempenhoTeste);

    %     Calculando o erro de classificacao para o conjunto de teste
    %     (A regra de classificacao e' winner-takes-all, ou seja, o nodo de saida que gerar o maior valor de saida
    %      corresponde a classe do padrao).
    %     Obs.: Esse erro so' faz sentido se o problema for de classificacao. Para problemas que nao sao de classificacao,
    %           esse trecho do script deve ser eliminado.

    [maiorSaidaRede, nodoVencedorRede] = max (saidasRedeTeste);
    [maiorSaidaDesejada, nodoVencedorDesejado] = max (saidasTeste);

    %      Obs.: O comando 'max' aplicado a uma matriz gera dois vetores: um contendo os maiores elementos de cada coluna
    %            e outro contendo as linhas onde ocorreram os maiores elementos de cada coluna.

    classificacoesErradas = 0;

    for padrao = 1 : numTeste;
        if nodoVencedorRede(padrao) ~= nodoVencedorDesejado(padrao),
            classificacoesErradas = classificacoesErradas + 1;
        end
    end

    erroClassifTeste = 100 * (classificacoesErradas/numTeste);

    fprintf('Erro de classificacao para o conjunto de teste: %6.5f%% \n',erroClassifTeste);
    if(print_flag == true)
        filename = ['plots\confusion_' config('neurons') '_' config('learn_rate') '_' ...
            config('activation') '_' config('trainning')];
        figure; plotconfusion(saidasTeste,saidasRedeTeste); % Matriz de confusao
        print('matriz', '-dpng')
    end
    
    %    Curva ROC com valores padrões
    [x, y, t, auc] = perfcurve(saidasTeste(2, :), saidasRedeTeste(2, :), 1);
    results_auc(iter) = auc;
    fprintf('Area under ROC curve %6.5f \n',auc);
    fprintf('===== Config =====\n');
    fprintf('Neuronios intermed.: %d \n',config('neurons'));
    fprintf('Taxa de aprendizagem: %6.5f \n', config('learning_rate'));
    fprintf('Função de ativação: %s \n', config('activation'));
    fprintf('Função de treinamento %s \n', config('trainning'));
    fprintf('===== End of Config =====\n');
    
    if(print_flag == true)
        filename = ['plots\roc_' config('neurons') '_' config('learning_rate') '_' ...
            config('activation') '_' config('trainning')];
        figure; plot (x,y); title('Roc Curve'); xlabel('False Positive'); ylabel('True Positive');
        print(filename,'-dpng');
    end
end

config_out = config;
config_out('auc_mean') = mean(results_auc);
end

