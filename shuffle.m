numTr         = 7596;   % Numero de padroes de treinamento
numVal        = 3798;    % Numero de padroes de validacao
numTeste      = 1961;    % Numero de padroes de teste

arquivoTreinamento = fopen('treinamento_original.txt','r');
dadosTreinamento = fscanf(arquivoTreinamento,'%f',[7, numTr])';   % Lendo arquivo de treinamento
dadosTreinamento = dadosTreinamento(randperm(size(dadosTreinamento,1)),:);
dlmwrite('treinamento.txt', dadosTreinamento, 'delimiter', '\t', 'precision','%.8f');

arquivoValidacao = fopen('validacao_original.txt','r');
dadosValidacao = fscanf(arquivoValidacao,'%f',[7, numVal])';    % Mesmo processo para validacao
dadosValidacao = dadosValidacao(randperm(size(dadosValidacao,1)),:);
dlmwrite('validacao.txt', dadosValidacao, 'delimiter', '\t', 'precision','%.8f');

arquivoTeste = fopen('teste_original.txt','r');
dadosTeste = fscanf(arquivoTeste,'%f',[7, numTeste])';      % Mesmo processo para teste
dadosTeste = dadosTeste(randperm(size(dadosTeste,1)),:);
dlmwrite('teste.txt', dadosTeste, 'delimiter', '\t', 'precision','%.8f');
