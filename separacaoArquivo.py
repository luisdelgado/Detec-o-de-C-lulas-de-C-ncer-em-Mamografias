import numpy as np
 
CLASS_INDEX = 6
TRAINNING = 0.5
VALIDATION = TEST = 0.25
 
a = np.loadtxt('mammography-consolidatedSemDuplicada.csv', delimiter=',') 
class0 = np.array([row for row in a if row[CLASS_INDEX] == 0])
class1 = np.array([row for row in a if row[CLASS_INDEX] == 1])

np.random.shuffle(class0)
np.random.shuffle(class1)
 
c0_rows = class0.shape[0]
print("classe 0")
print(c0_rows)
c1_rows = class1.shape[0]
print("classe 1")
print(c1_rows)
 
trainning0 = round(c0_rows*TRAINNING)
validation0 = round(c0_rows*VALIDATION)
test0 = round(c0_rows*TEST)
 
c0_trainning = class0[:trainning0]
c0_validation = class0[trainning0:trainning0+validation0]
c0_test = class0[trainning0+test0:]
 
trainning1 = round(c1_rows*TRAINNING)
validation1 = round(c1_rows*VALIDATION)
print(">>>>>>>>>>> ")
print(validation1)
test1 = round(c1_rows*TEST)
 
c1_trainning = class1[:trainning1]
c1_validation = class1[trainning1:trainning1+validation1]
c1_test = class1[trainning1+test1:]



#falta comprar o tamanho dos arquivos

c1_trainning_size = c1_trainning.shape[0]
print("tamanho de treinamento 1 eh ")
print(c1_trainning_size)
c0_trainning_size = c0_trainning.shape[0]
print("tamanho de treinamento 0 eh ")
print(c0_trainning_size)
trainning_diff = c0_trainning_size - c1_trainning_size;
print(" treinamento diferenca eh ")
print(trainning_diff)

c1_validation_size = c1_validation.shape[0]
print("tamanho de validation 1 eh ")
print(c1_validation_size)
c0_validation_size = c0_validation.shape[0]
print("tamanho de validation 0 eh ")
print(c0_validation_size)
validation_diff = c0_validation_size - c1_validation_size;
print("validation diferenca eh ")
print(validation_diff)



c1_test_size = c1_test.shape[0]
print("tamanho de teste 1 eh ")
print(c1_test_size)
c0_test_size = c0_test.shape[0]
print("tamanho de teste 0 eh ")
print(c0_test_size)
test_diff = c0_test_size - c1_test_size;
print(" teste diferenca eh ")
print(test_diff)


print("------------------------------")
#arrumando a classe de menor tamanho
#treinamento
aux = round(trainning_diff/c1_trainning_size)
aux = aux-1
c1_trainning_aux =  np.copy(c1_trainning)
for x in range(aux):
    print(c1_trainning_aux.shape[0])
    c1_trainning_aux = np.concatenate((c1_trainning_aux,c1_trainning),axis=0)
    #print(x,c1_trainning_aux.shape)


c1_trainning_size = c1_trainning_aux.shape[0]
print("tamanho de Treinamento 1 eh ")
print(c1_trainning_size)
c0_trainning_size = c0_trainning.shape[0]
print("tamanho de treinamento 0 eh ")
print(c0_trainning_size)
trainning_diff = c0_trainning_size - c1_trainning_size;
print(" treinamento diferenca eh ")
print(trainning_diff)
if(trainning_diff>0):
    c1_trainning_aux2 = c1_trainning_aux[:trainning_diff]
    c1_trainning = np.concatenate((c1_trainning_aux2,c1_trainning_aux),axis=0)
    c1_trainning_size = c1_trainning.shape[0]
    print("Por fim tamanho de Treinamento 1 eh ")
    print(c1_trainning_size)
else:
    c1_trainning = c1_trainning_aux
    c1_trainning_size = c1_trainning.shape[0]
    print("Por fim tamanho de Treinamento 1 eh ")
    print(c1_trainning_size)
    
print("--------------------------")
#validacao

aux = round(validation_diff/c1_validation_size)
aux = aux-1
c1_validation_aux =  np.copy(c1_validation)
for x in range(aux):
    c1_validation_aux = np.concatenate((c1_validation_aux,c1_validation),axis=0)


c1_validation_size = c1_validation_aux.shape[0]
print("tamanho de validacao 1 eh ")
print(c1_validation_size)
c0_validation_size = c0_validation.shape[0]
print("tamanho de validacao 0 eh ")
print(c0_validation_size)
validation_diff = c0_validation_size - c1_validation_size;
print(" validacao diferenca eh ")
print(validation_diff)
if(validation_diff>0):
    c1_validation_aux2 = c1_validation_aux[:validation_diff]
    c1_validation = np.concatenate((c1_validation_aux2,c1_validation_aux),axis=0)
    c1_validation_size = c1_validation.shape[0]
    print("Por fim tamanho de Validacao 1 eh ")
    print(c1_validation_size)
else:
    c1_validation = c1_validation_aux
    c1_validation_size = c1_validation.shape[0]
    print("Por fim tamanho de Validacao 1 eh ")
    print(c1_validation_size)


#agora da  um shuffle
np.random.shuffle(c1_trainning)
np.random.shuffle(c1_validation)
np.random.shuffle(c1_test)

np.savetxt("treinamento_Classe1.csv", c1_trainning, delimiter=",",fmt='%.8f')
np.savetxt("validacao_Classe1.csv", c1_validation, delimiter=",",fmt='%.8f')
np.savetxt("teste_Classe1.csv", c1_test, delimiter=",",fmt='%.8f')
np.savetxt("treinamento_Classe0.csv", c0_trainning, delimiter=",;",fmt='%.8f')
np.savetxt("validacao_Classe0.csv", c0_validation, delimiter=",",fmt='%.8f')
np.savetxt("teste_Classe0.csv", c0_test, delimiter=",",fmt='%.8f')
#aux = np.intersect1d(c1_validation,c1_test)
#print(aux)

