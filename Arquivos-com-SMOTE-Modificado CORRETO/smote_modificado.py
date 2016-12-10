import numpy as np
from copy import deepcopy
import math
import random
CLASS_INDEX = 6
TRAINNING = 0.5
VALIDATION = TEST = 0.25
 
a = np.loadtxt('mammography-consolidatedSemDuplicada.csv', delimiter=',') 
class0 = np.array([row for row in a if row[CLASS_INDEX] == 0])
class1 = np.array([row for row in a if row[CLASS_INDEX] == 1])

np.random.shuffle(class0)
np.random.shuffle(class1)
 
c0_rows = class0.shape[0]
c1_rows = class1.shape[0]

 
trainning0 = round(c0_rows*TRAINNING)
validation0 = round(c0_rows*VALIDATION)
test0 = round(c0_rows*TEST)
 
c0_trainning = class0[:trainning0]
c0_validation = class0[trainning0:trainning0+validation0]
c0_test = class0[trainning0+test0:]
 
trainning1 = round(c1_rows*TRAINNING)
validation1 = round(c1_rows*VALIDATION)
test1 = round(c1_rows*TEST)
 
c1_trainning = class1[:trainning1]
c1_validation = class1[trainning1:trainning1+validation1]
c1_test = class1[trainning1+test1:]
c1_test_size = c1_test.shape[0]


c0_trainning_size = c0_trainning.shape[0]
print("tamanho de treinamento 0 eh ",c0_trainning_size)
c0_validation_size = c0_validation.shape[0]
print("tamanho de validacao 0 eh ",c0_validation_size)

c1_trainning_size = c1_trainning.shape[0]
print("tamanho de treinamento 1 eh ",c1_trainning_size)
c1_validation_size = c1_validation.shape[0]
print("tamanho de validacao 1 eh ",c1_validation_size)


class entity:
	attributes =[]
	classification = 0
	distance = 0
	real_class = 0


training_classe0 =[];
for x in range(c0_trainning_size):
	temp_entity = entity();
	entity.classification =0;
	entity.real_class = 0;
	temp_entity.attributes = [];
	#print("linha ",c1_trainning[x]);
	linha = c0_trainning[x];
	for b in range (6) :
		temp_entity.attributes.append((linha[b]))
	temp_entity.classification = 1;
	temp_entity.distance = 0
	training_classe0.append(temp_entity)

# for i in range(len(training_classe1)):
# 	print(training_classe1[i].attributes);

validation_classe0 =[];
for x in range(c0_validation_size):
	temp_entity = entity();
	entity.classification =0;
	entity.real_class = 0;
	temp_entity.attributes = [];
	linha = c0_validation[x];
	for b in range (6) :
		temp_entity.attributes.append((linha[b]))
	temp_entity.classification = 1;
	temp_entity.distance = 0
	validation_classe0.append(temp_entity)

print(" Treinamento0   array objeto ",len(training_classe0));
print(" Validacao0 array objeto ",len(validation_classe0));
############################################################################

training_classe1 =[];
for x in range(c1_trainning_size):
	temp_entity = entity();
	entity.classification =1;
	entity.real_class = 1;
	temp_entity.attributes = [];
	#print("linha ",c1_trainning[x]);
	linha = c1_trainning[x];
	for b in range (6) :
		temp_entity.attributes.append((linha[b]))
	temp_entity.classification = 1;
	temp_entity.distance = 0
	training_classe1.append(temp_entity)

# for i in range(len(training_classe1)):
# 	print(training_classe1[i].attributes);

validation_classe1 =[];
for x in range(c1_validation_size):
	temp_entity = entity();
	entity.classification =1;
	entity.real_class = 1;
	temp_entity.attributes = [];
	linha = c1_validation[x];
	for b in range (6) :
		temp_entity.attributes.append((linha[b]))
	temp_entity.classification = 1;
	temp_entity.distance = 0
	validation_classe1.append(temp_entity)

print("Treinamento1   array objeto ",len(training_classe1));
print("Validacao1 array objeto ",len(validation_classe1));
def euclidian (target,array):
	for i in range (len(array)): #para o total de elementos
		diff = 0
		if(array[i].classification ==-1):
			array[i].distance = 1000;  #pq eh ele mesmo e nao queremos isso 
		else :
			for j in range(len(target.attributes)): #para o total de atributos de um elemento
				diff = diff + math.pow((target.attributes[j]-array[i].attributes[j]),2);
			raiz = math.sqrt(diff)
			array[i].distance = diff
	return array 

#agora mexer na classe que tem menos que vai ser a training_classe1 e validation_classe1
alfaClasse1 = 0.65;
alfaClasse0 = 0.35;

#para o treinamento vou ter que fazer 29 vezes o knn. Entao toda vez que acabar o knn dou um shuffle.
#depois eu faco para 115 valores para poder ficar igual ao c0_treinamento 
#usar k = 5;

arrayaux = training_classe1+training_classe0;
print("Total de treinamento ",len(arrayaux));
for y in range (28):
	novos_valores = [];
	print("aqui 28");
	for x in range(c1_trainning_size): 
		target = training_classe1[x];
		arrayaux = training_classe1+training_classe0;
		array = deepcopy(arrayaux);
		array[x].classification = -1;
		distances = euclidian(target,array);
		distances.sort(key = lambda x:x.distance,reverse = False) # da um sort no array de treinamento -> array.sort(key = lambda x:x.distance,reverse = False) 
		position = random.randint(0, 4);
		escolhido = distances[position];
		novo = entity();
		novo.attributes = [];
		for i in range(len(target.attributes)):
			atual = target.attributes[i];
			proximo = escolhido.attributes[i];
			if(escolhido.real_class==0):
				att_novo = atual + (proximo-atual)*alfaClasse0;
			else :
				att_novo = atual + (proximo-atual)*alfaClasse1;
			novo.attributes.append(att_novo);
		novo.classification =1;
		novos_valores.append(novo);
	for i in range (len(novos_valores)):
		training_classe1.append(novo);
	random.shuffle(training_classe1);

print(" Treinamento1 primeira parte ",len(training_classe1));
novos_valores =[];

for x in range(115): 
	print("aqui 115");
	target = training_classe1[x];
	arrayaux = validation_classe1+validation_classe0;
	array = deepcopy(arrayaux);
	array[x].classification = -1;
	distances = euclidian(target,array);
	distances.sort(key = lambda x:x.distance,reverse = False) # da um sort no array de treinamento -> array.sort(key = lambda x:x.distance,reverse = False) 
	position = random.randint(0, 4);
	escolhido = distances[position];
	novo = entity();
	novo.attributes = [];
	for i in range(len(target.attributes)):
		atual = target.attributes[i];
		proximo = escolhido.attributes[i];
		if(escolhido.real_class==0):
			att_novo = atual + (proximo-atual)*alfaClasse0;
		else :
			att_novo = atual + (proximo-atual)*alfaClasse1;
		novo.attributes.append(att_novo);
	novo.classification =1;
	novos_valores.append(novo);
for i in range (len(novos_valores)):
	training_classe1.append(novo);
random.shuffle(training_classe1);

print("Treinamento1 segunda parte ",len(training_classe1));

#para o de validacao vou ter que fazer 29 vezes o knn. Entao toda vez que acabar o knn dou um shuffle.
#depois eu faco para 43 valores para poder ficar igual ao c0_treinamento 
arrayaux = validation_classe1+validation_classe0;
print("Total de validacao ",len(arrayaux));
for y in range (28):
	novos_valores = [];
	print("aqui 28");
	for x in range(c1_validation_size): 
		target = validation_classe1[x];
		arrayaux = validation_classe1+validation_classe0;
		array = deepcopy(arrayaux);
		array[x].classification = -1;
		distances = euclidian(target,array);
		distances.sort(key = lambda x:x.distance,reverse = False) # da um sort no array de treinamento -> array.sort(key = lambda x:x.distance,reverse = False) 
		position = random.randint(0, 4);
		escolhido = distances[position];
		novo = entity();
		novo.attributes = [];
		for i in range(len(target.attributes)):
			atual = target.attributes[i];
			proximo = escolhido.attributes[i];
			if(escolhido.real_class==0):
				att_novo = atual + (proximo-atual)*alfaClasse0;
			else:
				att_novo = atual + (proximo-atual)*alfaClasse1
			novo.attributes.append(att_novo);
		novo.classification =1;
		novos_valores.append(novo);
	for i in range (len(novos_valores)):
		validation_classe1.append(novo);
	random.shuffle(validation_classe1);

print("Validacao1 primeira parte ",len(validation_classe1));
novos_valores =[];

for x in range(43): 
	print("aqui 43");
	target = validation_classe1[x];
	arrayaux = validation_classe1+validation_classe0;
	array = deepcopy(arrayaux);
	array[x].classification = -1;
	distances = euclidian(target,array);
	distances.sort(key = lambda x:x.distance,reverse = False) # da um sort no array de treinamento -> array.sort(key = lambda x:x.distance,reverse = False) 
	position = random.randint(0, 4);
	escolhido = distances[position];
	novo = entity();
	novo.attributes = [];
	for i in range(len(target.attributes)):
		atual = target.attributes[i];
		proximo = escolhido.attributes[i];
		if(escolhido.real_class==0):
			att_novo = atual + (proximo-atual)*alfaClasse0;
		else:
			att_novo = atual + (proximo-atual)*alfaClasse1
		novo.attributes.append(att_novo);
	novo.classification =1;
	novos_valores.append(novo);
for i in range (len(novos_valores)):
	validation_classe1.append(novo);
random.shuffle(validation_classe1);

print("Validacao1 segunda parte ",len(validation_classe1));

training_classe1_formatoArquivo =[];
for i in range(len(training_classe1)):
	array_linha =[];
	linha = training_classe1[i];
	attrs = linha.attributes;
	for j in range(len(attrs)):
		array_linha.append(attrs[j]);
	array_linha.append(1); #classe
	training_classe1_formatoArquivo.append(array_linha);

validation_classe1_formatoArquivo =[];
for i in range(len(validation_classe1)):
	array_linha =[];
	linha = validation_classe1[i];
	attrs = linha.attributes;
	for j in range(len(attrs)):
		array_linha.append(attrs[j]);
	array_linha.append(1); #classe
	validation_classe1_formatoArquivo.append(array_linha);




c1_trainning =  np.array(training_classe1_formatoArquivo);
c1_validation =  np.array(validation_classe1_formatoArquivo);

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

