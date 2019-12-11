#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:16:55 2019

@author: nd235
"""
from collections import defaultdict
import numpy


class word2vec():
    
    
    def __init__(self):
        self.n = settings["n"]
        self.lr = settings["learning_rate"]
        self.epochs = settings["epochs"]
        self.window = settings["window_size"]
    
    # Preprocesamiento de los datos
    def generate_training_data(self, settings, corpus):
        # Conteo de palabras únicas usando un diccionario
        word_count = defaultdict(int)
        for row in corpus:
            for word in row:
                word_count[word]+=1
        # Cantidad de palabras en el vocabulario
        self.v_count = len(word_count.keys())
        # Generar diccionarios de búsqueda
        self.word_list = list(word_count.keys())
        # Diccioario palabra:índice
        self.word_index = dict((word, i) for i, word in enumerate(self.word_list))
        # Diccionario índice:palabra
        self.index_word = dict((i, word) for i, word in enumerate(self.word_list))
        training_data = []
        # Inicia bucle en cada elemento del corpus
        for sentence in corpus:
            sent_len = len(sentence)
            # Bucle sobre cada palabra de cada enunciado
            for i,word in enumerate(sentence):
                # Convertir en 1hot encodding
                w_target = self.word2onehot(sentence[i])
                # Bucle sobre la ventana de contexto
                w_contex = []
                for j in range(i-self.window, i+self.window+1):
                    # Criterios para la ventana de contexto
                    # 1.- La palabra objetivo no puede estar en el contexto
                    # (j != i)
                    # 2.- El índice debe ser mayor o igual que 0 (j>=0) de lo
                    # contrario el índice estará fuera de rango
                    # 3.- El índice debe ser menor o igual a la longitud del
                    # enunciado (j<=sent_len-1)
                    if j != i and j <= sent_len-1 and j >= 0:
                        # Añade la representación 1hot de cada palabra a w_context
                        w_contex.append(self.word2onehot(sentence[j]))
                        #print(sentence[i], sentence[j])
                        # training_data contiene la representación 1hot de la
                        # palabra objetivo y las palabras de contexto
                training_data.append([w_target, w_contex])
        return numpy.array(training_data)

    # Generación de representaciones 1hot
    def word2onehot(self, word):
        # word_vec - inicializa un vector en blanco
        word_vec = [0]*self.v_count
        # Obtener la id de cada palabra de word_index
        #word_index = self.word_index[word]
        #cambiar el valor de 0 a 1 de acuerdo al índice de la palabra
        word_vec[self.word_index[word]] = 1
        return word_vec

    # Paso hacia adelante
    def forward_pass(self, x):
        # x es la representación 1hot de la palabra objetivo, shape v_countx1
        # Se ejecuta en la primera matriz transpuesta (w1) para obtener la
        # capa oculta shape nxv_count dot v_countx1 da como resultado v_countx1
        h = numpy.dot(self.w1.T, x) # x = w_t de entrada en la función
        # Producto punto de la capa oculta h y la matriz transpuesta w2
        # con shape v_countxn dot nx1 da como resultado v_countx1
        u = numpy.dot(self.w2.T, h)
        # Pasar 1xv_count por softmax para forzar el rango de [0, 1] en cada
        # elemento, da como resultado 1xv_count-1
        y_c = self.softmax(u)
        #print("{}\n{}\n{}\n".format(y_c, h, u))
        return y_c, h, u
 
    # Función de activación softmax
    def softmax(self, x):
        # softmax(x_{i}) = \frac{exp(x_i)}{\sum_{j} \exp(x_j)}
        e_x = numpy.exp(x - numpy.max(x))
        return e_x / e_x.sum(axis=0)
    
    # Retropropagación de errores
    def backprop(self, e, h, x):
        # El vector E representa la suma de filas de las predicciones de error
        # por cada palabra de contexto para la palabra objetivo actual
        # En el paso hacia atrás se toma la derivada de E con respecto a w2
        # h - shape nx1, e - shape v_count, dl_dw2 - shape nxv_count
        # Esta fución no retorna un valor, solo actualiza las variables
        # self.w1 y self.w2
        dl_dw2 = numpy.outer(h, e) # numpy.outer = producto exterior
        # x - shape 1xv_count-1, w2 - 5xv_count-1, e.T - v_count-1x1
        # x - 1xv_count-1, np.dot() - sx1, dl_dw - v_count-1x5
        dl_dw1 = numpy.outer(x, numpy.dot(self.w2, e.T))
        # Actualización de los pesos con la tasa de aprendizaje (self.lr)
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)
        pass

    # Bucle de entrenamiento
    def train(self, training_data):
        # Inicializar las matrices de pesos
        # s1 y s2 deben ser inicializados al azar
        # w1 = matriz de embeddings
        # w2 = matriz de contexto
        self.w1 = numpy.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = numpy.random.uniform(-1, 1, (self.n, self.v_count))
        
        # Bucle durante cada epoch de entrenamiento
        for i in range(self.epochs):
            # Inicialización de la pérdida en 0
            self.loss = 0
        
        # Bucle sobre cada muestra de entrenamiento
        # w_t = vector de la palabra objetivo
        # w_c = vectores de las palabras de contexto
        for w_t, w_c in training_data:
            # Paso hacia adelante - vector de las palabras objetivo (w_t)
            # para obtener (1) la predicción "y" usando la función softmax
            # (2) la matriz de la capa oculta, (3) la capa de salida (h) antes
            # de la aplicación de softmax
            y_pred, h, u = self.forward_pass(w_t)
            
            # Cálculo de error
            # (1) Para una palabra objetivo, calculamos la diferencia entre 
            # y_pred y cada una de las palabras de contexto
            # (2) Sumamos las diferencias usando numpy.sum para obtener el error
            # de cada palabra objetivo
            E = numpy.sum([numpy.subtract(y_pred, word) for word in w_c], axis=0)
            
            # Retropropagación de errores
            # word2vec emplea SDG (descenso de gradiente estocástico) para 
            # calcular la pérdida en la capa de salida
            self.backprop(E, h, w_t)
            
            # Cálculo de la pérdida
            # Esta función cuenta con dos partes
            # (1) -ve suma de todas las salidas +
            # (2) longitud de todas las palabras de contexto * la suma logarítmica
            # de todos los elementos (Exponente-ed) en la capa de salida antes
            # de softmax "u"
            # word.index(1) retorna el índice del vector de la palabra de contexto
            # con valor de 1
            # u[word.index(1)] retorna el valor de la capa de salida antes de la
            # aplicación de la función softmax
            
            self.loss += -numpy.sum([u[word.index(1)] for word in w_c] + 
                                       len(w_c) * numpy.log(numpy.sum(numpy.exp(u))))
        print("Epoch: {}\tLoss: {}\n".format(i, self.loss))

    # Función para obtener los vectores de cada palabra
    def get_word_vector(self, word):
        w_index = self.word_index[word] # busca el índice de la palabra
                                        # en el vocabulario
        v_w = self.w1[w_index]  # obtiene el vector correspondiente al índice
        return v_w
    
    # Similitud de palabras mediante función coseno
    def word_similarity(self, word, top_n=5):
        v_w1 = self.get_word_vector(word) # obtiene el vector de la palabra
        word_sim = {} # crea un diccionario en blanco
        
        for i in range(self.v_count):
            # Busca las palabras similares en el rango del vocabulario
            v_w2 = self.w1[i] # asigna el vector de cada palabra del vocabulario
            theta_sum = numpy.dot(v_w1, v_w2) # producto punto entre vectores
            theta_den = numpy.linalg.norm(v_w1) * numpy.linalg.norm(v_w2) # normas
            theta = theta_sum / theta_den
            
            word = self.index_word[i] # obtiene la palabra mediante el índice
            word_sim[word] = theta # asigna la similitud al diccionario
        
        # Ordena el diccionario en forma descendiente
        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
        
        # Retorna el diccionario de similitud ignorando el primer resultado
        # ya que siempre será la smimilitud de la palabra consultada consigo
        return words_sorted[1:top_n+1]

        
    
    def save_model(self, path):
        with open("{}".format(path), "w", encoding="utf8") as sf:
            sf.write("{} {}\n".format(self.v_count, self.n))
            for i in sorted(self.word_index):
                word = self.word_index[i]
                sf.write("{} {}\n".format(self.index_word[word], self.w1[word]))
        pass
    
    # Función para el cálculo de la similitud entre palabras
    

doc = ["el procesamiento del lenguaje natural es una actividad muy interesante",
        "esta será una lista de documentos para el entrenamiento de prueba",
        "cada entrada tiene que ser divisible y por lo tanto no una lista",
        "el héroe de leyenda pertenece al sueño de un destino"]

corpus = [[w.lower() for w in text.split()] for text in doc]
#print(corpus)

settings = {
        "window_size": 5,     # tamaño de la ventana de contexto
        "n": 5,               # dimensiones de los embeddings
                              # también se refiere a las dim de la capa oculta
        "epochs": 10,         # número de epochs de entrenamiento
        "learning_rate":0.001 # tasa de aprendizaje
        }

# iniclialización del objeto
w2v = word2vec()
# Array numpy con la representación 1hot de cada palabra, contexto
training_data = w2v.generate_training_data(settings, corpus)
w2v.train(training_data)
#print(w2v.get_word_vector("procesamiento"))
print(w2v.word_similarity("destino"))
w2v.save_model("modelo_de_prueba.txt")        