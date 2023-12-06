import numpy as np
import matplotlib.pyplot as plt


def distancia_euclideana(x1, x2): # Recibe dos arrays de centroides.
  return np.sqrt(np.sum((x1-x2)**2))  # Devuelve la suma de las distancias entre los puntos en los centroides.

class KMeans:

  """
  Se inicializan los centros de los grupos
  Se repite el ciclo hasta la convergencia
    Se actualizan las etiquetas de los grupos: se asignan los puntos a el centroide mas cercano
    Se actualizan los centros de cada grupo: establecer el centro al promedio de cada grupo
  """
  def __init__(self, K=5, max_iteraciones=100, graficar_pasos=False):
    self.K = K
    self.max_iteraciones = max_iteraciones
    self.graficar_pasos = graficar_pasos
    
    # lista de indices muestra para cada grupo
    self.grupos = [[] for _ in range(self.K)]

    # Los centros (promedio de vector) para cada grupo
    self.centroides = []

  def predecir(self, X):
    self.X = X
    # Basado en la forma de la muestra de datos, se almacena la cantidad de muestras y caracteristicas en el objeto de datos.
    self.n_muestras, self.n_caracteristicas = X.shape

    # Inicializamos los centroides de forma aleatoria
    indices_muestra_aleatoria = np.random.choice(self.n_muestras, self.K, replace=False)
    self.centroides = [self.X[indices] for indices in indices_muestra_aleatoria]

    # Optimizacion de los grupos
    for _ in range(self.max_iteraciones):
      # Asignar las muestras a los centroides mas cercanos (crear los grupos)
      self.grupos = self._crear_grupos(self.centroides)


      # Validamos si se graficaran los pasos y graficamos antes y despues de verificar la convergencia.
      if self.graficar_pasos:
        self.graficar()

      # Calcular los nuevos centroides desde los grupos
      centroides_old = self.centroides
      self.centroides = self._get_centroides(self.grupos)

      if self._converge(centroides_old, self.centroides):
        break

      if self.graficar_pasos:
        self.graficar()

      # CLasificar las muestras de datos en los indices de sus grupos
    return self._get_etiquetas_grupo(self.grupos)

  def _crear_grupos(self, centroides):
    # Asignar las muestras a el centroide mas cercano
    grupos = [[] for _ in range(self.K)]
    for indice, muestra in enumerate(self.X):
      indice_centroide = self._centroide_cercano(muestra, centroides)
      grupos[indice_centroide].append(indice)
    
    return grupos

  def _get_centroides(self, grupos):
    # Asignar el valor promedio de los grupos a los centroides.
    centroides = np.zeros((self.K, self.n_caracteristicas)) #G eneramos un array de dimensiones (K, N_car)
    for indice_grupo, grupo in enumerate(grupos): # Por cada elemento en el array de grupos
      # Por cada grupo se calcula su promedio, usando el indice del misno (se obtiene de la enumeracion anterior)
      promedio_grupo = np.mean(self.X[grupo], axis=0) # Axis es el eje, se obtiene el promedio sobre este (arrreglo unidimensional)
      # Y se le asigna a cada centroide en el vector
      centroides[indice_grupo] = promedio_grupo
    
    return centroides

  def _converge(self, centroides_old, centroides):
    # Distancias entre los centroides de antes y despues de la iteracion, para cada centroide.
    distancias = [distancia_euclideana(centroides_old[i], centroides[i]) for i in range(self.K)]

    #Funcion sum de python al distancias ser una lista de python
    return sum(distancias) == 0

  def _get_etiquetas_grupo(self, grupos):
    # Cada muestra de datos obtendra la etiqueta del grupo al que fue asignada
    etiquetas = np.empty(self.n_muestras)
    for indice_grupo, grupo in enumerate(grupos):
      for indice_muestra in grupo:
        etiquetas[indice_muestra] = indice_grupo

    return etiquetas

  def _centroide_cercano(self, muestra, centroides):
    # Calcula la distancia de la muestra actual a cada centroide y devuelve el mas cercano
    distancias = [distancia_euclideana(muestra, punto) for punto in centroides]

    indice_mas_cercano = np.argmin(distancias)

    return indice_mas_cercano

  def graficar(self):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, index in enumerate(self.grupos):
        punto = self.X[index].T
        ax.scatter(*punto)

    for punto in self.centroides:
        ax.scatter(*punto, marker="x", color="black", linewidth=2)

    plt.show()


#Prueba
if __name__ == "__main__":
    #Se genera una semilla aleatoria
    np.random.seed(42)

    #Se usa la funcion make_blobs libreria de data sets de sklearn
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=5, n_samples=500, n_features=2, shuffle=True, random_state=40)

    #Se muestra la forma de la coleccion de datos (la funcion shape regresa las dimensiones del arreglo de datos)
    print(X.shape)

    #Almacenamos cuantos gros existen en la coleccion de datos
    grupos = len(np.unique(y))
    print(grupos)
    
    #Usamos nuestra funcion para identificar los grupos (clusters) y sus centroides
    k = KMeans(K=grupos, max_iteraciones=150, graficar_pasos=True)
    y_prediccion = k.predecir(X)

    k.graficar()