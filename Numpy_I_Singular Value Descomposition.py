# -*- coding: utf-8 -*-

#Singular Value Descomposition (SVD) con Numpy

La factorización de matrices o descomposición de matrices consiste en describir una matriz utilizando sus elementos constitutivos. El método más conocido y utilizado para esta factorización matricial es la descomposición en valores singulares. Se utiliza para la compresión, eliminación de ruido, reducción de datos...

**IMPORTANTE:** si te aparece el error 'TruncatedSVD' object is not callable reinicia el entorno de ejecución y ejecuta todas las celdas de nuevo (puedes hacer esto desde el menú de entorno de ejecucción de la barra superior o pulsando Ctrl+ M + . y después Ctrl+ F9).

##Descomposicion en valores singulares (SVD)

Una matriz A de m filas y n columnas se puede descomponer en:
* Matirz U (m x m)
* Matriz Sigma (m x n diagonal)
* Matriz V^T o V transpuesta (n x n )

      A(m x n)= U (m x m) . Sigma (m x n) . V^T (n x n)

Donde los elementos de la diagonal de la matriz sigma son los valores singulares, las columnas de la matriz U son los vectores singulares izquierdos de A (left-singular vectors) y las columnas de la matriz V son los vectores singulares derechos de A (rigth-singular vectors)

Se pueden obtener las matrices U, sigma y V^T importando la funcion svd del módulo numpy.linalg.



Crea la función calcular_svd que tomando una matriz como parámetro devuelva un diccionario que contenga las claves 'U', 's' y 'V_T' y su respectiva matriz como valor.

"""
Ejemplo

dada la matriz A:

      [[1 2]
      [5 7]
      [8 9]]

Entrada:

      print(calcular_svd(A))

Salida:

      {'U': array([[-0.14599302, -0.55018941, -0.8221786 ],
       [-0.57456139, -0.62939351,  0.52320456],
       [-0.80533549,  0.5487763 , -0.22423053]]), 's': array([14.9398133 ,  0.89553249]), 'V_T': array([[-0.6333067 , -0.77390091],
       [ 0.77390091, -0.6333067 ]])}
"""

from numpy.linalg import svd
import numpy as np
def calcular_svd(matriz):
  result=svd(matriz)
  dic={}
  dic['U']=result[0]
  dic['s']=result[1]
  dic['V_T']=result[2]
  return dic

A=np.array([[1,2],[5,7],[8,9]])
print(calcular_svd(A))

#@title Comprueba la funcion calcular_svd
from numpy.linalg import svd
import numpy as np
A_check=np.array([[10,23],[58,71],[89,98],[16,40]])
def check():
  if str(calcular_svd(A_check))=="{'U': array([[-0.14353134,  0.43448506, -0.28421655, -0.84252148],\n       [-0.54635191,  0.04501559, -0.75033941,  0.36941028],\n       [-0.78816521, -0.36517047,  0.4505506 , -0.20603468],\n       [-0.24432346,  0.82209905,  0.391428  ,  0.33353157]]), 's': array([167.79646837,  16.10419829]), 'V_T': array([[-0.63874767, -0.76941628],\n       [-0.76941628,  0.63874767]])}":
    return 'Correcto'
  else:
    return 'Incorrecto'
check()

"""##Reconstruir la Matriz dese SVD

Es posible reconstruir la matriz A desde los elementos de las matrices U,sigma y V^T.

Antes hemos dicho que A se puede descomponer en:

      A(m x n)= U (m x m) . Sigma (m x n) . V^T (n x n)

Por lo que si multiplicamos matricialmente U,sigma y V^T volveríamos a obtener A.

Crea la función reconstruir_svd que tome como parámetro una matriz U, una matriz sigma y una matriz V^T y devuelva la matriz resultante.

Ten en cuenta que la matriz sigma obtenida al aplicar la función svd es de tamaño 1 x n y debería ser m x n para poder multiplicarse matricialmente. La función reconstruir_svd debe comprobar las dimensiones de la matriz sigma y corregirlas si no son las adecuadas.

Entrada

      dic=calcular_svd(A)
      print(reconstruir_svd(dic['U'],dic['s'],dic['V_T']))

Salida

      [[1. 2.]
      [5. 7.]
      [8. 9.]]
"""

from numpy.linalg import svd
import numpy as np
#Crea la funcion reconstruir_svd
def reconstruir_svd(U, s, V_T):
    # Obtener las dimensiones de U, s y V_T
    U_forma = np.shape(U)
    V_T_forma = np.shape(V_T)

    # Ajustar las dimensiones de Sigma según U y V_T
    sigma = np.zeros((U_forma[0], V_T_forma[0]))
    sigma[:min(U_forma[1], V_T_forma[0]), :min(U_forma[1], V_T_forma[0])] = np.diag(s)

    # Reconstruir la matriz original
    A = U.dot(sigma.dot(V_T))
    return A

dic1=calcular_svd(A)
print(dic1)
print(reconstruir_svd(dic1['U'],dic1['s'],dic1['V_T']))

#@title Comprueba la funcion reconstruir_svd
A_check2=np.array([[3,5,8,10],[11,13,25,19],[2,6,17,29],[3,5,2,1],[7,9,5,4]],float)
dic_check2=calcular_svd(A_check2)
result=reconstruir_svd(dic_check2['U'],dic_check2['s'],dic_check2['V_T'])
def check2():
  if str(result)==str(A_check2):
    return 'Correcto'
  else:
    return 'Incorrecto'

check2()

"""#SVD para Pseudoinversa

Si la matriz es cuadrada es posible calcular la matriz inversa, pero si se trata de una matriz cuadrada (no tiene el mismo numero de filas y columnas) no es posible calcular la matriz inversa.

En cambio si es posible calcular la matriz pseudoinversa, tambien llamada inversa generalizada o inversa de Moore-Penrose, en honor a dos descubridores del método que trabajaron independientes el uno del otro.

El pseudoinverso se denota como A^+, donde A es la matriz que se invierte y + es un superíndice. Para calcularla se aplica la siguiente fórmula:

      A^+ = V . D^+ . U^T

Siendo los SVD U,Sigma y V^T podemos obtener:

* U^T: Calculando la transpuesta de U
* V: calculando la transpuesta de V^T, puesto que como sanes la transpuesta de la transpuesta es la original.
* D^+: se puede calular creando la matriz diagonal sigma y calculando el valor inverso de cada elelemtno de la diagonal.

Crear la función pseuinversa que tomando una matriz como parámetro calcule su pseudoinversa.


"""

from numpy.linalg import svd
import numpy as np
def pseudoinversa(matriz):
  dic2=calcular_svd(matriz)
  d=1/dic2['s']
  D=np.zeros(matriz.shape)
  D[:matriz.shape[1], :matriz.shape[1]]=np.diag(d)
  A_gen_inv=dic2['V_T'].T.dot(D.T).dot(dic2['U'].T)
  return A_gen_inv

A=np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])

print(pseudoinversa(A))

# opción 2

from numpy.linalg import svd
import numpy as np

# Crea la función pseudoinversa
def pseudoinversa_2(matriz):
    # Comprobar si es cuadrada y se puede tener la matriz inversa
    if matriz.shape[0] == matriz.shape[1]:
        matriz_inv = np.linalg.inv(matriz)
        return matriz_inv
    else:
        U, s, V_T = np.linalg.svd(matriz)
        U_T = np.transpose(U)
        V = np.transpose(V_T)

        # Construir la pseudoinversa de Sigma
        sigma_inv = np.zeros((matriz.shape[1], matriz.shape[0]))
        s_inv = 1/s
        np.fill_diagonal(sigma_inv, s_inv)

        # Calcular la pseudoinversa
        matriz_pseudo = V.dot(sigma_inv).dot(U_T)
        return matriz_pseudo

A = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])

print(pseudoinversa_2(A))

A_check3=np.array([[3,5],[2,7],[9,2],[6,4]])
result=pseudoinversa(A_check3)
result

from numpy.linalg import svd
import numpy as np
#@title Comprueba la funcion pseudoinversa
A_check3=np.array([[3,5],[2,7],[9,2],[6,4]])
result=pseudoinversa(A_check3)
def check3():
  if str(result)==str(np.linalg.pinv(A_check3)):
    return 'Correcto'
  else:
    return 'Incorrecto'
check3()

"""##SVD para reducir las dimensiones lineales de una matriz

Los datos con un gran número de características, es decir más características (columnas) que observaciones (filas), pueden reducirse a un subconjunto más pequeño de características que sean más relevantes para el problema de predicción.

El resultado es una matriz con un rango inferior que se aproxima a la matriz original.

Para ello, podemos realizar una operación SVD sobre los datos originales y seleccionar los k mayores valores singulares de Sigma. Estas columnas se pueden seleccionar de Sigma y las filas de V^T.

Entonces se puede reconstruir una matriz B aproximada de la matriz A original aplicando la siguiente fórmula:

      B (m x n) = U (m x m) . Sigmak (m x k). V^Tk (k x n)

En el procesamiento del lenguaje natural, este enfoque puede utilizarse en matrices de ocurrencias de palabras o frecuencias de palabras en los documentos (como el que preparaste en el proyecto de Bag of Words) y se denomina análisis semántico latente o indexación semántica latente.

En la práctica podemos utilizar otra matriz aún más simplificada que B, denominada T. Hay dos formas de calcular T:

      T (m x k)= U (m x m). Sigmak (m x k)
      T (m x k)= A (m x n) . V^KT (n x k)

Donde V^KT se calcula realizando la transpuesta al subconjunto de elementos de V^T (k x n)
Crea la funcion reducir_dimensiones1 que tome como parámetro una matriz y el número de características (columnas, k valores singulares) que se desea tener en cuenta y devuelva la matriz T (m x k) aplicando la primera fórmula y otra función reducir_dimensiones2 que haga lo mismo pero aplicando la segunda fórmula.

"""

from numpy.linalg import svd
import numpy as np
#Crea la funcion reducir_dimensiones1

def reducir_dimensiones1(matriz,k_valores):
  dic3=calcular_svd(matriz)
  sigma=np.zeros(matriz.shape)
  sigma[:matriz.shape[0],:matriz.shape[0]]=np.diag(dic3['s'])
  sigma=sigma[:,:k_valores]
  T=dic3['U'].dot(sigma)
  return T


#Crea la funcion reducir_dimensiones2
def reducir_dimensiones2(matriz,k_valores):
  dic4=calcular_svd(matriz)
  V_T=dic4['V_T'][:k_valores, :]
  T=matriz.dot(V_T.T)
  return T


A=np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])

print(reducir_dimensiones1(A,2))

print('.......')

print(reducir_dimensiones2(A,2))

from numpy.linalg import svd
import numpy as np
A_check4=np.array([[11,32,53,14,35,56,17,38,59,110],[211,612,713,214,615,716,217,618,719,220],[421,622,923,424,625,926,427,628,929,30]])
def check4():
  if str(reducir_dimensiones1(A_check4,3))==str(reducir_dimensiones2(A_check4,3)):
    return 'Correcto'
  else:
    return 'Incorrecto'
check4()

#@title Consigue el Token para corregir en Nodd3r:
from numpy.linalg import svd
import numpy as np
import hashlib
A_token1=np.array([[1,3,4],[4,9,5],[1,2,3],[7,8,9]])
A_token2=np.array([[1,3,4,9,10],[4,9,5,7,3],[1,2,3,4,5]])
dic_token=calcular_svd(A_token1)
pwd = hashlib.sha256((str(calcular_svd(A_token1))+str(reducir_dimensiones1(A_token2,2))+str(reconstruir_svd(dic_token['U'],dic_token['s'],dic_token['V_T']))).encode())
print('El token es:\n',pwd.hexdigest())

"""## Token Correcto

45963a47073d2c27bb2f41b4271e932d9407ad9a6c5c05d5345274146251b41e
"""

calcular_svd(A_token1)

"""Una vez que has visto como se calcula manualmente la pseudoinversa y la matriz transformada T para la reducción dimensional de matrices, existen funciones ya predefinidas que facilitan estos procesos.

"""

from numpy.linalg import svd
import numpy as np
#calcular pseudoinversa

matriz=np.array([[1,3,4,9,10],[4,9,5,7,3],[1,2,3,4,5]])
print('Matriz original:\n', matriz)
print('Matriz pseudoinversa:\n',np.linalg.pinv(matriz))

#calcular T transformada
import sklearn.decomposition as skd
svd = skd.TruncatedSVD(n_components=2)
svd.fit(matriz)
result = svd.transform(matriz)
print('Matriz transformada:\n', result)

print(hashlib.sha256((str(calcular_svd(A_token1))).encode()).hexdigest())
print(hashlib.sha256((str(reducir_dimensiones1(A_token2,2))).encode()).hexdigest())
print(hashlib.sha256((str(reconstruir_svd(dic_token['U'],dic_token['s'],dic_token['V_T']))).encode()).hexdigest())
