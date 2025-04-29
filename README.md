# Descomposición en Valores Singulares (SVD) con NumPy y Colab

Este proyecto explora la **Descomposición en Valores Singulares (SVD)**, una poderosa técnica de álgebra lineal utilizada para la **reducción de dimensionalidad**, **compresión de datos**, **eliminación de ruido** y **cálculo de pseudoinversas**.

> ⚠️ **Importante**: Si recibes el error `'TruncatedSVD' object is not callable` en Google Colab, reinicia el entorno de ejecución (`Ctrl + M + .` y luego `Ctrl + F9`).

---

## Contenido

- ✅ Cálculo manual de SVD (`calcular_svd`)
- ✅ Reconstrucción de matrices desde SVD (`reconstruir_svd`)
- ✅ Cálculo de pseudoinversas (inversa de Moore-Penrose) (`pseudoinversa`)
- ✅ Reducción de dimensionalidad con dos enfoques (`reducir_dimensiones1`, `reducir_dimensiones2`)
- ✅ Comparaciones y validaciones automáticas (`check`, `check2`, `check3`, `check4`)
- ✅ Cálculo de token de validación para Nodd3r
- ✅ Alternativas usando `scikit-learn` (`TruncatedSVD`)

---

## Requisitos

Este proyecto utiliza:

- Python 3
- NumPy
- scikit-learn (solo para validación alternativa)

Puedes instalar los requisitos ejecutando:

```bash
pip install numpy scikit-learn
````

## Estructura del proyecto
├── main.ipynb               # Código principal con explicación didáctica

├── .gitignore               # Ignora archivos temporales, checkpoints, etc.

├── README.md                # Este archivo

## Ejemplo de uso
````
python
Copiar
Editar
import numpy as np

A = np.array([[1, 2], [5, 7], [8, 9]])
dic = calcular_svd(A)
A_reconstruida = reconstruir_svd(dic['U'], dic['s'], dic['V_T'])
print(A_reconstruida)
````

## Autovalidación
El cuaderno incluye funciones de chequeo (check, check2, etc.) para validar automáticamente tus funciones frente a resultados esperados.

## Créditos
Desarrollado como parte del aprendizaje práctico de técnicas de álgebra lineal aplicadas a la ciencia de datos y procesamiento de lenguaje natural (NLP).

