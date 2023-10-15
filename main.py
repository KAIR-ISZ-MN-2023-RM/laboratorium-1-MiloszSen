import numpy as np
import math
def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if (r > 0 and h > 0):
        P = 2 * np.pi * r * r + 2 * np.pi * r * h
        return P
    else:
        return math.nan

def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    if n <= 0 or isinstance(n, float):
        return None
    elif n == 1:
        return np.array([1.0])
    elif n == 2:
        return np.array([1.0, 1.0])
    else:
        tab = [1.0, 1.0]
        for i in range(2, n):
            tab.append(tab[i - 1] + tab[i - 2])
        return np.array(tab, dtype = float).reshape(1, -1)


def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    Mdet = np.linalg.det(M)
    if Mdet == 0:
        Minv = math.nan
    else:
        Minv = np.linalg.inv(M)
    Mt = np.transpose(M)
    return Minv, Mt, Mdet

def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """

    if m > 0 and n > 0 and isinstance(m, int) and isinstance(n, int) :
        zero_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range (n):
                if i > j:
                    zero_matrix[i, j] = i
                else:
                    zero_matrix[i, j] = j
        return zero_matrix
    else:
        return None
