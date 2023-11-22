import matplotlib.pyplot as plt
import numpy as np

# Interpolacja liniowa
def linear(x1, y1, x2):
    y2 = []
    
    # Porównaj każdą wartość z wektora x2 z wartościami wektora x1 aby sprawdzić w którym przedziale znajduje się punkt.
    for i in range (len(x2)):
        for j in range (len(x1)):
            if x2[i] <= x1[j]:

                # Oblicz wagi na podstawie odległości nowego punktu od 2 sąsiadujących
                w1 = x1[j] - x2[i]
                w2 = x2[i] - x1[j-1]

                # Oblicz wartość dla punktu z wektora x2 i dodaj ją do macierzy y2
                temp = (y1[j] * w2 + y1[j-1] * w1)/(w1+w2)

                y2.append(float(temp))
                break

    return y2


# Interpolacja najbliższy-sąsiad
def closest_neightbour(x1,y1,x2):
    y2 = []

    # Porównaj każdą wartość z wektora x2 z wartościami wektora x1 aby sprawdzić do którego punktu z macierzy x1 jest bliżej
    for i in range (len(x2)):
        for j in range (len(x1)):
            if x2[i] <= x1[j]:
                w1 = x1[j] - x2[i]
                w2 = x2[i] - x1[j-1]

                # Zapisz odpowiednią wartość dla puntku z macierzy x2
                if w1 >= w2:
                    y2.append(y1[j-1])
                else:
                    y2.append(y1[j])

                break

    return y2

# Interpolacja kwadratowa
def find_quadratic_factors(x1,x2,x3,y1,y2,y3):
    y = [y1,y2,y3]
    mat = [[x1**2,x1,1],[x2**2,x2,1],[x3**2,x3,1]]
    # Odwrotność macierzy
    mat = np.linalg.inv(mat)
    # Odwrotność razy dane punkty
    wsp = mat @ y
    return wsp

def quadratic_interpolation(x1, x2, y1):
    y2=[]
    for i in range(0, len(x1), 2):
        if (i + 1) >= len(x1):
            factors = find_quadratic_factors(x1[i-2], x1[i-1], x1[i],y1[i-2], y1[i-1], y1[i])
            for j in range(len(x2)):
                if x2[j] >= x1[i]:
                    tmp = factors[0]*x2[j]**2 + factors[1]*x2[j] + factors[2]
                    y2.append(tmp)
            return y2
        if (i + 2) >= len(x1):
            factors = find_quadratic_factors(x1[i-1], x1[i], x1[i+1],y1[i-1], y1[i], y1[i+1])
            for j in range(len(x2)):
                if x2[j] >= x1[i]:
                    tmp = factors[0]*x2[j]**2 + factors[1]*x2[j] + factors[2]
                    y2.append(tmp)
            return y2

        else:
            factors = find_quadratic_factors(x1[i], x1[i+1], x1[i+2],y1[i], y1[i+1], y1[i+2])
            for j in range(len(x2)):
                if x2[j] >= x1[i] and x2[j] < x1[i+2]:
                    tmp = factors[0]*x2[j]**2 + factors[1]*x2[j] + factors[2]
                    y2.append(tmp)

# Interpolacja trzeciego stopnia
def find_cubic_factors(x1,x2,x3,x4,y1,y2,y3,y4):
    y = [y1,y2,y3,y4]
    mat = [[x1**3,x1**2,x1,1],[x2**3,x2**2,x2,1],[x3**3,x3**2,x3,1],[x4**3,x4**2,x4,1]]
    # Odwrotność macierzy
    mat = np.linalg.inv(mat)
    # Odwrotność razy dane punkty
    wsp = mat @ y
    return wsp
def cubic_interpolation(x1,x2,y1):
    y2=[]
    for i in range(0, len(x1), 3):
        if (i + 1) >= len(x1):
            factors = find_cubic_factors(x1[i-3],x1[i-2], x1[i-1], x1[i],y1[i-3], y1[i-2], y1[i-1], y1[i])
            for j in range(len(x2)):
                if x2[j] >= x1[i]:
                    tmp = factors[0]*x2[j]**3 + factors[1]*x2[j]**2 + factors[2]*x2[j] + factors[3]
                    y2.append(tmp)
            return y2

        if (i + 2) >= len(x1):
            factors = find_cubic_factors(x1[i-2], x1[i-1], x1[i], x1[i+1], y1[i-2], y1[i-1], y1[i], y1[i+1])
            for j in range(len(x2)):
                if x2[j] >= x1[i]:
                    tmp = factors[0]*x2[j]**3 + factors[1]*x2[j]**2 + factors[2]*x2[j] + factors[3]
                    y2.append(tmp)
            return y2

        if (i + 3) >= len(x1):
            factors = find_cubic_factors(x1[i-1], x1[i], x1[i+1], x1[i+2], y1[i-1], y1[i], y1[i+1], y1[i+2])
            for j in range(len(x2)):
                if x2[j] >= x1[i]:
                    tmp = factors[0]*x2[j]**3 + factors[1]*x2[j]**2 + factors[2]*x2[j] + factors[3]
                    y2.append(tmp)
            return y2
        else:
            factors = find_cubic_factors(x1[i], x1[i+1], x1[i+2], x1[i+3], y1[i], y1[i+1], y1[i+2], y1[i+3])
            for j in range(len(x2)):
                if x2[j] >= x1[i] and x2[j] < x1[i+3]:
                    tmp = factors[0]*x2[j]**3 + factors[1]*x2[j]**2 + factors[2]*x2[j] + factors[3]
                    y2.append(tmp)
