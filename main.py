import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Interpolacja liniowa
def linear(x1, y1, x2):
    y2 = []
    
    # Porównaj każdą wartość z wektora x2 z wartościami wektora x1 aby sprawdzić w którym przedziale znajduje się punkt.
    for i in range (len(x2)):
        for j in range (len(x1)):
            if x2[i] == x1[j]:
                y2.append(y1[j])
                break
            elif x2[i] < x1[j]:

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

def apply_filter(image, matrix):
    # resize matrix
    resized_matrix = np.tile(matrix, ( (image.shape[0] // matrix.shape[0])+1, (image.shape[1] // matrix.shape[1])+1, 1))
    resized_matrix = resized_matrix[:image.shape[0], :image.shape[1]]

    # Pomnóż obie macierze przez siebie (element-wise)
    result = image*resized_matrix
    return result

def interpolacja_liniowa_czerwonego(matrix):
    matrix_size_rows = matrix.shape[0]
    matrix_size_col = matrix.shape[1]
    for i in range(0,matrix_size_rows,2):
        for j in range(2,matrix_size_col,2):
            matrix[i,j] = (matrix[i,j-1] + matrix[i,j+1])/2

    for kolumna in range(1,matrix_size_col):
        for wiersz in range(1,matrix_size_rows-1,2):
            matrix[wiersz,kolumna] = (matrix[wiersz-1,kolumna] + matrix[wiersz+1,kolumna])/2
    return matrix

def interpolacja_liniowa_zielonego(matrix):
    matrix_size_rows = matrix.shape[0]
    matrix_size_col = matrix.shape[1]
    for wiersz in range(0,matrix_size_rows,2):
        for kolumna in range(1,matrix_size_col-1,2):
            matrix[wiersz,kolumna] = (matrix[wiersz,kolumna-1] + matrix[wiersz,kolumna+1])/2

    for kolumna in range(0,matrix_size_col,2):
        for wiersz in range(1,matrix_size_rows-1,2):
            matrix[wiersz,kolumna] = (matrix[wiersz-1,kolumna] + matrix[wiersz+1,kolumna])/2
    return matrix

def interpolacja_liniowa_niebieskiego(matrix):
    matrix_size_rows = matrix.shape[0]
    matrix_size_col = matrix.shape[1]
    for wiersz in range(1,matrix_size_rows,2):
        for kolumna in range(1,matrix_size_col-1,2):
            matrix[wiersz,kolumna] = (matrix[wiersz,kolumna-1] + matrix[wiersz,kolumna+1])/2

    for kolumna in range(matrix_size_col):
        for wiersz in range(2,matrix_size_rows,2):
            matrix[wiersz,kolumna] = (matrix[wiersz-1,kolumna] + matrix[wiersz+1,kolumna])/2
    return matrix

def suma_macierzy(matrix):
    sum = 0
    matrix_size_rows = matrix.shape[0]
    matrix_size_col = matrix.shape[1]
    for wiersz in range(matrix_size_rows):
        for element in range(matrix_size_col):
            sum += matrix[wiersz][element]
    return sum
def wyłuskaj_podmacierz(macierz, start_wiersz, start_kolumna, liczba_wierszy, liczba_kolumn):
    podmacierz = np.array(macierz[start_wiersz:start_wiersz+liczba_wierszy, start_kolumna:start_kolumna+liczba_kolumn])
    return podmacierz

def interpolacja_fuji(matrix, matrix_kopia,kolor):
    if kolor == "red" or kolor == "blue":
        dzielnik = 8
    elif kolor == "green":
        dzielnik = 20
    else:
        return
    matrix_size_rows = matrix.shape[0]
    matrix_size_col = matrix.shape[1]
    for wiersz in range(2,matrix_size_rows):
        for kolumna in range(2,matrix_size_col-1):
            if wiersz == matrix_size_rows-1 or kolumna == matrix_size_col-1:
                break
            if matrix[wiersz][kolumna] != 0:
                continue

            podmacierz = wyłuskaj_podmacierz(matrix, wiersz-2, kolumna-2, 6,6)
            matrix_kopia[wiersz][kolumna] = suma_macierzy(podmacierz)/dzielnik

def main():
            # Nakładnanie filtru Bayera na obraz, interpolacja i połączenie obrazów.
    with Image.open("kicia.jpeg") as im:
        photo_array = np.array(im)

        red_Bayer_filtr = np.array([[[0,0,0], [1,0,0]],
                                [[0,0,0], [0,0,0]]])
        
        green_Bayer_filtr = np.array([[[0,1,0], [0,0,0]],
                                  [[0,0,0], [0,1,0]]])
        
        blue_Bayer_filtr = np.array([[[0,0,0], [0,0,0]],
                                 [[0,0,1], [0,0,0]]])

        red_Fuji_filtr = np.array([[[0,0,0], [0,0,0], [1,0,0], [0,0,0], [1,0,0], [0,0,0]],
                                [[1,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,0,0], [0,0,0], [1,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [1,0,0], [0,0,0], [0,0,0], [0,0,0], [1,0,0]],
                                [[0,0,0], [0,0,0], [0,0,0], [1,0,0], [0,0,0], [0,0,0]],
                                [[1,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]])

        green_Fuji_filtr = np.array([[[0,1,0], [0,0,0], [0,0,0], [0,1,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,1,0], [0,1,0], [0,0,0], [0,1,0], [0,1,0]],
                                [[0,0,0], [0,1,0], [0,1,0], [0,0,0], [0,1,0], [0,1,0]],
                                [[0,1,0], [0,0,0], [0,0,0], [0,1,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,1,0], [0,1,0], [0,0,0], [0,1,0], [0,1,0]],
                                [[0,0,0], [0,1,0], [0,1,0], [0,0,0], [0,1,0], [0,1,0]]])
        
        blue_Fuji_filtr = np.array([[[0,0,0], [0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,1]],
                                [[0,0,0], [0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,0]],
                                [[0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,1], [0,0,0]],
                                [[0,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
                                [[0,0,0], [0,0,0], [0,0,0], [0,0,1], [0,0,0], [0,0,0]]])


        # -----INTERPOLACJA LINIOWA----------------------------------
        # red = apply_filter(photo_array, red_Bayer_filtr)[:, :, 0]
        # green = apply_filter(photo_array, green_Bayer_filtr)[:, :, 1]
        # blue = apply_filter(photo_array, blue_Bayer_filtr)[:, :, 2]

        # red = interpolacja_liniowa_czerwonego(red)
        # green = interpolacja_liniowa_zielonego(green)
        # blue = interpolacja_liniowa_niebieskiego(blue)

        # red = (red * 255 / 256).astype(np.uint8)
        # green = (green * 255 / 256).astype(np.uint8)
        # blue = (blue * 255 / 256).astype(np.uint8)

        # image_tmp = np.dstack((red,green,blue))
        # final_image= Image.fromarray(image_tmp)
        # final_image.show(title='udało się')
        # -----------------------------------------------------------
        

        # ---------Interpolacja Fiji------------------------
        red = apply_filter(photo_array,red_Fuji_filtr)
        green = apply_filter(photo_array,green_Fuji_filtr)
        blue = apply_filter(photo_array,blue_Fuji_filtr)

        result_image = red.astype(np.uint8)
        plt.imshow(result_image)
        plt.show()
        result_image = green.astype(np.uint8)
        plt.imshow(result_image)
        plt.show()
        result_image = blue.astype(np.uint8)
        plt.imshow(result_image)
        plt.show()

        # red_wynikowa = red.copy()
        # green_wynikowa = green.copy()
        # blue_wynikowa = blue.copy()

        # interpolacja_fuji(red, red_wynikowa,"red")
        # interpolacja_fuji(green, green_wynikowa,"green")
        # interpolacja_fuji(blue, blue_wynikowa,"blue")

        # red_wynikowa = (red_wynikowa * 255 / 256).astype(np.uint8)
        # green_wynikowa = (green_wynikowa * 255 / 256).astype(np.uint8)
        # blue_wynikowa = (blue_wynikowa * 255 / 256).astype(np.uint8)

        # image_tmp = np.dstack((red_wynikowa,green_wynikowa,blue_wynikowa))
        # final_image= Image.fromarray(image_tmp)
        # final_image.show(title='udało się')

        # print("Fragment macierzy przed:")
        # print(green[:6, :6])
        # print("Fragment macierzy wynikowej:")
        # print(green_wynikowa[:10, :10])

    

if __name__ == '__main__':
    main()