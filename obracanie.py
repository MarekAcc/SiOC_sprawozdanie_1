import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

def obracanie(image_matrix, degrees):
    # Kąt obrotu w stopniach
    theta_degrees = -degrees

    # Przelicz kąt na radiany
    theta_radians = np.radians(theta_degrees)

    # Macierz obrotu
    rotation_matrix = np.array([[np.cos(theta_radians), -np.sin(theta_radians)],
                                [np.sin(theta_radians), np.cos(theta_radians)]])

    # Wymiary obrazu
    rows, cols = image_matrix.shape

    # Środek obrazu
    center_x, center_y = (cols - 1) / 2, (rows - 1) / 2

    # Przygotowanie pustej macierzy na obrócony obraz
    rotated_image = np.zeros_like(image_matrix)
    # Jesli obrot o wielokrotnosc 90 uzyj prostszego algorytmu
    if theta_degrees%90 == 0: 
        for i in range(rows):
            for j in range(cols):
                # Przesunięcie punktów do środka obrazu
                x_shifted, y_shifted = j - center_x, i - center_y
                # Obrót punktu
                x_rotated = rotation_matrix[0, 0] * x_shifted + rotation_matrix[0, 1] * y_shifted
                y_rotated = rotation_matrix[1, 0] * x_shifted + rotation_matrix[1, 1] * y_shifted
                x_rotated += center_x
                y_rotated += center_y

                x_rounded, y_rounded = int(round(x_rotated)), int(round(y_rotated))
                    # Sprawdzenie, czy punkt znajduje się w granicach obrazu
                if 0 <= x_rounded < cols and 0 <= y_rounded < rows:
                    # Przypisanie wartości z oryginalnego obrazu do nowego obrazu
                        rotated_image[i, j] = image_matrix[y_rounded, x_rounded]
                else:
                        rotated_image[i,j] = 0;
    #Jesli obrot o dowolną wartość kąta zastosuj interpolacje liniową z 4 punktow
    else:
        for i in range(rows):
            for j in range(cols):
                # Przesunięcie punktów do środka obrazu
                x_shifted, y_shifted = j - center_x, i - center_y
                # Obrót punktu
                x_rotated = rotation_matrix[0, 0] * x_shifted + rotation_matrix[0, 1] * y_shifted
                y_rotated = rotation_matrix[1, 0] * x_shifted + rotation_matrix[1, 1] * y_shifted
                x_rotated += center_x
                y_rotated += center_y
                ceil_x = math.ceil(x_rotated)
                floor_x = math.floor(x_rotated)
                ceil_y = math.ceil(y_rotated)
                floor_y = math.floor(y_rotated)
                if 0 <= x_rotated < cols-1 and 0 <= y_rotated < rows-1:
                    tmp1 = image_matrix[floor_y, floor_x] * abs(x_rotated - ceil_x) + image_matrix[ceil_y,floor_x] * abs(x_rotated - floor_x)
                    tmp2 = image_matrix[floor_y, ceil_x] * abs(x_rotated - ceil_x) + image_matrix[ceil_y,ceil_x] * abs(x_rotated - floor_x)
                    wartosc_koncowa = tmp1 * abs(y_rotated - ceil_y) + tmp2 * abs(y_rotated - floor_y)
                    rotated_image[i,j] = wartosc_koncowa
                else:
                    rotated_image[i,j] = 0;
    return rotated_image

def main():

    # Stopien obrotu
    degrees = 37
    num_rotations = 2
    # Wczytaj obraz
    image_path = 'kicia.jpeg'
    # image_path = 'LetItRoll.jpg'
    # image_path = 'Midland-On-The-Rocks-Cover.jpg'
    image = Image.open(image_path)
    image_array = np.array(image)
    red_channel = image_array[:, :, 0]
    blue_channel = image_array[:, :, 1]
    green_channel = image_array[:, :, 2]


    
    result_matrix_red = obracanie(red_channel, degrees)
    result_matrix_green = obracanie(green_channel, degrees)
    result_matrix_blue = obracanie(blue_channel, degrees)

    for i in range(num_rotations - 1):
        result_matrix_red = obracanie(result_matrix_red, degrees)
        result_matrix_green = obracanie(result_matrix_green, degrees)
        result_matrix_blue = obracanie(result_matrix_blue, degrees)

    plt.show()
    plt.subplot(1, 2, 1)
    plt.imshow(image_array)
    plt.title('Oryginalny obraz')

    rotated_image = np.dstack((result_matrix_red, result_matrix_green, result_matrix_blue))

    plt.subplot(1, 2, 2)
    plt.imshow(rotated_image)
    plt.title('Obrócony obraz')

    plt.show()
if __name__ == '__main__':
        main()