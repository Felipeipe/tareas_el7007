# Seccion 3


## 3.3
se ven unos puntos pequeños de valores altos donde están las activaciones del filtro.

los bordes tienen valores menores por el zero padding.

Los valores altos se deben a que el kernel no está normalizado (sus valores están entre 0 y 255), al igual que la imagen.

el valor máximo que se espera tener es de 255^2

## 3.4

Se elige este umbral ya que permite tener la mejor presición sin tener clasificaciones erróneas.

## 3.6
se tuvieron 7 falsos positivos, con 11 detecciones. Sin falsos negativos.
quizás se mejoraría si es que se usa un umbral de binarización distinto.

No es robusto frente al ruido ya que el valor de las activaciones baja bastante y provoca que se homogeneicen las lecturas en la matriz de activaciones, lo que provoca que sea más dificil diferenciar entre otras.

Se podría usar varios filtros para poder ir descartando las lecturas incorrectas.

## 4.1
Se pudo obtener la cuenta correctamente.
una de las opciones sería obtener el número de la carta si es que se tienen distintas pintas. esta solucion tambien serviría si es que las cartas están parcialmente ocluidas.
Se podría tambien analizar el ángulo de la carta si es que no está ocluida y tener una lookup table que permita utilizar un filtro acorde al ángulo de la carta
