// @formatter:off
// ESTE ARCHIVO ES EL ALGORITMO DEL PROBLEMA 3.1 (convolucion 2D con padding)

enum ModoPadding { ZeroPadding, ReplicatePadding, ReflectPadding }

static class Filtro
{
    // Por cada pixel: pone el kernel encima, multiplica y suma
    public static double[,] Convolucion(double[,] A, double[,] H, ModoPadding modo)
    {
        int M = A.GetLength(0), N = A.GetLength(1), K = H.GetLength(0);
        int radio = K / 2;
        double[,] R = new double[M, N];

        for (int i = 0; i < M; i++)            // filas (ciclo de afuera)
        {
            for (int j = 0; j < N; j++)        // columnas (ciclo de adentro)
            {
                double suma = 0;
                for (int ki = 0; ki < K; ki++)
                {
                    for (int kj = 0; kj < K; kj++)
                    {
                        int fila = i + ki - radio;
                        int col = j + kj - radio;
                        suma += Pixel(A, fila, col, modo) * H[K - 1 - ki, K - 1 - kj];
                    }
                }
                R[i, j] = suma;
            }
        }
        return R;
    }

    // Da el pixel. Si cae afuera de la imagen, aplica el padding
    static double Pixel(double[,] A, int f, int c, ModoPadding modo)
    {
        int M = A.GetLength(0), N = A.GetLength(1);
        if (f >= 0 && f < M && c >= 0 && c < N) return A[f, c];

        if (modo == ModoPadding.ZeroPadding) return 0;
        if (modo == ModoPadding.ReplicatePadding) return A[Limitar(f, M), Limitar(c, N)];
        return A[Reflejar(f, M), Reflejar(c, N)];
    }

    // Replicate: se queda en el borde
    static int Limitar(int i, int n)
    {
        if (i < 0) return 0;
        if (i >= n) return n - 1;
        return i;
    }

    // Reflect: rebota como espejo
    static int Reflejar(int i, int n)
    {
        if (i < 0) return -i;
        if (i >= n) return 2 * (n - 1) - i;
        return i;
    }
}
