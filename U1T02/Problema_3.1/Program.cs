// @formatter:off
// PROBLEMA 3.1: lee las imagenes de "entrada", aplica kernels con los 3 paddings y guarda en "salida"
using System.Diagnostics;
using System.Drawing;

// 1. Buscar las carpetas entrada y salida (estan junto a este proyecto)
DirectoryInfo dir = new DirectoryInfo(Directory.GetCurrentDirectory());
while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, "entrada")))
    dir = dir.Parent;
if (dir == null)
{
    Console.WriteLine("No encontre la carpeta 'entrada'.");
    return;
}
string carpetaEntrada = Path.Combine(dir.FullName, "entrada");
string carpetaSalida = Path.Combine(dir.FullName, "salida");
Directory.CreateDirectory(carpetaSalida);

// 2. LOS KERNELS H (K x K): blur, nitidez y bordes (silueta)
int K = 21;                                          // blur: todos valen 1/(K*K)
double[,] Hblur = new double[K, K];
for (int i = 0; i < K; i++)
    for (int j = 0; j < K; j++)
        Hblur[i, j] = 1.0 / (K * K);

double[,] Hnitidez = new double[,] { {  0, -1,  0 },  // nitidez (sharpen): realza los detalles
                                     { -1,  5, -1 },
                                     {  0, -1,  0 } };

double[,] Hbordes = new double[,] { { -1, -1, -1 },  // bordes (silueta): 8 al centro, -1 alrededor
                                    { -1,  8, -1 },
                                    { -1, -1, -1 } };

// 3. Por cada imagen: original + un resultado por cada padding (cada uno con un filtro distinto)
string[] extensiones = { ".png", ".jpg", ".jpeg", ".bmp" };
string[] archivos = Directory.GetFiles(carpetaEntrada)
    .Where(f => extensiones.Contains(Path.GetExtension(f).ToLower()))
    .ToArray();

if (archivos.Length == 0)
{
    Console.WriteLine("No hay imagenes en: " + carpetaEntrada);
    return;
}

foreach (string archivo in archivos)
{
    string nombre = Path.GetFileNameWithoutExtension(archivo);
    double[,] A = ImagenATabla(archivo);             // la matriz A (M x N)

    Guardar(A, carpetaSalida, nombre + "_0_original.png");
    Guardar(Filtro.Convolucion(A, Hblur,    ModoPadding.ZeroPadding),      carpetaSalida, nombre + "_1_blur_ZeroPadding.png");
    Guardar(Filtro.Convolucion(A, Hnitidez, ModoPadding.ReplicatePadding), carpetaSalida, nombre + "_2_nitidez_ReplicatePadding.png");
    Guardar(Filtro.Convolucion(A, Hbordes,  ModoPadding.ReflectPadding),   carpetaSalida, nombre + "_3_bordes_ReflectPadding.png");

    // HTML interactivo que reconstruye ESTA imagen paso a paso (usa plantilla_reconstruccion.html)
    string plantilla = Path.Combine(dir.FullName, "plantilla_reconstruccion.html");
    if (File.Exists(plantilla))
    {
        string ext = Path.GetExtension(archivo).ToLower();
        string mime = ext == ".png" ? "image/png" : ext == ".bmp" ? "image/bmp" : "image/jpeg";
        string src = "data:" + mime + ";base64," + Convert.ToBase64String(File.ReadAllBytes(archivo));
        File.WriteAllText(Path.Combine(carpetaSalida, nombre + "_reconstruccion.html"),
                          File.ReadAllText(plantilla).Replace("@@SRC@@", src));
    }
    Console.WriteLine("Listo: " + nombre);
}

Console.WriteLine("Resultados en: " + carpetaSalida);
Process.Start(new ProcessStartInfo(carpetaSalida) { UseShellExecute = true });   // abre la carpeta


// ---------- Funciones de ayuda (solo leer y guardar imagenes) ----------

// Imagen -> tabla de numeros (0 = negro, 255 = blanco). Si es muy ancha, la achica a 300
static double[,] ImagenATabla(string ruta)
{
    using Bitmap original = new Bitmap(ruta);
    int ancho = Math.Min(original.Width, 300);
    int alto = original.Height * ancho / original.Width;
    using Bitmap bmp = new Bitmap(original, ancho, alto);

    double[,] t = new double[alto, ancho];
    for (int y = 0; y < alto; y++)
        for (int x = 0; x < ancho; x++)
        {
            Color c = bmp.GetPixel(x, y);
            t[y, x] = 0.299 * c.R + 0.587 * c.G + 0.114 * c.B;   // a gris
        }
    return t;
}

// Tabla de numeros -> imagen PNG (los numeros se limitan a 0..255)
static void Guardar(double[,] t, string carpeta, string nombreArchivo)
{
    int alto = t.GetLength(0), ancho = t.GetLength(1);
    using Bitmap bmp = new Bitmap(ancho, alto);
    for (int y = 0; y < alto; y++)
        for (int x = 0; x < ancho; x++)
        {
            int v = (int)Math.Round(Math.Clamp(t[y, x], 0, 255));
            bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
        }
    bmp.Save(Path.Combine(carpeta, nombreArchivo));
}
