// Problema 3.2 (Dart 3.0+): Pure Coordinate Transformation Matrix Engine
//
// Convención: vectores columna, p' = M · p, matrices 4x4 row-major.
// Componer [A, B, C] "en orden de aplicación" equivale a C · B · A.
//
// ANÁLISIS DE COMPLEJIDAD (matriz fija n = 4)
//   identity / translation / rotationZ : tiempo O(1), espacio O(1) (16 doubles)
//   multiply (A·B)     : n^3 = 64 multiplicaciones, 48 sumas -> O(1) / O(1)
//   transformPoint     : 16 mult. + 12 sumas + 3 divisiones   -> O(1) / O(1)
//   compose(k matrices): (k-1) multiplicaciones                -> O(k) / O(1) extra
//   Generalizado a n x n: multiply O(n^3), transformPoint O(n^2).
//
// PUREZA / THREAD-SAFETY: no hay estado global ni mutación. Matrix4 es inmutable
// (buffer privado copiado en el constructor, nunca expuesto); cada función
// devuelve una instancia nueva, así que es segura entre isolates.

import 'dart:math' as math;
import 'dart:typed_data';

/// Punto 3D inmutable.
final class Point3D {
  final double x, y, z;
  const Point3D(this.x, this.y, this.z);

  /// Indica si las tres coordenadas contienen valores finitos.
  bool get isFinite => x.isFinite && y.isFinite && z.isFinite;

  /// Compara este punto con otro usando una tolerancia para errores decimales.
  bool approxEquals(Point3D o, [double eps = 1e-9]) =>
      // La diferencia absoluta de cada coordenada debe quedar dentro del margen.
      (x - o.x).abs() <= eps && (y - o.y).abs() <= eps && (z - o.z).abs() <= eps;

  /// Comprueba si otro objeto contiene exactamente las mismas coordenadas.
  @override
  bool operator ==(Object o) => o is Point3D && x == o.x && y == o.y && z == o.z;

  /// Genera un valor hash a partir de las coordenadas del punto.
  @override
  int get hashCode => Object.hash(x, y, z);

  /// Presenta las coordenadas con cuatro decimales para facilitar su lectura.
  @override
  String toString() => 'Point3D(${x.toStringAsFixed(4)}, '
      '${y.toStringAsFixed(4)}, ${z.toStringAsFixed(4)})';
}

/// Matriz homogénea 4x4 inmutable (Float64List row-major privado).
final class Matrix4 {
  final Float64List _d;
  Matrix4._(this._d);

  /// Construye la matriz a partir de cuatro filas con cuatro valores cada una.
  /// El constructor valida la forma y los valores, y luego copia los datos.
  factory Matrix4.fromRows(List<List<double>> rows) {
    // La matriz debe tener exactamente cuatro filas y cuatro columnas.
    if (rows.length != 4 || rows.any((r) => r.length != 4)) {
      throw ArgumentError('La matriz debe ser exactamente 4x4.');
    }
    final d = Float64List(16);
    // Cada elemento se valida y se almacena en una lista plana por filas.
    for (var r = 0; r < 4; r++) {
      for (var c = 0; c < 4; c++) {
        final v = rows[r][c];
        if (!v.isFinite) throw ArgumentError('Elemento no finito en [$r][$c].');
        // r * 4 salta las filas anteriores; c selecciona la columna actual.
        d[r * 4 + c] = v;
      }
    }
    return Matrix4._(d);
  }

  /// Crea la matriz identidad con unos en la diagonal y ceros en las demás posiciones.
  factory Matrix4.identity() {
    // Float64List inicia todos los elementos en cero, como requiere la identidad.
    final d = Float64List(16);
    for (var i = 0; i < 4; i++) {
      // En un arreglo de 4 columnas, cada posición diagonal avanza cinco índices.
      d[i * 5] = 1.0; // diagonal: índices 0, 5, 10, 15
    }
    return Matrix4._(d);
  }

  /// Devuelve el elemento ubicado en la fila [row] y la columna [col].
  double at(int row, int col) {
    // Los índices se validan antes de convertir la posición a un índice lineal.
    RangeError.checkValidIndex(row, _d, 'row', 4);
    RangeError.checkValidIndex(col, _d, 'col', 4);
    // El arreglo es row-major, por eso la fila aporta cuatro posiciones por elemento.
    return _d[row * 4 + col];
  }

  /// Compara todos los elementos de ambas matrices con una tolerancia numérica.
  bool approxEquals(Matrix4 o, [double eps = 1e-9]) {
    for (var i = 0; i < 16; i++) {
      // La comparación tolera pequeñas diferencias de redondeo en cada elemento.
      if ((_d[i] - o._d[i]).abs() > eps) return false;
    }
    return true;
  }

  /// Convierte la matriz en texto, conservando visualmente sus cuatro filas.
  @override
  String toString() {
    final sb = StringBuffer();
    for (var r = 0; r < 4; r++) {
      // Cada iteración agrega una fila completa al texto resultante.
      sb.write('[');
      for (var c = 0; c < 4; c++) {
        // Los valores se redondean a cuatro decimales y se alinean en nueve espacios.
        sb.write(_d[r * 4 + c].toStringAsFixed(4).padLeft(9));
      }
      sb.writeln(' ]');
    }
    return sb.toString();
  }
}

// ---------------------------------------------------------------- funciones puras

/// Construye una matriz de traslación a partir de los desplazamientos recibidos.
Matrix4 translation(double tx, double ty, double tz) {
  // La función rechaza desplazamientos NaN o infinitos antes de crear la matriz.
  if (!(tx.isFinite && ty.isFinite && tz.isFinite)) {
    throw ArgumentError('Los desplazamientos deben ser finitos.');
  }
  // La última columna incorpora los desplazamientos en los ejes x, y y z.
  // Los unos de la diagonal conservan las coordenadas originales antes de sumar esos desplazamientos.
  return Matrix4.fromRows([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1],
  ]);
}

/// Construye una matriz que rota alrededor del eje Z el ángulo [radians].
Matrix4 rotationZ(double radians) {
  // El ángulo debe ser finito para que seno y coseno produzcan valores válidos.
  if (!radians.isFinite) throw ArgumentError('El ángulo debe ser finito.');
  // Estas funciones trigonométricas reciben el ángulo en radianes.
  final c = math.cos(radians), s = math.sin(radians);
  // Los signos de seno orientan el giro; las filas tercera y cuarta conservan z y w.
  return Matrix4.fromRows([
    [c, -s, 0, 0],
    [s, c, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]);
}

/// Multiplica las matrices [a] y [b], y devuelve una matriz nueva con el producto a · b.
Matrix4 multiply(Matrix4 a, Matrix4 b) {
  final d = Float64List(16);
  // Las dos primeras iteraciones seleccionan una posición de la matriz resultado.
  for (var i = 0; i < 4; i++) {
    for (var j = 0; j < 4; j++) {
      var sum = 0.0;
      // La iteración interna calcula el producto punto de una fila por una columna.
      for (var k = 0; k < 4; k++) {
        // a[i,k] se multiplica por b[k,j] y se acumula en el elemento resultado [i,j].
        sum += a._d[i * 4 + k] * b._d[k * 4 + j];
      }
      // El resultado también se guarda en formato row-major.
      d[i * 4 + j] = sum;
    }
  }
  return Matrix4._(d);
}

/// Combina matrices en el orden en que se aplican; [A, B, C] produce C · B · A.
Matrix4 compose(List<Matrix4> inApplicationOrder) {
  // La identidad permite iniciar la acumulación y también representa una lista vacía.
  var acc = Matrix4.identity();
  // Cada transformación se incorpora a la izquierda para respetar el orden de aplicación.
  for (final m in inApplicationOrder) {
    acc = multiply(m, acc);
  }
  return acc;
}

/// Aplica la matriz al punto y normaliza el resultado usando la coordenada homogénea w.
Point3D transformPoint(Matrix4 m, Point3D p, {double wEps = 1e-12}) {
  // Las coordenadas de entrada se validan antes de realizar las multiplicaciones.
  if (!p.isFinite) throw ArgumentError('Coordenadas no finitas: $p');
  final d = m._d;
  // Cada resultado corresponde a una fila de la matriz multiplicada por [x, y, z, 1].
  // En cada suma, los tres primeros productos transforman las coordenadas y el último agrega la traslación.
  final x = d[0] * p.x + d[1] * p.y + d[2] * p.z + d[3];
  final y = d[4] * p.x + d[5] * p.y + d[6] * p.z + d[7];
  final z = d[8] * p.x + d[9] * p.y + d[10] * p.z + d[11];
  // La cuarta fila calcula w; esta coordenada determina la normalización homogénea.
  final w = d[12] * p.x + d[13] * p.y + d[14] * p.z + d[15];
  // Cuando w vale uno, las coordenadas ya están en forma cartesiana.
  if (w == 1.0) return Point3D(x, y, z); // caso afín: sin división
  // Un w cercano a cero representa un punto en el infinito y no puede normalizarse.
  if (w.abs() < wEps) {
    throw StateError('w ≈ 0: punto en el infinito, no se puede normalizar.');
  }
  // Para los demás casos, cada coordenada se divide entre w.
  return Point3D(x / w, y / w, z / w);
}

// ---------------------------------------------------------------- step-by-step trace

/// Imprime un encabezado para separar visualmente cada etapa de la demostración.
void _title(String t) => print('\n>>> $t');

/// Muestra cómo se construyen las matrices, cómo se componen y cómo se transforma un punto.
void demoSteps() {
  // La matriz identidad sirve como referencia porque conserva el punto sin modificarlo.
  print('=============== STEP-BY-STEP MATRIX DEMO ===============');
  _title('1) Identity: the starting point (does nothing)');
  print(Matrix4.identity());

  _title('2) Translation T(5, 2, 1): last column stores the offset');
  final t = translation(5, 2, 1);
  print(t);

  // El ángulo se expresa en radianes antes de construir la matriz de rotación.
  final theta = math.pi / 4;
  _title('3) Rotation Rz(45°): cos = ${math.cos(theta).toStringAsFixed(4)}, '
      'sin = ${math.sin(theta).toStringAsFixed(4)}');
  final r = rotationZ(theta);
  print(r);

  // Las matrices se acumulan en orden inverso al producto para aplicar primero la rotación.
  _title('4) Composition M = T · Rz (Rz is applied first)');
  var acc = Matrix4.identity();
  var step = 1;
  for (final (name, m) in [('Rz(45°)', r), ('T(5,2,1)', t)]) {
    acc = multiply(m, acc);
    print('Step $step: acc = $name · acc');
    print(acc);
    step++;
  }

  // El punto se representa como vector columna y se calcula cada fila por separado.
  _title('5) Point (1,1,1) through T(5,2,1): row × column products');
  const p = Point3D(1, 1, 1);
  // El cuarto componente vale uno para representar el punto en coordenadas homogéneas.
  final v = [p.x, p.y, p.z, 1.0];
  const names = ['x′', 'y′', 'z′', 'w′'];
  for (var row = 0; row < 4; row++) {
    // La suma comienza en cero y reúne los cuatro productos de la fila actual.
    var sum = 0.0;
    final terms = <String>[];
    // La columna recorre las cuatro componentes del vector homogéneo.
    for (var c = 0; c < 4; c++) {
      // Se acumula el producto fila-columna y se guarda su representación para mostrarla.
      sum += t.at(row, c) * v[c];
      terms.add('${t.at(row, c).toStringAsFixed(1)}·${v[c].toStringAsFixed(1)}');
    }
    print('${names[row]} = ${terms.join(' + ')} = ${sum.toStringAsFixed(4)}');
  }
  // El resultado final se normaliza mediante transformPoint y se imprime.
  print('Normalized (÷ w′) -> ${transformPoint(t, p)}');
  print('========================================================\n');
}

// ---------------------------------------------------------------- harness

// Los contadores reúnen los resultados de todas las comprobaciones ejecutadas.
int _passed = 0, _failed = 0;

/// Registra una comprobación y muestra en consola si el resultado fue correcto.
void check(String name, bool ok) {
  ok ? _passed++ : _failed++;
  print('${ok ? "[PASS]" : "[FAIL]"} $name');
}

/// Devuelve true cuando la función recibida lanza una excepción del tipo esperado.
bool throwsA<T extends Object>(void Function() f) {
  try {
    f();
  } on T {
    return true;
  }
  return false;
}

/// Ejecuta la demostración y las pruebas funcionales, de robustez y rendimiento.
void main() {
  // La demostración introductoria explica primero las operaciones principales.
  demoSteps();

  // Estos valores se reutilizan en varias pruebas para evitar repetir su creación.
  const origin = Point3D(0, 0, 0);
  final px = const Point3D(1, 0, 0);
  final halfPi = math.pi / 2;

  // Las primeras comprobaciones confirman el comportamiento básico de las transformaciones.
  print('=== Casos funcionales ===');
  check('Identidad no altera el punto',
      transformPoint(Matrix4.identity(), const Point3D(3, -2, 7)) == const Point3D(3, -2, 7));
  check('Traslación (1,2,3) sobre origen',
      transformPoint(translation(1, 2, 3), origin) == const Point3D(1, 2, 3));
  check('Rz(90°) lleva (1,0,0) a (0,1,0)',
      transformPoint(rotationZ(halfPi), px).approxEquals(const Point3D(0, 1, 0)));
  check('Rz no altera la coordenada z',
      transformPoint(rotationZ(1.234), const Point3D(1, 1, 5)).z == 5.0);

  // Las siguientes pruebas comparan el resultado cuando cambia el orden de aplicación.
  print('\n=== Composición (orden importa) ===');
  final rt = compose([rotationZ(halfPi), translation(5, 0, 0)]); // rota, luego traslada
  final tr = compose([translation(5, 0, 0), rotationZ(halfPi)]); // traslada, luego rota
  check('Rotar y luego trasladar: (1,0,0) -> (5,1,0)',
      transformPoint(rt, px).approxEquals(const Point3D(5, 1, 0)));
  check('Trasladar y luego rotar: (1,0,0) -> (0,6,0)',
      transformPoint(tr, px).approxEquals(const Point3D(0, 6, 0)));
  check('La multiplicación NO es conmutativa', !rt.approxEquals(tr));
  final a = rotationZ(0.3), b = translation(1, 2, 3), c = rotationZ(-1.1);
  check('Asociatividad (AB)C = A(BC)',
      multiply(multiply(a, b), c).approxEquals(multiply(a, multiply(b, c))));
  check('R(θ)·R(-θ) = I', multiply(rotationZ(0.7), rotationZ(-0.7)).approxEquals(Matrix4.identity()));
  check('R(2π) ≈ I', rotationZ(2 * math.pi).approxEquals(Matrix4.identity()));
  check('compose([]) = identidad', compose([]).approxEquals(Matrix4.identity()));
  check('compose = multiplicación manual',
      compose([a, b, c]).approxEquals(multiply(c, multiply(b, a))));

  // Estas matrices producen valores w distintos de uno para verificar la normalización.
  print('\n=== Normalización homogénea (w) ===');
  final scaleW = Matrix4.fromRows([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 2],
  ]); // w = 2 -> las coordenadas se dividen entre 2
  check('w=2: (2,4,6) -> (1,2,3)',
      transformPoint(scaleW, const Point3D(2, 4, 6)).approxEquals(const Point3D(1, 2, 3)));
  final persp = Matrix4.fromRows([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
  ]); // w = z (proyección en perspectiva simple)
  check('w=z: (2,4,2) -> (1,2,1)',
      transformPoint(persp, const Point3D(2, 4, 2)).approxEquals(const Point3D(1, 2, 1)));

  // Las pruebas de robustez confirman que las entradas inválidas generen excepciones.
  print('\n=== Casos límite / robustez ===');
  check('w=0 lanza StateError', throwsA<StateError>(() => transformPoint(persp, const Point3D(1, 1, 0))));
  check('Punto NaN lanza ArgumentError',
      throwsA<ArgumentError>(() => transformPoint(Matrix4.identity(), Point3D(double.nan, 0, 0))));
  check('Punto infinito lanza ArgumentError',
      throwsA<ArgumentError>(() => transformPoint(Matrix4.identity(), Point3D(double.infinity, 0, 0))));
  check('Ángulo NaN lanza ArgumentError', throwsA<ArgumentError>(() => rotationZ(double.nan)));
  check('Traslación infinita lanza ArgumentError',
      throwsA<ArgumentError>(() => translation(double.infinity, 0, 0)));
  check('fromRows 3x3 lanza ArgumentError',
      throwsA<ArgumentError>(() => Matrix4.fromRows([[1, 0, 0], [0, 1, 0], [0, 0, 1]])));
  check('Índice fuera de rango lanza RangeError',
      throwsA<RangeError>(() => Matrix4.identity().at(4, 0)));
  check('Ángulo grande (1e6 rad) sigue ortonormal',
      () {
        final r = rotationZ(1e6);
        // El determinante 2x2 de la parte XY debe aproximarse a uno en una rotación.
        final det = r.at(0, 0) * r.at(1, 1) - r.at(0, 1) * r.at(1, 0);
        // La diferencia respecto a uno debe ser menor que la tolerancia establecida.
        return (det - 1).abs() < 1e-9;
      }());

  // Este bloque comprueba que las operaciones no modifiquen sus matrices de entrada.
  print('\n=== Pureza e inmutabilidad ===');
  final m1 = translation(1, 1, 1);
  final snapshot = m1.toString();
  multiply(m1, rotationZ(0.5));
  transformPoint(m1, px);
  check('Las operaciones no mutan las matrices de entrada', m1.toString() == snapshot);
  final rows = [
    [1.0, 0.0, 0.0, 9.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ];
  final copyM = Matrix4.fromRows(rows);
  // Se modifica la lista original para comprobar que la matriz conserva su copia independiente.
  rows[0][3] = 100.0; // mutar la fuente no debe afectar a la matriz
  check('fromRows copia defensivamente los datos', copyM.at(0, 3) == 9.0);
  check('Determinismo: mismas entradas, mismo resultado',
      transformPoint(rt, px) == transformPoint(rt, px));

  // El benchmark aplica repetidamente la misma transformación y mide el tiempo transcurrido.
  print('\n=== Benchmark (rendimiento) ===');
  const n = 2000000;
  final m = compose([rotationZ(0.01), translation(0.1, 0.2, 0.3)]);
  var p = const Point3D(1, 2, 3);
  // El cronómetro comienza antes del ciclo y se detiene cuando terminan las iteraciones.
  final sw = Stopwatch()..start();
  for (var i = 0; i < n; i++) {
    // Cada iteración transforma el resultado anterior y lo usa como nuevo punto de entrada.
    p = transformPoint(m, p);
  }
  sw.stop();
  // Los microsegundos se convierten a segundos para mostrar una medida legible.
  final secs = sw.elapsedMicroseconds / 1e6;
  print('$n transformaciones en ${secs.toStringAsFixed(3)} s '
      '(~${(n / secs / 1e6).toStringAsFixed(1)} M/s) -> $p');

  // Al final se informa el total de pruebas y se muestra una matriz compuesta de ejemplo.
  print('\nResumen: $_passed OK, $_failed fallos.');
  print('\nMatriz compuesta R(90°) luego T(5,0,0):\n$rt');
}