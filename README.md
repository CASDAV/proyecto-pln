# proyecto-pln

Proyecto de la asignatura de Procesamiento de Lenguaje Natural, Maestría en
Ingeniería de Sistemas y Computación, Pontificia Universidad Javeriana.

Análisis de un corpus de prensa colombiana (1980–2011) proveniente de
[yabramuvdi/NoticiasColombia](https://huggingface.co/datasets/yabramuvdi/NoticiasColombia),
publicado bajo licencia ODC-BY.

## Contenido

```
.
├── notebooks/
│   ├── 01-corpus.ipynb        Exploración del corpus y su metadata
│   ├── 02-textometria.ipynb   Limpieza, tokenización, frecuencias y figuras
│   └── utils.py               Carga del corpus y normalización compartidas
├── results/                   Salidas: JSON con cifras, CSV con tablas, PNG a 300 dpi
├── data/                      Corpus descargado (no versionado)
├── pyproject.toml             Dependencias
├── uv.lock                    Versiones exactas (solo uv)
└── .python-version            Python 3.12
```

El corpus **no está en el repositorio**: pesa 126 MB y los notebooks lo
descargan automáticamente desde un GitHub Release de este mismo repo,
verificando su SHA-256.

## Requisitos

Python 3.12. Versiones más recientes pueden no ser compatibles con spaCy 3.8.

## Montaje del entorno

### Opción A — con uv (recomendada)

Reproduce exactamente las versiones con las que se generaron los resultados.

```bash
git clone https://github.com/CASDAV/proyecto-pln.git
cd proyecto-pln
uv sync
```

`uv sync` instala Python 3.12 si no lo tienes, crea `.venv` y aplica
`uv.lock`.

### Opción B — sin uv (pip + venv)

Funciona igual, pero resuelve las versiones desde cero en lugar de usar el
lock, así que podrían diferir de las originales.

```bash
git clone https://github.com/CASDAV/proyecto-pln.git
cd proyecto-pln

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pandas pyarrow spacy nltk matplotlib wordcloud ipykernel
```


Si no tienes Python 3.12 disponible, instálalo con
[pyenv](https://github.com/pyenv/pyenv) o desde python.org antes de crear el
entorno.

### Modelo de spaCy

La tokenización usa `spacy.blank("es")`, que aplica reglas y no carga ningún
modelo estadístico.

### Datos de NLTK

No hay que descargarlos a mano: la función `setup()` de `utils.py`, que
invocan ambos notebooks en su primera celda, se encarga de traer el corpus de
stopwords si falta.

## Ejecución

Abre los notebooks en VS Code o Jupyter y selecciona el kernel de `.venv`.

```bash
# VS Code
code .

# o Jupyter
uv run jupyter lab            # con uv
jupyter lab                   # con pip, dentro del venv activado
```

Ejecuta en orden:

1. `01-corpus.ipynb` — descarga y verifica el corpus, describe su estructura,
   duplicados, distribución por medio y por año.
2. `02-textometria.ipynb` — normaliza el texto, tokeniza, calcula frecuencias
   y genera las figuras.

Ambos corren de principio a fin con **Restart & Run All**. La primera
ejecución descarga el corpus (126 MB); las siguientes lo reutilizan desde
`data/`.

La celda más lenta es la tokenización del corpus completo: alrededor de un
minuto.

### En Google Colab

Los notebooks detectan el entorno y se adaptan solos. La llamada a `setup()`
de la primera celda instala allí las dependencias que Colab no trae, y el
corpus se carga en memoria sin escribir a disco. Las salidas van a
`/content/results`.

Tres cosas a tener en cuenta:

- `utils.py` debe estar en el `sys.path`. El directorio de trabajo en Colab
  es `/content`, no la carpeta del notebook, así que hay que añadir la ruta
  de la carpeta compartida o copiar el archivo a `/content`.
- Instalar spaCy puede actualizar numpy y obligar a reiniciar el runtime. Si
  Colab lo pide, reinicia y vuelve a ejecutar desde la primera celda.
- `/content` se borra al reciclar la sesión, así que las figuras y los JSON
  se pierden salvo que montes Drive.

## Resultados

Todo lo que produce `02-textometria.ipynb` queda en `results/`:

| Archivo | Contenido |
|---|---|
| `textometria_stats.json` | Medidas del corpus y cifras citadas en el informe |
| `top20_con_sw.csv` | Veinte formas más frecuentes, sin filtrar |
| `top20_sin_sw_nltk.csv` | Ídem, excluyendo palabras vacías de NLTK |
| `top20_sin_sw_spacy.csv` | Ídem, excluyendo palabras vacías de spaCy |
| `top_por_fuente.csv` | Comparación por medio de comunicación |
| `top_por_decada.csv` | Comparación por década |
| `top_por_bloque.csv` | Comparación por períodos de volumen equivalente |
| `*.png` | Gráficos de barras y nubes de palabras, 300 dpi |

Las cifras del informe se leen de estos archivos, no se transcriben a mano.

## Notas para quien contribuya

Los notebooks se versionan **sin salidas**, mediante un filtro de git. Tras
clonar hay que activarlo una vez, porque los filtros no viajan con el repo:

```bash
uv run nbstripout --install --attributes .gitattributes   # con uv
nbstripout --install --attributes .gitattributes          # con pip
```

Sin ese paso, cada commit arrastrará las imágenes y los diffs quedarán
ilegibles.

Linting con ruff, que también revisa los `.ipynb`:

```bash
uv run ruff check .
```