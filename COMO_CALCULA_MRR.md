# 🔴 CÓMO SE CALCULA LA RELEVANCIA DE CADA DOCUMENTO EN MRR

## Resumen visual del flujo

```
┌─────────────────────────────────────────────────────────────┐
│        PREGUNTA: ¿Cómo tratar tizón tardío?                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────┐
        │   DOCUMENTOS RECUPERADOS (3)     │
        └──────────────────────────────────┘
         │
         ├─ Doc 1: "Tizón causado por Phytophthora..."
         ├─ Doc 2: "Tomates rojos son ricos en..."
         └─ Doc 3: "Fungicidas como clorotalonil..."
                    ↓
        ┌──────────────────────────────────┐
        │  CALCULAR RELEVANCIA C/DOCUMENTO │
        └──────────────────────────────────┘
         │
         ├─ Doc 1: RELEVANCIA = 0.737 ✅ RELEVANTE
         ├─ Doc 2: RELEVANCIA = 0.545 ❌ NO RELEVANTE
         └─ Doc 3: RELEVANCIA = 0.620 ❌ NO RELEVANTE
                    ↓
        ┌──────────────────────────────────┐
        │    CALCULAR MRR                  │
        │  MRR = 1 / posición relevante    │
        │  MRR = 1 / 1 = 1.0 (Perfecto!)   │
        └──────────────────────────────────┘
```

---

## Paso 1: CÁLCULO DE RELEVANCIA POR DOCUMENTO

Cada documento se califica con **3 factores combinados**:

### Formula:

```
RELEVANCIA = (Similitud × 60%) + (Keywords × 25%) + (Metadatos × 15%)
```

### Ejemplo real - Documento 1:

```
📄 DOCUMENTO 1
Contenido: "El tizón tardío causado por Phytophthora infestans
           es una enfermedad fúngica..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FACTOR 1: SIMILITUD SEMÁNTICA (60% peso)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ├─ Distancia embedding: 0.15
   ├─ Conversión: Similitud = 1/(1+0.15) = 0.870
   ├─ Peso aplicado: 0.870 × 60% = 0.522
   └─ Interpretación: El contenido es MUY similar a la pregunta

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FACTOR 2: COINCIDENCIA DE PALABRAS CLAVE (25% peso)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Palabras de la pregunta: ["tratar", "tizón", "tardío", "tomate"]

   Búsqueda en documento:
   ├─ "tratar" → ❌ NO aparece
   ├─ "tizón" → ✅ Aparece (2 veces)
   ├─ "tardío" → ✅ Aparece (1 vez)
   └─ "tomate" → ❌ NO aparece

   Coincidencias: 2/4 = 50%
   Score: 0.50 × 25% = 0.125
   └─ Interpretación: Documento habla de tizón pero no de tomate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FACTOR 3: METADATOS Y CONFIABILIDAD (15% peso)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Archivo: "documento1.pdf"
   ├─ ¿Es documento compilado? NO → 0.6 puntos
   ├─ ¿Es documento maestro? NO → 0.6 puntos
   └─ Score metadatos: 0.6 × 15% = 0.090

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL RELEVANCIA DEL DOCUMENTO 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   = 0.522 + 0.125 + 0.090 = 0.737 ✅ RELEVANTE

   ¿Es >= 0.7 (umbral)? SÍ → Este es el PRIMER DOCUMENTO RELEVANTE
```

---

## Paso 2: CÁLCULO DE MRR

```
┌─────────────────────────────────────────────────────────────┐
│  MRR = MEAN RECIPROCAL RANK                                 │
│  = 1 / posición del PRIMER documento con relevancia >= 0.7  │
└─────────────────────────────────────────────────────────────┘

En nuestro ejemplo:
   Documento 1: Relevancia = 0.737 >= 0.7 ✅ RELEVANTE
   Posición: 1

   MRR = 1 / 1 = 1.0  ⭐ PERFECTO!

¿Qué significa MRR = 1.0?
→ El primer documento recuperado ES relevante
→ El usuario ve la respuesta correcta INMEDIATAMENTE
```

### Otros escenarios posibles:

```
Escenario A: Primer doc relevante en posición 3
   Documentos: [NO relevante, NO relevante, SÍ relevante, ...]
   MRR = 1/3 = 0.333
   → Usuario debe desplazarse para encontrar respuesta

Escenario B: Primer doc relevante en posición 10
   Documentos: [NO, NO, NO, NO, NO, NO, NO, NO, NO, SÍ, ...]
   MRR = 1/10 = 0.1
   → Muy malo, usuario probablemente se rinde

Escenario C: NINGUNO es relevante
   Documentos: [NO, NO, NO, NO, NO, ...]
   MRR = 0.0
   → Búsqueda fracasó completamente
```

---

## Paso 3: CÁLCULO DE NDCG

```
┌─────────────────────────────────────────────────────────────┐
│  NDCG = Normalized Discounted Cumulative Gain               │
│                                                              │
│  A diferencia de MRR que solo ve el PRIMERO,                │
│  NDCG ve TODOS los documentos y su orden                    │
└─────────────────────────────────────────────────────────────┘

Formula:
   DCG = sum( relevancia_i / log2(i+1) )  para i = 1 a k
   IDCG = DCG del ranking IDEAL (mejor posible)
   NDCG = DCG / IDCG

En nuestro ejemplo:

   Relevancias reales:  [0.737, 0.545, 0.620]
   Relevancias ideales: [0.737, 0.620, 0.545]  (ordenadas descendente)

   DCG real  = 0.737/log2(2) + 0.545/log2(3) + 0.620/log2(4)
             = 0.737/1 + 0.545/1.585 + 0.620/2
             = 0.737 + 0.344 + 0.310
             = 1.391

   DCG ideal = 0.737/log2(2) + 0.620/log2(3) + 0.545/log2(4)
             = 0.737 + 0.391 + 0.273
             = 1.401

   NDCG = 1.391 / 1.401 = 0.993

   ✅ NDCG = 0.993 es EXCELENTE (99.3% del ideal)
```

---

## Comparación MRR vs NDCG

| Métrica  | ¿Qué mide?                    | Ventaja                  | Desventaja         |
| -------- | ----------------------------- | ------------------------ | ------------------ |
| **MRR**  | Posición del primer relevante | Fácil de entender        | Solo ve el primero |
| **NDCG** | Calidad del ranking completo  | Más justo con el usuario | Más complejo       |

### Ejemplo que muestra la diferencia:

```
Ranking A:
   1. Relevante (0.9)  ✅
   2. NO (0.1)         ❌
   3. NO (0.1)         ❌

   MRR = 1/1 = 1.0    (Excelente)
   NDCG ≈ 0.5         (Pobre, ranking después es malo)

Ranking B:
   1. NO (0.2)        ❌
   2. Relevante (0.9) ✅
   3. Muy relevante (0.95) ✅✅

   MRR = 1/2 = 0.5    (Medio)
   NDCG ≈ 0.95        (Excelente, ranking posterior es bueno)
```

---

## 📊 RESUMEN DEL CÓDIGO

El archivo `rag_evaluator_fixed.py` implementa:

```python
def _calculate_document_relevance(query, doc, similarity):
    """🔴 AQUÍ SE CALCULA LA RELEVANCIA DE UN DOCUMENTO"""

    # Factor 1: Similitud semántica (60%)
    semantic = similarity

    # Factor 2: Coincidencia de palabras (25%)
    keywords = contar_palabras_en_documento(query, doc)

    # Factor 3: Metadatos (15%)
    metadata = bonus_por_fuente_confiable(doc)

    # Combinación ponderada
    relevance = (semantic*0.60 + keywords*0.25 + metadata*0.15)

    return relevance  # 0.0 a 1.0

def _calculate_mrr(relevance_scores):
    """Encontrar posición del primer documento relevante"""

    for position, relevance in enumerate(relevance_scores, start=1):
        if relevance >= 0.7:  # umbral
            return 1.0 / position  # MRR

    return 0.0  # Ninguno relevante
```

---

## ✅ Conclusión

En tu sistema RAG:

1. **Se recuperan documentos** con distancia/similitud embedding
2. **Se calcula relevancia** combinando:
   - Similitud semántica
   - Coincidencia de keywords
   - Confiabilidad de la fuente
3. **Se ordena por relevancia** y se calcula MRR
4. **Se reporta la métrica** para evaluar calidad del RAG

**Tu MRR actual = 0.1667** significa que en promedio el primer documento relevante está en **posición 6**, lo cual se puede mejorar ajustando los pesos o el threshold.
