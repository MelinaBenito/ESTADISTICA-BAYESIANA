# ESTADISTICA-BAYESIANA
import streamlit as st
import numpy as np
import pandas as pd
from faker import Faker
import plotly.express as px

# ----------------------------------------------------------------------
# 1. PARÁMETROS DEL PROBLEMA Y SIMULACIÓN
# ----------------------------------------------------------------------

# Contexto: Pruebas PCR de COVID-19 a inicios del  2020 (Alta Prevalencia de Pandemia)
# Nota: La prevalencia real varió mucho. Usaremos una prevalencia inicial del 5% para la simulación inicial, 
# pero el análisis se enfocará en el TEÓRICO 0.1% para demostrar la paradoja en poblaciones generales.
PREVALENCIA_INICIAL = 0.05    # P(A) en la simulación (5%)
SENSIBILIDAD = 0.95           # P(B|A) (95%)
ESPECIFICIDAD = 0.98          # P(B^c|A^c) (98%)
N_REGISTROS = 100

PROB_FALSO_POSITIVO = 1 - ESPECIFICIDAD # P(B|A^c) = 0.02
PROB_SANO = 1 - PREVALENCIA_INICIAL

# Inicializar Faker
fake = Faker('es_ES')
np.random.seed(42) 

# ----------------------------------------------------------------------
# 2. FUNCIÓN DE GENERACIÓN DE DATOS (FAKER Y SIMULACIÓN)
# ----------------------------------------------------------------------

@st.cache_data
def generar_datos_salud(n_registros, prevalencia, sensibilidad, especificidad):
    """Genera n_registros simulando datos de diagnóstico COVID-19 con Faker."""
    data = []
    prob_falso_positivo = 1 - especificidad
    
    for i in range(n_registros):
        # SIMULACIÓN 1: ESTADO REAL (Basado en Prevalencia)
        estado_real = np.random.choice(
            ['Enfermo (A)', 'Sano (Ac)'],
            p=[prevalencia, 1 - prevalencia]
        )
        
        # SIMULACIÓN 2: RESULTADO DE LA PRUEBA (Condicional al Estado Real)
        if estado_real == 'Enfermo (A)':
            # Si está enfermo: usa Sensibilidad
            resultado_prueba = np.random.choice(
                ['Positivo (B)', 'Negativo (Bc)'],
                p=[sensibilidad, 1 - sensibilidad]
            )
        else:
            # Si está sano: usa Tasa de Falso Positivo
            resultado_prueba = np.random.choice(
                ['Positivo (B)', 'Negativo (Bc)'],
                p=[prob_falso_positivo, especificidad]
            )

        # GENERACIÓN CONTEXTUAL CON FAKER
        data.append({
            'ID_Prueba': 1000 + i,
            'Nombre_Completo': fake.name(),
            'Edad': fake.random_int(min=18, max=75),
            'Fecha_Muestra': fake.date_between(start_date='-1y', end_date='today'),
            'Estado_Real_COVID': estado_real,
            'Resultado_Prueba_PCR': resultado_prueba
        })

    return pd.DataFrame(data)


# ----------------------------------------------------------------------
# 3. LÓGICA DEL TEOROMA DE BAYES
# ----------------------------------------------------------------------

def calcular_bayes(prevalencia_teorica, sensibilidad, especificidad):
    """Calcula la probabilidad a posteriori P(A|B) usando Bayes."""
    
    # Probabilidades a priori y derivadas
    Pr_A = prevalencia_teorica
    Pr_Ac = 1 - Pr_A
    Pr_B_dado_A = sensibilidad
    Pr_B_dado_Ac = 1 - especificidad # Tasa de Falso Positivo
    
    # 1. Probabilidad Total de Positivo P(B)
    Pr_B = (Pr_B_dado_A * Pr_A) + (Pr_B_dado_Ac * Pr_Ac)
    
    # 2. Teorema de Bayes: P(A|B)
    Pr_A_dado_B = (Pr_B_dado_A * Pr_A) / Pr_B if Pr_B > 0 else 0
    
    # 3. Probabilidad de Falso Positivo (P(Ac|B))
    Pr_Ac_dado_B = 1 - Pr_A_dado_B
    
    return Pr_A_dado_B, Pr_B, Pr_Ac_dado_B


# ----------------------------------------------------------------------
# 4. INTERFAZ STREAMLIT
# ----------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Bayes y la Paradoja del Falso Positivo")

st.title("Demostración Bayesiana: La Paradoja del Falso Positivo")
st.markdown("---")
st.markdown("Con este ejmplo simulado trato de desmostrar la teoria de los falsos positivos, tomando de ejemplo el caso de covid-19 en el año 2020. Según datos del SINADEF del gobierno regional de Puno se señala que “por cada 100 000 habitantes fallecieron 31 personas” dejando un saldo aproximado de 388 victimas hasta septiembre de mismo año, siendo los picos mas altos en julio y agosto" \
", Sin embargo esta informacion no es del todo cierta , ya que ese mismo año ciudadanos de distintas partes del departamento de puno señalaban que algunas muertes no eran por covid ,aún así entraban al registro como fallecimiento por covid-19; lo cual nos lleva a hacernos la siguiente pregunta ¿HUBO FALSOS POSITIVOS EN LAS PRUEBAS COVID EN EL AÑO 2020 EN EL DEPARTAMENTO DE PUNO?")
st.sidebar.header("Parámetros del Problema de Diagnóstico")
st.sidebar.markdown(f"**Prueba PCR (Simulación):**")
st.sidebar.metric("Sensibilidad P(B|A)", SENSIBILIDAD)
st.sidebar.metric("Especificidad P(B^c|A^c)", ESPECIFICIDAD)

st.subheader("1. Generación y Exploración de la Muestra (N=100)")
st.caption("Esta muestra simula 100 pruebas COVID-19 (ene 2020-diciembre 2020 ) con Faker y una prevalencia inicial del 5%.")

df_simulacion = generar_datos_salud(N_REGISTROS, PREVALENCIA_INICIAL, SENSIBILIDAD, ESPECIFICIDAD)

col1_data, col2_data = st.columns([1, 2])

with col1_data:
    st.dataframe(df_simulacion.head(100))

with col2_data:
    st.markdown("**Tabla de Contingencia Observada (N=100):**")
    tabla = pd.crosstab(df_simulacion['Estado_Real_COVID'], df_simulacion['Resultado_Prueba_PCR'], margins=True, margins_name="Total")
    st.table(tabla)
    
    try:
        VP = tabla.loc['Enfermo (A)', 'Positivo (B)']
        FP = tabla.loc['Sano (Ac)', 'Positivo (B)']
        Total_Positivos = tabla.loc['Total', 'Positivo (B)']
        VPP_obs = VP / Total_Positivos if Total_Positivos > 0 else 0
        st.info(f"**Valor Predictivo Positivo (VPP) Observado:** {VPP_obs:.2%} (VP/Total Positivos)")
    except KeyError:
        st.warning("Advertencia: No se encontraron todos los casos en la muestra pequeña (N=100). El VPP observado puede ser 0 o irrelevante. Procedemos al análisis TEÓRICO.")


st.markdown("---")

# ----------------------------------------------------------------------
# 5. DEMOSTRACIÓN BAYESIANA INTERACTIVA
# ----------------------------------------------------------------------

st.subheader("2. Demostración del Teorema de Bayes y la Paradoja")
st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|A^c) \cdot P(A^c)}")

# Slider para manipular la prevalencia (P(A))
prevalencia_analisis = st.slider(
    "Ajuste la Prevalencia Teórica P(A) para el Análisis Bayesiano:",
    min_value=0.0001,
    max_value=0.5,
    value=0.001, # Valor bajo para forzar la paradoja (0.1%)
    step=0.001,
    format="%.4f"
)

# Cálculo bayesiano con el valor del slider
Pr_A_dado_B, Pr_B, Pr_Ac_dado_B = calcular_bayes(prevalencia_analisis, SENSIBILIDAD, ESPECIFICIDAD)

col_teorico, col_metricas, col_conclusion = st.columns(3)

with col_teorico:
    st.markdown("#### Probabilidades Teóricas")
    st.metric("P(A) Prevalencia (Input)", f"{prevalencia_analisis:.4f}")
    st.metric("P(A^c) Prob. Sano", f"{1-prevalencia_analisis:.4f}")
    st.metric("P(B|A^c) Falso Positivo Rate", f"{PROB_FALSO_POSITIVO:.2f} (2%)")

with col_metricas:
    st.markdown("#### Resultados Bayes (Salida)")
    st.metric("P(B) Prob. Total Positivo", f"{Pr_B:.4f}")
    st.metric(
        "P(A|B) VPP (Prob. de estar Enfermo si da Positivo)", 
        f"{Pr_A_dado_B:.4f}", 
        delta=f"{Pr_A_dado_B:.2%}" # Mostrar como porcentaje
    )

with col_conclusion:
    st.markdown("#### Conclusión Clave")
    st.metric(
        "P(A^c|B) ¡Prob. de Falso Positivo!", 
        f"{Pr_Ac_dado_B:.4f}", 
        delta=f"{Pr_Ac_dado_B:.2%}",
        delta_color="inverse"
    )
    
    st.warning(
        f"Con una prevalencia de {prevalencia_analisis:.2%}, a pesar de la alta calidad de la prueba, hay un {Pr_Ac_dado_B:.2%} de probabilidad de que el resultado positivo sea un FALSO POSITIVO."
    )

st.markdown("---")

# ----------------------------------------------------------------------
# 6. EXPLICACIÓN DE LA PARADOJA Y GRÁFICO DE PREVALENCIA
# ----------------------------------------------------------------------

st.subheader("3. Explicación y Demostración del Efecto de la Prevalencia")

st.markdown("""
La **Paradoja del Falso Positivo** ocurre cuando el **Valor Predictivo Positivo (VPP)**, $P(A|B)$, es bajo a pesar de que la prueba es altamente precisa (alta Sensibilidad y Especificidad).
""")

st.markdown("**¿Por qué sucede?**")

st.markdown(
    """
    * **Prevalencia Baja ($P(A)$):** En poblaciones generales o cuando la enfermedad es rara (ej. $P(A)=0.1\%$), la **cantidad de personas sanas** ($P(A^c) \ aprox 99.9\%$) es abrumadora.
    * **Falsos Positivos Acumulados:** Aunque la tasa de Falso Positivo ($P(B|A^c)$) es baja (2%), al multiplicarse por la enorme cantidad de personas sanas, el **número absoluto de Falsos Positivos** supera con creces el número de Verdaderos Positivos.
    * **El Teorema de Bayes pondera este conocimiento previo ($P(A)$) con la nueva evidencia ($P(B|A)$),** llevando a una probabilidad a posteriori $P(A|B)$ sorprendentemente baja.
    """
)
st.markdown("---")

# Generación de datos para el gráfico de impacto de prevalencia
prevalencias = np.linspace(0.0001, 0.5, 100)
resultados_bayes = [calcular_bayes(p, SENSIBILIDAD, ESPECIFICIDAD) for p in prevalencias]
Pr_A_dado_B_values = [res[0] for res in resultados_bayes]

df_grafico = pd.DataFrame({
    'Prevalencia P(A)': prevalencias,
    'Valor Predictivo Positivo P(A|B)': Pr_A_dado_B_values
})

st.markdown("#### Demostración Visual: La Prevalencia Domina el VPP")

fig = px.line(
    df_grafico,
    x='Prevalencia P(A)',
    y='Valor Predictivo Positivo P(A|B)',
    title='Impacto de la Prevalencia en la Probabilidad a Posteriori P(A|B)'
)
fig.update_layout(xaxis_tickformat = '.1%', yaxis_tickformat = '.1%')
st.plotly_chart(fig, use_container_width=True)

st.success(
    f"""
    **Conclusión del Gráfico:** La probabilidad de que una persona con resultado positivo esté realmente enferma y haya fallecido por COVID-19 (VPP) es casi nula (cercana al 4.5% si $P(A)=0.1\%$) hasta que la prevalencia poblacional supera aproximadamente el **5%** (donde el VPP se eleva por encima del $70\%$).
    """
)
