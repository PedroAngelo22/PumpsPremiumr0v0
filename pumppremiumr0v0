import streamlit as st
import pandas as pd
import math
import time
import numpy as np
from scipy.optimize import root
import graphviz
import matplotlib.pyplot as plt
import io

# Configura o Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')

# --- BIBLIOTECAS DE DADOS ---
MATERIAIS = {
    "Aço Carbono (novo)": 0.046, "Aço Carbono (pouco uso)": 0.1, "Aço Carbono (enferrujado)": 0.2,
    "Aço Inox": 0.002, "Ferro Fundido": 0.26, "PVC / Plástico": 0.0015, "Concreto": 0.5
}
K_FACTORS = {
    "Entrada de Borda Viva": 0.5, "Entrada Levemente Arredondada": 0.2, "Entrada Bem Arredondada": 0.04,
    "Saída de Tubulação": 1.0, "Válvula Gaveta (Totalmente Aberta)": 0.2, "Válvula Gaveta (1/2 Aberta)": 5.6,
    "Válvula Globo (Totalmente Aberta)": 10.0, "Válvula de Retenção (Tipo Portinhola)": 2.5,
    "Cotovelo 90° (Raio Longo)": 0.6, "Cotovelo 90° (Raio Curto)": 0.9, "Cotovelo 45°": 0.4,
    "Curva de Retorno 180°": 2.2, "Tê (Fluxo Direto)": 0.6, "Tê (Fluxo Lateral)": 1.8,
}
FLUIDOS = { "Água a 20°C": {"rho": 998.2, "nu": 1.004e-6}, "Etanol a 20°C": {"rho": 789.0, "nu": 1.51e-6} }

# --- Funções de Callback (sem alterações) ---
def adicionar_item(tipo_lista):
    novo_id = time.time()
    st.session_state[tipo_lista].append({"id": novo_id, "comprimento": 10.0, "diametro": 100.0, "material": "Aço Carbono (novo)", "acessorios": []})

def remover_ultimo_item(tipo_lista):
    if len(st.session_state[tipo_lista]) > 0: st.session_state[tipo_lista].pop()

# --- Funções de Cálculo (com novas adições) ---
def calcular_perda_serie(lista_trechos, vazao_m3h, fluido_selecionado):
    perda_total = 0
    for trecho in lista_trechos:
        perdas = calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado)
        perda_total += perdas["principal"] + perdas["localizada"]
    return perda_total

def calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado):
    if vazao_m3h < 0: vazao_m3h = 0
    rugosidade_mm = MATERIAIS[trecho["material"]]
    vazao_m3s, diametro_m = vazao_m3h / 3600, trecho["diametro"] / 1000
    nu = FLUIDOS[fluido_selecionado]["nu"]
    if diametro_m <= 0: return {"principal": 1e12, "localizada": 0, "velocidade": 0}
    area = (math.pi * diametro_m**2) / 4
    velocidade = vazao_m3s / area
    reynolds = (velocidade * diametro_m) / nu if nu > 0 else 0
    fator_atrito = 0
    if reynolds > 4000:
        rugosidade_m = rugosidade_mm / 1000
        log_term = math.log10((rugosidade_m / (3.7 * diametro_m)) + (5.74 / reynolds**0.9))
        fator_atrito = 0.25 / (log_term**2)
    elif reynolds > 0: fator_atrito = 64 / reynolds
    perda_principal = fator_atrito * (trecho["comprimento"] / diametro_m) * (velocidade**2 / (2 * 9.81))
    k_total_trecho = sum(ac["k"] * ac["quantidade"] for ac in trecho["acessorios"])
    perda_localizada = k_total_trecho * (velocidade**2 / (2 * 9.81))
    return {"principal": perda_principal, "localizada": perda_localizada, "velocidade": velocidade}

def calcular_analise_energetica(vazao_m3h, h_man, eficiencia_bomba_percent, eficiencia_motor_percent, horas_dia, custo_kwh, fluido_selecionado):
    rho = FLUIDOS[fluido_selecionado]["rho"]
    ef_bomba = eficiencia_bomba_percent / 100
    ef_motor = eficiencia_motor_percent / 100
    potencia_eletrica_kW = (vazao_m3h / 3600 * rho * 9.81 * h_man) / (ef_bomba * ef_motor) / 1000 if ef_bomba * ef_motor > 0 else 0
    custo_anual = potencia_eletrica_kW * horas_dia * 30 * 12 * custo_kwh
    return {"potencia_eletrica_kW": potencia_eletrica_kW, "custo_anual": custo_anual}

# --- NOVAS FUNÇÕES PARA CURVA DA BOMBA ---
def criar_funcao_curva(df_curva, col_x, col_y, grau=2):
    """Cria uma função polinomial a partir de pontos de uma curva."""
    # Garante que os dados sejam numéricos e remove linhas com NaN
    df_curva[col_x] = pd.to_numeric(df_curva[col_x], errors='coerce')
    df_curva[col_y] = pd.to_numeric(df_curva[col_y], errors='coerce')
    df_curva = df_curva.dropna(subset=[col_x, col_y])
    if len(df_curva) < grau + 1:
        return None # Não há pontos suficientes para o ajuste
    
    coeficientes = np.polyfit(df_curva[col_x], df_curva[col_y], grau)
    polinomio = np.poly1d(coeficientes)
    return polinomio

def encontrar_ponto_operacao(sistema, h_geometrica, fluido, func_curva_bomba):
    """Encontra a interseção entre a curva da bomba e a curva do sistema."""
    
    def curva_sistema(vazao_m3h):
        """Calcula a altura manométrica total do sistema para uma dada vazão."""
        perda_total = 0
        # Perdas em série antes
        perda_total += calcular_perda_serie(sistema['antes'], vazao_m3h, fluido)
        # Perdas em paralelo
        perda_par, _ = calcular_perdas_paralelo(sistema['paralelo'], vazao_m3h, fluido)
        if perda_par == -1: return 1e12 # Erro no cálculo
        perda_total += perda_par
        # Perdas em série depois
        perda_total += calcular_perda_serie(sistema['depois'], vazao_m3h, fluido)
        
        return h_geometrica + perda_total

    # Equação a ser resolvida: H_bomba(Q) - H_sistema(Q) = 0
    def erro(vazao_m3h):
        if vazao_m3h < 0: return 1e12
        return func_curva_bomba(vazao_m3h) - curva_sistema(vazao_m3h)

    # Chute inicial para o solver (ex: 50 m³/h)
    solucao = root(erro, 50.0)
    
    if solucao.success:
        vazao_op = solucao.x[0]
        altura_op = func_curva_bomba(vazao_op)
        return vazao_op, altura_op, curva_sistema
    else:
        return None, None, curva_sistema


# --- Inicialização do Estado da Sessão ---
if 'trechos_antes' not in st.session_state: st.session_state.trechos_antes = []
if 'trechos_depois' not in st.session_state: st.session_state.trechos_depois = []
if 'ramais_paralelos' not in st.session_state:
    st.session_state.ramais_paralelos = {
        "Ramal 1": [{"id": time.time(), "comprimento": 50.0, "diametro": 80.0, "material": "Aço Carbono (novo)", "acessorios": []}],
        "Ramal 2": [{"id": time.time() + 1, "comprimento": 50.0, "diametro": 100.0, "material": "Aço Carbono (novo)", "acessorios": []}]
    }

# --- Interface do Aplicativo ---
st.set_page_config(layout="wide", page_title="Análise de Redes Hidráulicas")
st.title("💧 Análise de Redes de Bombeamento (Série e Paralelo)")

# --- Barra Lateral ---
with st.sidebar:
    st.header("⚙️ Parâmetros Gerais")
    fluido_selecionado = st.selectbox("Selecione o Fluido", list(FLUIDOS.keys()))
    h_geometrica = st.number_input("Altura Geométrica (m)", 0.0, value=15.0)
    
    st.divider()

    # --- NOVA SEÇÃO: CURVA DA BOMBA ---
    with st.expander("📈 Curva da Bomba", expanded=True):
        st.info("Insira pelo menos 3 pontos da curva de performance da bomba.")
        
        # Tabela para Curva de Altura (Vazão vs. Altura)
        st.subheader("Curva de Altura Manométrica")
        if 'curva_altura_df' not in st.session_state:
            st.session_state.curva_altura_df = pd.DataFrame([
                {"Vazão (m³/h)": 0, "Altura (m)": 40},
                {"Vazão (m³/h)": 50, "Altura (m)": 35},
                {"Vazão (m³/h)": 100, "Altura (m)": 25},
            ])
        st.session_state.curva_altura_df = st.data_editor(st.session_state.curva_altura_df, num_rows="dynamic", key="editor_altura")

        # Tabela para Curva de Eficiência (Vazão vs. Eficiência)
        st.subheader("Curva de Eficiência")
        if 'curva_eficiencia_df' not in st.session_state:
            st.session_state.curva_eficiencia_df = pd.DataFrame([
                {"Vazão (m³/h)": 0, "Eficiência (%)": 0},
                {"Vazão (m³/h)": 50, "Eficiência (%)": 70},
                {"Vazão (m³/h)": 100, "Eficiência (%)": 65},
            ])
        st.session_state.curva_eficiencia_df = st.data_editor(st.session_state.curva_eficiencia_df, num_rows="dynamic", key="editor_eficiencia")

    st.divider()
    st.header("🔧 Rede de Tubulação")
    # (As seções de trechos em série e paralelo continuam aqui, sem alterações)
    # ...

    st.divider()
    st.header("🔌 Equipamentos e Custo")
    rend_motor = st.slider("Eficiência do Motor (%)", 1, 100, 90)
    horas_por_dia = st.number_input("Horas por Dia", 1.0, 24.0, 8.0, 0.5)
    tarifa_energia = st.number_input("Custo da Energia (R$/kWh)", 0.10, 5.00, 0.75, 0.01, format="%.2f")

# --- Lógica Principal e Exibição de Resultados ---
try:
    # 1. Criar as funções da curva da bomba a partir dos dados da tabela
    func_curva_bomba = criar_funcao_curva(st.session_state.curva_altura_df, "Vazão (m³/h)", "Altura (m)")
    func_curva_eficiencia = criar_funcao_curva(st.session_state.curva_eficiencia_df, "Vazão (m³/h)", "Eficiência (%)")

    if func_curva_bomba is None or func_curva_eficiencia is None:
        st.error("Por favor, forneça pontos de dados suficientes (pelo menos 3) para as curvas da bomba.")
        st.stop()
    
    # 2. Encontrar o Ponto de Operação
    sistema_atual = {'antes': st.session_state.trechos_antes, 'paralelo': st.session_state.ramais_paralelos, 'depois': st.session_state.trechos_depois}
    vazao_op, altura_op, func_curva_sistema = encontrar_ponto_operacao(sistema_atual, h_geometrica, fluido_selecionado, func_curva_bomba)

    if vazao_op is None:
        st.error("Não foi possível encontrar o ponto de operação. Verifique a curva da bomba e os parâmetros da rede. A bomba pode ser incompatível com o sistema.")
        st.stop()
    
    # 3. Calcular tudo com base no ponto de operação real
    eficiencia_op = func_curva_eficiencia(vazao_op)
    resultados_energia = calcular_analise_energetica(vazao_op, altura_op, eficiencia_op, rend_motor, horas_por_dia, tarifa_energia, fluido_selecionado)
    perda_total_sistema = altura_op - h_geometrica

    # --- Exibição de Resultados ---
    st.header("📊 Resultados da Análise no Ponto de Operação")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vazão de Operação", f"{vazao_op:.2f} m³/h")
    c2.metric("Altura Manométrica de Operação", f"{altura_op:.2f} m")
    c3.metric("Eficiência da Bomba no Ponto", f"{eficiencia_op:.1f} %")
    c4.metric("Custo Anual de Energia", f"R$ {resultados_energia['custo_anual']:.2f}")

    # --- Gráfico da Curva da Bomba vs. Sistema ---
    st.header("📈 Gráfico de Curvas: Bomba vs. Sistema")
    max_vazao_curva = st.session_state.curva_altura_df['Vazão (m³/h)'].max()
    vazao_range = np.linspace(0, max_vazao_curva * 1.2, 100)
    
    altura_bomba = func_curva_bomba(vazao_range)
    altura_sistema = [func_curva_sistema(q) for q in vazao_range]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(vazao_range, altura_bomba, label='Curva da Bomba', color='royalblue', lw=2)
    ax.plot(vazao_range, altura_sistema, label='Curva do Sistema', color='seagreen', lw=2)
    ax.scatter(vazao_op, altura_op, color='red', s=100, zorder=5, label=f'Ponto de Operação ({vazao_op:.1f} m³/h, {altura_op:.1f} m)')
    
    ax.set_xlabel("Vazão (m³/h)")
    ax.set_ylabel("Altura Manométrica (m)")
    ax.set_title("Curva da Bomba vs. Curva do Sistema")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, vazao_range.max())
    ax.set_ylim(0, max(altura_bomba.max(), altura_sistema[-1]) * 1.1)
    
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Ocorreu um erro durante o cálculo. Verifique os parâmetros de entrada. Detalhe: {str(e)}")

