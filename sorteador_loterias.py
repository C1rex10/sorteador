import io
import random
from typing import List, Set, Tuple
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Constantes dos jogos
# -----------------------------
GAMES = {
    "Mega-Sena": {
        "n_bolas": 60,
        "n_escolhas": 6,
        "min_colunas": 6,
        "api": "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"
    },
    "Lotofácil": {
        "n_bolas": 25,
        "n_escolhas": 15,
        "min_colunas": 15,
        "api": "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
    },
}

# -----------------------------
# Utilidades
# -----------------------------
def detect_number_cols(df: pd.DataFrame, n_bolas: int) -> List[str]:
    candidate_cols = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna().astype(float)
        if len(valid) == 0:
            continue
        if ((valid % 1 == 0) & (valid >= 1) & (valid <= n_bolas)).mean() >= 0.7:
            candidate_cols.append(col)
    return candidate_cols

def rows_to_sets(df: pd.DataFrame, cols: List[str]) -> List[Set[int]]:
    jogos = []
    for _, row in df[cols].iterrows():
        vals = pd.to_numeric(row, errors="coerce").dropna().astype(int).tolist()
        if len(vals) >= 1:
            jogos.append(set(vals))
    return jogos

def frequency_stats(draws: List[Set[int]], n_bolas: int, recency_decay: float = None) -> pd.DataFrame:
    freq = np.zeros(n_bolas + 1, dtype=float)
    wfreq = np.zeros(n_bolas + 1, dtype=float)
    if len(draws) == 0:
        return pd.DataFrame({"dezena": [], "freq": [], "freq_ponderada": []})
    for s in draws:
        for n in s:
            freq[n] += 1
    if recency_decay is not None and recency_decay > 0:
        for i, s in enumerate(draws):
            power = (len(draws) - 1 - i)
            w = np.exp(-recency_decay * power)
            for n in s:
                wfreq[n] += w
    else:
        wfreq = freq.copy()

    df = pd.DataFrame({
        "dezena": np.arange(1, n_bolas + 1, dtype=int),
        "freq": freq[1:],
        "freq_ponderada": wfreq[1:],
    })
    df["pct"] = df["freq"] / max(1, len(draws))
    df = df.sort_values(["freq_ponderada", "freq", "dezena"], ascending=[False, False, True]).reset_index(drop=True)
    return df

def passes_constraints(combo: Set[int],
                       already_drawn: Set[Tuple[int, ...]] = None,
                       min_sum: int = None, max_sum: int = None,
                       max_consecutivos: int = None,
                       min_pares: int = None, max_pares: int = None) -> bool:
    arr = sorted(combo)
    s = sum(arr)
    if min_sum is not None and s < min_sum:
        return False
    if max_sum is not None and s > max_sum:
        return False
    if max_consecutivos is not None:
        consecutivos, atual = 1, 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1] + 1:
                atual += 1
                consecutivos = max(consecutivos, atual)
            else:
                atual = 1
        if consecutivos > max_consecutivos:
            return False
    if min_pares is not None or max_pares is not None:
        pares = sum(1 for x in arr if x % 2 == 0)
        if min_pares is not None and pares < min_pares:
            return False
        if max_pares is not None and pares > max_pares:
            return False
    if already_drawn is not None:
        if tuple(arr) in already_drawn:
            return False
    return True

def build_already_drawn(draws: List[Set[int]]) -> Set[Tuple[int, ...]]:
    return {tuple(sorted(d)) for d in draws}

# -----------------------------
# API oficial CAIXA
# -----------------------------
def fetch_last6m(jogo: str) -> pd.DataFrame:
    import re

    def _to_int_any(x):
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x)
        m = re.findall(r"\d+", s)
        return int("".join(m)) if m else None

    def _parse_date_any(s: str) -> datetime:
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        return pd.to_datetime(s, dayfirst=True).to_pydatetime()

    home_url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/home/ultimos-resultados"
    r = requests.get(home_url, timeout=30)
    r.raise_for_status()
    data_home = r.json()

    sec_key = None
    for k in data_home.keys():
        if jogo == "Mega-Sena" and "mega" in k.lower():
            sec_key = k; break
        if jogo == "Lotofácil" and "facil" in k.lower():
            sec_key = k; break

    section = data_home[sec_key]
    if isinstance(section, list) and section:
        section = section[0]

    num = None
    for k in ("numero", "concurso", "numeroConcurso", "concursoNumero", "numeroDoConcurso"):
        if k in section and section[k]:
            num = _to_int_any(section[k]); break
    if not num:
        st.error(f"Não achei número do concurso em {sec_key}.")
        st.stop()

    dt_ap = None
    for k in ("dataApuracao", "data", "dtApuracao", "data_sorteio"):
        if k in section and section[k]:
            dt_ap = _parse_date_any(section[k]); break
    if not dt_ap:
        st.error(f"Não achei data em {sec_key}.")
        st.stop()

    base_url = GAMES[jogo]["api"]
    limite_data = dt_ap - timedelta(days=180)

    rows = []
    for num_conc in range(num, 0, -1):
        url = f"{base_url}/{num_conc}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            break
        data = r.json()

        data_ap = None
        for k in ("dataApuracao", "data", "dtApuracao"):
            if k in data and data[k]:
                data_ap = _parse_date_any(data[k]); break
        if not data_ap:
            data_ap = dt_ap

        if data_ap < limite_data:
            break

        dezenas = None
        for k in ("listaDezenas", "dezenas", "resultadoOrdenado", "listaDezenasOrdemSorteio"):
            if k in data and data[k]:
                dezenas = [int(d) for d in data[k]]
                break
        if dezenas is None:
            continue

        row = {"concurso": _to_int_any(data.get("numero", data.get("numeroDoConcurso", num_conc))),
               "data": data_ap.strftime("%d/%m/%Y"),
               "acumulado": data.get("acumulado"),
               "valorPremio": data.get("valorEstimadoProximoConcurso")}
        for i, d in enumerate(dezenas, start=1):
            row[f"d{i}"] = d
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("concurso").reset_index(drop=True)
    return df

# -----------------------------
# Estratégia fixa: números quentes
# -----------------------------
def gen_weighted(freq_df: pd.DataFrame, k: int, power: float = 1.0) -> Set[int]:
    pool = freq_df["dezena"].to_numpy()
    w = (freq_df["freq_ponderada"].to_numpy() + 1e-6) ** power
    w = w / w.sum()
    chosen = set()
    available = pool.tolist()
    weights = w.tolist()
    for _ in range(k):
        idx = np.random.choice(np.arange(len(available)), p=np.array(weights) / sum(weights))
        chosen.add(int(available[idx]))
        del available[idx]
        del weights[idx]
    return chosen

# -----------------------------
# Função para criar cards
# -----------------------------
def card_container(title: str, color: str, icon: str, content: str) -> str:
    return f"""
    <div style='border:2px solid {color}; border-radius:10px; padding:16px; margin:18px 0;'>
        <h3 style='color:{color}; margin-top:0;'>{icon} {title}</h3>
        <div style='margin-top:12px; font-size:16px;'>
            {content}
        </div>
    </div>
    """

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Sorteador Mega-Sena & Lotofácil", page_icon="🎲", layout="wide")
st.title("🎲 SORTEADOR INTELIGENTE • MEGA-SENA & LOTOFÁCIL")

with st.sidebar:
    jogo = st.selectbox("JOGO", list(GAMES.keys()))
    n_bolas = GAMES[jogo]["n_bolas"]
    n_escolhas = GAMES[jogo]["n_escolhas"]

with st.spinner("Buscando resultados oficiais da CAIXA..."):
    df = fetch_last6m(jogo)

cols_dezenas = detect_number_cols(df, n_bolas)
df_sorted = df.sort_values("concurso").reset_index(drop=True)
draws = rows_to_sets(df_sorted, cols_dezenas)
already_drawn = build_already_drawn(draws)
freq_df = frequency_stats(draws, n_bolas=n_bolas)

# ==== Último Concurso ====
ultimo = df_sorted.iloc[-1]
dezenas_ultimo = [int(ultimo[c]) for c in cols_dezenas if c in df_sorted.columns]
valor_premio = ultimo.get("valorPremio")

dezenas_html = "".join([f"<div class='ball'>{d}</div>" for d in dezenas_ultimo])
ultimo_content = f"""
<div style='display:flex; justify-content:space-between; font-size:18px; font-weight:600; margin-bottom:12px;'>
    <span>Concurso: {ultimo['concurso']}</span>
    <span>Data: {ultimo['data']}</span>
    <span style='color:#f1c40f;'>Prêmio: R$ {valor_premio:,}</span>
</div>
<h4>Dezenas sorteadas:</h4>
<div class='balls'>{dezenas_html}</div>
"""
st.markdown(card_container("Último Concurso", "#3498db", "📌", ultimo_content), unsafe_allow_html=True)

# ==== Palpites ====
palpite_content = """
Defina a quantidade de palpites e clique no botão abaixo para gerar.
"""
st.markdown(card_container("Palpites (baseados em números quentes)", "#9b59b6", "🧪", palpite_content), unsafe_allow_html=True)

n_palpites = st.number_input("Quantidade de palpites", 1, 200, 10, 1, key="palpites")
if st.button("🔄 Gerar novos palpites"):
    generated, tries = [], 0
    while len(generated) < n_palpites and tries < n_palpites*200:
        tries += 1
        combo = gen_weighted(freq_df, n_escolhas, power=1.2)
        if passes_constraints(combo, already_drawn=already_drawn):
            generated.append(sorted(list(combo)))
    out_df = pd.DataFrame(generated, columns=[f"DEZENA {i}" for i in range(1, n_escolhas + 1)])
    out_df["SOMA"] = out_df.sum(axis=1)
    out_df["PARES"] = out_df.apply(lambda r: sum(1 for x in r[:n_escolhas] if x % 2 == 0), axis=1)
    st.dataframe(out_df, use_container_width=True)
    st.download_button("⬇️ Baixar palpites (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"palpites_{jogo.replace(' ', '').lower()}.csv", mime="text/csv")

# ==== Aposta Aleatória ====
aposta_content = """
Clique no botão abaixo para gerar uma aposta misturando números quentes e frios.
"""
st.markdown(card_container("Gerar Aposta Aleatória", "#27ae60", "🎲", aposta_content), unsafe_allow_html=True)

if st.button("🎰 SORTEAR APOSTA ALEATÓRIA"):
    metade = n_escolhas // 2
    quentes = freq_df.head(20)["dezena"].tolist()
    frios = freq_df.tail(20)["dezena"].tolist()
    aposta = sorted(random.sample(quentes, metade) + random.sample(frios, n_escolhas - metade))
    st.markdown("<div class='balls'>" + "".join([f"<div class='ball'>{d}</div>" for d in aposta]) + "</div>",
                unsafe_allow_html=True)

# ==== Últimos 5 Concursos ====
ultimos_html = ""
ultimos5 = df_sorted.tail(5)
for _, row in ultimos5.iterrows():
    dezenas = [int(row[c]) for c in cols_dezenas if c in df_sorted.columns]
    dezenas_html = "".join([f"<div class='ball'>{d}</div>" for d in dezenas])
    ultimos_html += f"<div style='margin-bottom:12px;'><b>Concurso {row['concurso']} ({row['data']})</b><br><div class='balls'>{dezenas_html}</div></div>"

st.markdown(card_container("Últimos 5 Concursos", "#e74c3c", "📅", ultimos_html), unsafe_allow_html=True)

# ==== Estilo bolas ====
st.markdown("""
<style>
.balls{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.ball{width:44px;height:44px;border-radius:50%;
      display:flex;align-items:center;justify-content:center;
      font-weight:700;color:#fff;box-shadow:0 2px 6px rgba(0,0,0,.25)}
.ball:nth-child(6n+1){background:#1abc9c}
.ball:nth-child(6n+2){background:#3498db}
.ball:nth-child(6n+3){background:#9b59b6}
.ball:nth-child(6n+4){background:#f39c12}
.ball:nth-child(6n+5){background:#e74c3c}
.ball:nth-child(6n+6){background:#2ecc71}
</style>
""", unsafe_allow_html=True)

# ==== Rodapé ====
rodape_content = """
⚠️ Este app usa estatísticas históricas apenas para entretenimento.  
As loterias da CAIXA são aleatórias.  

📌 Criado e desenvolvido por <b>Diogo Amaral</b> — todos os direitos reservados
"""
st.markdown(card_container("Informações", "gray", "ℹ️", rodape_content), unsafe_allow_html=True)
