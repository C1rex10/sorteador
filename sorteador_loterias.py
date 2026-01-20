import io
import random
from typing import List, Set, Tuple
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from collections import Counter

# -----------------------------
# Constantes dos jogos
# -----------------------------
GAMES = {
    "MEGA-SENA": {
        "n_bolas": 60,
        "n_escolhas": 6,
        "min_colunas": 6,
        "api": "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"
    },
    "LOTOFÁCIL": {
        "n_bolas": 25,
        "n_escolhas": 15,
        "min_colunas": 15,
        "api": "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
    },
}
GAME_PARAMS = {
    "MEGA-SENA": {
        "recency_decay": 0.045,
        "power": 1.6,
        "top_quentes": 18,
        "top_frios": 18
    },
    "LOTOFÁCIL": {
        "recency_decay": 0.025,
        "power": 1.25,
        "top_quentes": 12,
        "top_frios": 12
    }
}

# -----------------------------
# Utilidades
# -----------------------------
def detect_number_cols(df: pd.DataFrame, n_bolas: int) -> List[str]:
    import re
    # 1) Preferir explicitamente colunas no formato d1, d2, ...
    cols = [c for c in df.columns if re.fullmatch(r"d\d+", c)]
    if cols:
        return sorted(cols, key=lambda x: int(x[1:]))

    # 2) Fallback heurístico (se algum dia mudar o schema da API)
    candidate_cols = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        if series.empty:
            continue
        ratio = ((series >= 1) & (series <= n_bolas)).mean()
        if ratio >= 0.95 and series.nunique() >= 10:
            candidate_cols.append(col)
    return candidate_cols


def rows_to_sets(df: pd.DataFrame, cols: List[str], n_bolas: int) -> List[Set[int]]:
    jogos = []
    for _, row in df[cols].iterrows():
        vals = pd.to_numeric(row, errors="coerce").dropna().astype(int).tolist()
        vals = [v for v in vals if 1 <= v <= n_bolas]  # filtra 1..n_bolas
        if len(vals) >= 1:
            jogos.append(set(vals))
    return jogos


def frequency_stats(draws: List[Set[int]],
                    n_bolas: int,
                    n_escolhas: int,
                    recency_decay: float = 0.03) -> pd.DataFrame:

    total_draws = len(draws)
    esperado = total_draws * (n_escolhas / n_bolas)

    freq = np.zeros(n_bolas + 1, dtype=float)
    wfreq = np.zeros(n_bolas + 1, dtype=float)

    for i, s in enumerate(draws):
        peso = np.exp(-recency_decay * (total_draws - 1 - i))
        for n in s:
            freq[n] += 1
            wfreq[n] += peso

    df = pd.DataFrame({
        "dezena": np.arange(1, n_bolas + 1),
        "freq": freq[1:],
        "freq_recente": wfreq[1:]
    })
    # Desvio do esperado
    df["desvio"] = df["freq"] - esperado

    # Normalização (z-score suave)
    df["z_freq"] = (df["freq"] - df["freq"].mean()) / (df["freq"].std() + 1e-6)
    df["z_recente"] = (df["freq_recente"] - df["freq_recente"].mean()) / (df["freq_recente"].std() + 1e-6)

    # Score final de calor
    df["score_quente"] = (
        0.55 * df["z_recente"] +
        0.35 * df["z_freq"] +
        0.10 * (df["desvio"] / esperado)
    )

    df = df.sort_values("score_quente", ascending=False).reset_index(drop=True)
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


def format_brl(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = str(v).strip().replace("R$", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        num = float(s)
    except Exception:
        return str(v)
    return f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def normalize_prize(valor, jogo: str):
    """
    Corrige erro sistêmico da API da CAIXA.
    Para Lotofácil, TODOS os valores vêm 10x maiores.
    """
    try:
        v = float(valor)
    except Exception:
        return valor

    if jogo == "LOTOFÁCIL":
        v = v / 10

    return v



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
        if jogo == "MEGA-SENA" and "mega" in k.lower():
            sec_key = k
            break
        if jogo == "LOTOFÁCIL" and "facil" in k.lower():
            sec_key = k
            break

    section = data_home[sec_key]
    if isinstance(section, list) and section:
        section = section[0]

    num = None
    for k in ("numero", "concurso", "numeroConcurso", "concursoNumero", "numeroDoConcurso"):
        if k in section and section[k]:
            num = _to_int_any(section[k])
            break
    if not num:
        st.error(f"Não achei número do concurso em {sec_key}. Campos: {list(section.keys())}")
        st.stop()

    dt_ap = None
    for k in ("dataApuracao", "data", "dtApuracao", "data_sorteio"):
        if k in section and section[k]:
            dt_ap = _parse_date_any(section[k])
            break
    if not dt_ap:
        st.error(f"Não achei data em {sec_key}. Campos: {list(section.keys())}")
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
                data_ap = _parse_date_any(data[k])
                break
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

        row = {
            "concurso": _to_int_any(data.get("numero", data.get("numeroDoConcurso", num_conc))),
            "data": data_ap.strftime("%d/%m/%Y"),
            "acumulado": data.get("acumulado"),
            # prêmio REAL do concurso
            "valorPremio": data.get("valorPremio"),
            # prêmio estimado do próximo
            "valorEstimado": data.get("valorEstimadoProximoConcurso")
        }

        for i, d in enumerate(dezenas, start=1):
            row[f"d{i}"] = d
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("concurso").reset_index(drop=True)
    return df

# -----------------------------
# Estratégia fixa: números quentes
# -----------------------------
def gen_mixed(freq_df, n_escolhas, jogo, recent_usage):
    quentes_n = int(n_escolhas * 0.45)
    medios_n  = int(n_escolhas * 0.35)
    livres_n  = n_escolhas - quentes_n - medios_n

    quentes = gen_weighted(freq_df.head(20), quentes_n, recent_usage, power=1.3)
    medios  = gen_weighted(freq_df.iloc[20:40], medios_n, recent_usage, power=1.0)
    livres  = set(random.sample(range(1, GAMES[jogo]["n_bolas"] + 1), livres_n))

    return quentes | medios | livres



def gen_weighted(freq_df: pd.DataFrame,
                 k: int,
                 recent_usage: dict,
                 power: float = 1.4) -> Set[int]:

    pool = freq_df["dezena"].to_numpy()
    base_w = np.maximum(freq_df["score_quente"].to_numpy(), 0) + 1e-6
    base_w = base_w ** power

    penalized_w = []
    for dez, w in zip(pool, base_w):
        penalized_w.append(
            w / (1 + 0.35 * recent_usage.get(int(dez), 0))
        )

    w = np.array(penalized_w)
    w = w / w.sum()

    chosen = set()
    available = pool.tolist()
    weights = w.tolist()

    for _ in range(k):
        idx = np.random.choice(len(available), p=np.array(weights) / sum(weights))
        d = int(available[idx])
        chosen.add(d)

        del available[idx]
        del weights[idx]

    return chosen


def is_too_similar(combo: Set[int],
                   existing: List[Set[int]],
                   max_overlap: int) -> bool:
    for prev in existing:
        if len(combo & prev) >= max_overlap:
            return True
    return False


# -----------------------------
# Função para criar cards
# -----------------------------
def card_container(title: str, color: str, icon: str, inner_html: str) -> str:
    return f"""
    <div style='border:2px solid {color}; border-radius:10px; padding:16px; margin:18px 0;'>
        <h3 style='color:{color}; margin-top:0;'>{icon} {title}</h3>
        {inner_html}
    """

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Sorteador Mega-Sena & Lotofácil", page_icon="🎲", layout="wide")
st.title("🎲 SORTEADOR INTELIGENTE • MEGA-SENA & LOTOFÁCIL")
st.caption("Gera palpites com base nos últimos sorteios da **CAIXA**. Uso recreativo — loterias são aleatórias; não há garantia de ganho.")

with st.sidebar:
    jogos = list(GAMES.keys())
    jogo = st.selectbox(
        "JOGO",
        jogos,
        index=jogos.index("LOTOFÁCIL"))

    n_bolas = GAMES[jogo]["n_bolas"]
    n_escolhas = GAMES[jogo]["n_escolhas"]
    st.write(f"FAIXA DEZENAS 1..{n_bolas} • QUANTIDADE POR VOLANTE: {n_escolhas}")

with st.spinner("Buscando resultados oficiais da CAIXA..."):
    df = fetch_last6m(jogo)

cols_dezenas = detect_number_cols(df, n_bolas)
df_sorted = df.sort_values("concurso").reset_index(drop=True)
draws = rows_to_sets(df_sorted, cols_dezenas, n_bolas)
already_drawn = build_already_drawn(draws)
params = GAME_PARAMS[jogo]

freq_df = frequency_stats(
    draws,
    n_bolas=n_bolas,
    n_escolhas=n_escolhas,
    recency_decay=params["recency_decay"]
)


# ==== Último Concurso ====
ultimo = df_sorted.iloc[-1]
dezenas_ultimo = [int(ultimo[c]) for c in cols_dezenas if c in df_sorted.columns]
valor_premio = ultimo.get("valorPremio")

if not valor_premio:
    valor_premio = ultimo.get("valorEstimado")

valor_premio = normalize_prize(valor_premio, jogo)


# monta as dezenas em HTML
dezenas_html = "".join([f"<div class='ball'>{d}</div>" for d in dezenas_ultimo])

# conteúdo do card
ultimo_content = f"""
<div style='display:flex; justify-content:space-between; font-size:18px; font-weight:600; margin-bottom:12px;'>
    <span>Concurso: {ultimo['concurso']}</span>
    <span>Data: {ultimo['data']}</span>
    <span style='color:#f1c40f;'>PRÊMIO: R$ {format_brl(valor_premio)}</span>
</div>
<h4 style='color:#3498db;'>DEZENAS SORTEADAS:</h4>
<div class='balls'>{dezenas_html}</div>
"""

st.markdown(card_container("ÚLTIMO CONCURSO", "#3498db", "📌", ultimo_content), unsafe_allow_html=True)

# ==== Palpites ====
palpite_content = "<p>Defina a quantidade de palpites e clique no botão abaixo para gerar.</p>"
st.markdown(card_container("PALPITES (BASEADO EM NÚMEROS QUENTES)", "#9b59b6", "🧪", palpite_content), unsafe_allow_html=True)

n_palpites = st.number_input("Quantidade de palpites", 1, 200, 10, 1, key="palpites")



if st.button("🔄 GERAR PALPITES"):
    recent_usage = Counter()   # 👈 AQUI
    generated = []
    generated_sets = []

    tries = 0
    max_tries = n_palpites * 300


    pool_size = max(int(n_bolas * 0.80), n_escolhas + 5)

    freq_df_used = freq_df.head(pool_size)

    while len(generated) < n_palpites and tries < max_tries:
        tries += 1

        combo = gen_weighted(
            freq_df_used,
            n_escolhas,
            recent_usage,
            power=params["power"]
        )

        # filtros leves (somente enquanto dá)
        if passes_constraints(
                combo,
                already_drawn=already_drawn if jogo == "MEGA-SENA" else None
        ):
            generated.append(sorted(combo))
            generated_sets.append(set(combo))

            for d in combo:
                recent_usage[d] += 1

    if len(generated) < n_palpites:
        st.warning(
            f"⚠️ Foram gerados apenas {len(generated)} de {n_palpites} palpites. "
            f"Filtros ficaram muito restritivos."
        )

    out_df = pd.DataFrame(
        generated,
        columns=[f"DEZENA {i}" for i in range(1, n_escolhas + 1)]
    )

    out_df["SOMA"] = out_df.sum(axis=1)
    out_df["PARES"] = out_df.apply(
        lambda r: sum(1 for x in r[:n_escolhas] if x % 2 == 0),
        axis=1
    )

    st.dataframe(out_df, use_container_width=True)

    st.download_button(
        "⬇️ Baixar palpites (CSV)",
        out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"palpites_{jogo.replace(' ', '').lower()}.csv",
        mime="text/csv"
    )



# ==== Aposta Aleatória ====
aposta_content = "<p>Clique no botão abaixo para gerar uma aposta misturando números quentes e frios.</p>"
st.markdown(card_container("GERAR APOSTA ALEATÓRIA", "#27ae60", "🎲", aposta_content), unsafe_allow_html=True)

if st.button("🎰 SORTEAR ALEATÓRIA"):
    metade = n_escolhas // 2
    quentes = freq_df.head(20)["dezena"].tolist()
    frios = freq_df.tail(20)["dezena"].tolist()

    escolhidos_quentes = random.sample(quentes, min(metade, len(quentes)))
    escolhidos_frios = random.sample(frios, min(n_escolhas - metade, len(frios)))

    escolhidos = set(escolhidos_quentes + escolhidos_frios)

    todas_dezenas = list(range(1, n_bolas + 1))

    aposta = sorted(escolhidos)

    st.markdown(
        "<div class='balls'>" + "".join([f"<div class='ball'>{d}</div>" for d in aposta]) + "</div>",
        unsafe_allow_html=True
    )
st.subheader("🔥 NÚMEROS QUENTES")

quentes = freq_df.head(params["top_quentes"])

st.dataframe(
    quentes[["dezena", "freq", "score_quente"]],
    use_container_width=True
)

st.subheader("❄️ NÚMEROS FRIOS")

frios = freq_df.tail(params["top_frios"]).sort_values("score_quente")

st.dataframe(
    frios[["dezena", "freq", "score_quente"]],
    use_container_width=True
)

# ==== Últimos 5 Concursos ====
ultimos_html = ""
ultimos5 = df_sorted.tail(5)
for _, row in ultimos5.iterrows():
    dezenas = [int(row[c]) for c in cols_dezenas if c in df_sorted.columns]
    dezenas_html = "".join([f"<div class='ball'>{d}</div>" for d in dezenas])
    ultimos_html += f"<div style='margin-bottom:12px;'><b>CONCURSO {row['concurso']} ({row['data']})</b><br><div class='balls'>{dezenas_html}</div></div>"

st.markdown(card_container("ÚLTIMOS 5 CONCURSOS", "#e74c3c", "📅", ultimos_html), unsafe_allow_html=True)



chart_data = quentes.copy()
chart_data["dezena"] = chart_data["dezena"].astype(str)

bars = alt.Chart(chart_data).mark_bar().encode(
    x=alt.X("dezena:N", title="Dezena"),
    y=alt.Y("score_quente:Q", title="Score de Calor"),
    tooltip=["dezena", "score_quente"]
)

labels = alt.Chart(chart_data).mark_text(
    dy=-5,
    color="white",
    fontSize=12
).encode(
    x="dezena:N",
    y="score_quente:Q",
    text=alt.Text("score_quente:Q", format=".2f")
)

st.altair_chart(
    (bars + labels).properties(
        height=320,
        title="🔥 Ranking de Números Quentes"
    ),
    use_container_width=True
)


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
st.markdown("""
<hr>
<div style='text-align:center; padding:10px; font-size:14px; color:gray;'>
⚠️ Este app usa estatísticas históricas apenas para entretenimento.<br>
As loterias da CAIXA são aleatórias.<br><br>
📌 Criado e desenvolvido por <b>Diogo Amaral</b> — todos os direitos reservados
</div>
""", unsafe_allow_html=True)
