# app.py
# ------------------------------------------------------------
# AMF QCM + suivi des scores (SQLite) + "user_id" déterministe via hash/UUID
# + Graphs Altair (axe Y 0..100 + points visibles + labels)
# + Sous-thèmes APPLATIS au niveau 1 : 1.1, 1.2, 1.3... (1.2 inclut 1.2.1, 1.2.2, etc.)
# + Libellés de sous-thèmes "synthétiques" (pas basés sur UNE question, et sans "/")
# ------------------------------------------------------------
import random
import re
import sqlite3
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter

import pandas as pd
import streamlit as st
import altair as alt

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
EXCEL_PATH = "AMF_values.xlsx"   # mets le fichier à côté de app.py
DB_PATH = "scores.db"           # base locale (ou sur le serveur si déployé)

THEME_LABELS: Dict[str, str] = {
    "1": "Cadre institutionnel et réglementaire",
    "2": "Déontologie et protection des investisseurs",
    "3": "Abus de marché et opérations interdites",
    "4": "Instruments financiers",
    "5": "Marchés financiers",
    "6": "Gestion collective",
    "7": "Gestion pour compte de tiers",
    "8": "Produits d’épargne et assurance-vie",
    "9": "Fonctionnement des marchés et post-marché",
    "10": "Information financière et comptabilité",
    "11": "Finance durable et critères ESG",
    "12": "Lutte contre le blanchiment et le financement du terrorisme",
}

# Stopwords FR (petit set pour titres synthétiques)
STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "dans", "pour", "par", "au", "aux",
    "sur", "avec", "sans", "d", "l", "ce", "ces", "cette", "cet", "qui", "que", "quoi", "dont", "où",
    "est", "sont", "peut", "peuvent", "doit", "doivent", "a", "ont", "à", "se", "sa", "son", "ses",
    "leur", "leurs", "plus", "moins", "cas", "quelle", "quelles", "quel", "quels"
}

st.set_page_config(page_title="QCM AMF", layout="wide")


# ------------------------------------------------------------
# Helpers: user_id (hash -> uuid)
# ------------------------------------------------------------
def normalize_username(raw: str) -> str:
    """
    Normalise un input type "prenom_nom" (tolérant aux espaces / majuscules).
    """
    s = (raw or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_àâäçéèêëîïôöùûüÿñæœ-]", "", s, flags=re.IGNORECASE)
    s = s.replace("-", "_")
    s = re.sub(r"_+", "_", s)
    return s


def username_to_uuid(username_norm: str) -> str:
    """
    UUID déterministe à partir du texte (stable d'une session à l'autre).
    - SHA-256 pour éviter collisions faciles
    - uuid5 pour obtenir un UUID standard et stable
    """
    h = hashlib.sha256(username_norm.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, h))


# ------------------------------------------------------------
# Subtheme helpers (aplatir à X.Y)
# ------------------------------------------------------------
def subtheme_to_level1(sub: str) -> str:
    """
    "1.2.1" -> "1.2"
    "1.2"   -> "1.2"
    """
    s = str(sub).strip()
    parts = s.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    return s


def sub_sort_key(s: str):
    """
    tri humain : 1.2 < 1.10
    """
    s = str(s)
    parts = s.split(".")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    return (9999, 9999)


def _make_synth_title_from_questions(questions: pd.Series, top_k: int = 3) -> str:
    """
    Fabrique un intitulé synthétique à partir des questions d'un sous-thème lvl1.
    - extrait des mots fréquents (>=4 lettres)
    - enlève stopwords
    - construit un titre lisible sans "/"
    """
    text = " ".join(questions.astype(str).str.lower().tolist())
    words = re.findall(r"[a-zàâäçéèêëîïôöùûüÿñæœ]{4,}", text, flags=re.IGNORECASE)
    words = [w for w in words if w not in STOPWORDS]

    common = [w for w, _ in Counter(words).most_common(top_k)]
    if not common:
        return "Révision du sous-thème"

    # join lisible : "x et y" ou "x, y et z"
    if len(common) == 1:
        title = common[0]
    elif len(common) == 2:
        title = f"{common[0]} et {common[1]}"
    else:
        title = f"{common[0]}, {common[1]} et {common[2]}"

    # petite cosmétique
    title = title.strip()
    return title[:1].upper() + title[1:]


# ------------------------------------------------------------
# Chart helper (Altair): axe 0..100 + points visibles + labels
# ------------------------------------------------------------
def score_chart(df: pd.DataFrame, title: str = ""):
    if df.empty:
        st.info("Pas de données à afficher.")
        return

    tmp = df.copy()
    tmp["percent"] = pd.to_numeric(tmp["percent"], errors="coerce").fillna(0)
    tmp = tmp.sort_values("ts")

    base = (
        alt.Chart(tmp)
        .encode(
            x=alt.X("ts:T", title="Date"),
            y=alt.Y(
                "percent:Q",
                title="Score (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(format=".0f"),
            ),
            tooltip=[
                alt.Tooltip("ts:T", title="Date"),
                alt.Tooltip("percent:Q", title="Score (%)", format=".1f"),
            ],
        )
        .properties(title=title, height=320)
    )

    line = base.mark_line()
    points = base.mark_point(size=90)
    labels = base.mark_text(dy=-12, fontSize=12).encode(
        text=alt.Text("percent:Q", format=".0f")
    )

    st.altair_chart(line + points + labels, use_container_width=True)


# ------------------------------------------------------------
# Data loading / cleaning
# ------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df.rename(
        columns={
            "question contenant le numéro unique": "question",
            "Reponse": "answer_letter",
            "Sous_theme": "subtheme",
            "Theme": "theme",
        }
    )

    keep = [
        "n°identifiant",
        "theme",
        "subtheme",
        "question",
        "Choix_A",
        "Choix_B",
        "Choix_C",
        "answer_letter",
        "Document_justifiant_reponse",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError("Colonnes manquantes dans l'Excel : " + ", ".join(missing))

    df = df[keep].copy()

    df["theme"] = df["theme"].astype(str).str.strip()
    df["subtheme"] = df["subtheme"].astype(str).str.strip()
    df["answer_letter"] = df["answer_letter"].astype(str).str.strip().str.upper()

    df["question_display"] = df["question"].astype(str).apply(
        lambda s: re.sub(r"^\s*\d+\s*-\s*", "", s).strip()
    )

    df = df[df["answer_letter"].isin(["A", "B", "C"])].reset_index(drop=True)
    df["theme_label"] = df["theme"].map(THEME_LABELS).fillna(
        df["theme"].apply(lambda x: f"Thème {x}")
    )

    # Ajout sous-thème lvl1 (X.Y)
    df["subtheme_lvl1"] = df["subtheme"].apply(subtheme_to_level1)

    return df


def build_subtheme_labels_level1(df_theme: pd.DataFrame) -> Dict[str, str]:
    """
    Renvoie des libellés pour les sous-thèmes lvl1 :
      "1.2 — Agréments, autorités et ... (synth)"
    On calcule le titre à partir de toutes les questions du groupe.
    """
    labels: Dict[str, str] = {}
    for sub1, g in df_theme.groupby("subtheme_lvl1", sort=False):
        synth = _make_synth_title_from_questions(g["question_display"], top_k=3)
        labels[sub1] = f"{sub1} — {synth}"
    return labels


# ------------------------------------------------------------
# DB (SQLite)
# ------------------------------------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                ts TEXT NOT NULL,
                mode TEXT NOT NULL,
                theme TEXT,
                theme_label TEXT,
                subtheme TEXT,
                n_questions INTEGER NOT NULL,
                score INTEGER NOT NULL,
                percent REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_user_ts ON attempts(user_id, ts)"
        )
        conn.commit()


def save_attempt(
    user_id: str,
    user_name: str,
    mode: str,
    theme: Optional[str],
    theme_label: Optional[str],
    subtheme: Optional[str],
    n_questions: int,
    score: int,
):
    init_db()
    percent = round(100.0 * score / n_questions, 2)
    ts = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO attempts (user_id, user_name, ts, mode, theme, theme_label, subtheme, n_questions, score, percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, user_name, ts, mode, theme, theme_label, subtheme, n_questions, score, percent),
        )
        conn.commit()


def load_attempts(user_id: str) -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM attempts WHERE user_id = ? ORDER BY ts ASC",
            conn,
            params=(user_id,),
        )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


# ------------------------------------------------------------
# Quiz state helpers
# ------------------------------------------------------------
def init_state():
    defaults = {
        "quiz_rows": None,
        "i": 0,
        "answers": {},
        "submitted": {},
        "quiz_title": "",
        "score_saved": False,
        "quiz_meta": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def start_quiz(rows: pd.DataFrame, title: str, meta: Dict[str, Any]):
    st.session_state["quiz_rows"] = rows.to_dict("records")
    st.session_state["i"] = 0
    st.session_state["answers"] = {}
    st.session_state["submitted"] = {}
    st.session_state["quiz_title"] = title
    st.session_state["score_saved"] = False
    st.session_state["quiz_meta"] = meta


def current_row() -> Dict[str, Any]:
    return st.session_state["quiz_rows"][st.session_state["i"]]


def answer_to_text(row: Dict[str, Any], letter: str) -> str:
    return {"A": row["Choix_A"], "B": row["Choix_B"], "C": row["Choix_C"]}.get(
        letter, ""
    )


# ------------------------------------------------------------
# UI: Sidebar user + navigation
# ------------------------------------------------------------
df = load_data(EXCEL_PATH)
init_state()

with st.sidebar:
    st.header("Utilisateur")
    raw_name = st.text_input("Prénom_Nom", placeholder="ex: paul_dupont")
    name_norm = normalize_username(raw_name)

    if not name_norm:
        st.info("Entre ton **Prénom_Nom** (ex: paul_dupont) pour accéder au QCM et au suivi.")
        st.stop()

    user_id = username_to_uuid(name_norm)
    st.caption(f"ID (hashé) : `{user_id}`")

    st.divider()
    page = st.radio("Navigation", ["QCM", "Suivi"], index=0)


# ------------------------------------------------------------
# PAGE: QCM
# ------------------------------------------------------------
if page == "QCM":
    st.title("Entraînement QCM (AMF)")

    with st.sidebar:
        st.header("Génération")

        mode = st.radio(
            "Mode",
            ["Sous-thème (20 questions)", "Tous thèmes (100 questions)"],
            index=0,
        )

        themes_present = sorted(
            df["theme"].unique(), key=lambda x: int(x) if x.isdigit() else 9999
        )
        selected_theme = st.selectbox(
            "Choisir un thème",
            themes_present,
            format_func=lambda x: THEME_LABELS.get(str(x), f"Thème {x}"),
        )
        df_theme = df[df["theme"] == str(selected_theme)].copy()

        selected_subtheme_lvl1 = None
        if mode == "Sous-thème (20 questions)":
            subtheme_labels = build_subtheme_labels_level1(df_theme)
            subthemes_lvl1 = sorted(subtheme_labels.keys(), key=sub_sort_key)

            if not subthemes_lvl1:
                st.warning("Aucun sous-thème trouvé pour ce thème.")
            else:
                selected_subtheme_lvl1 = st.selectbox(
                    "Choisir un sous-thème",
                    subthemes_lvl1,
                    format_func=lambda x: subtheme_labels[x],
                )

        st.divider()
        colA, colB = st.columns(2)

        with colA:
            if st.button("🎲 Générer un nouveau QCM", use_container_width=True):
                if mode == "Sous-thème (20 questions)":
                    # IMPORTANT : 1.2 inclut 1.2.1 / 1.2.2 / ...
                    pool = df_theme[df_theme["subtheme"].str.startswith(str(selected_subtheme_lvl1))]
                    n = min(20, len(pool))
                    title = (
                        f"{THEME_LABELS.get(str(selected_theme), f'Thème {selected_theme}')}"
                        f" — {selected_subtheme_lvl1} (20 questions)"
                    )
                    meta = {
                        "mode": "subtheme_20",
                        "theme": str(selected_theme),
                        "theme_label": THEME_LABELS.get(
                            str(selected_theme), f"Thème {selected_theme}"
                        ),
                        # on stocke le lvl1 en DB
                        "subtheme": str(selected_subtheme_lvl1),
                    }
                else:
                    pool = df
                    n = min(100, len(pool))
                    title = "Tous thèmes (100 questions)"
                    meta = {
                        "mode": "global_100",
                        "theme": None,
                        "theme_label": None,
                        "subtheme": None,
                    }

                if n == 0:
                    st.error("Pas assez de questions disponibles pour générer ce QCM.")
                else:
                    rows_sample = pool.sample(
                        n=n, replace=False, random_state=random.randint(0, 10**9)
                    )
                    start_quiz(rows_sample, title, meta)
                    st.rerun()

        with colB:
            if st.button("🧹 Réinitialiser", use_container_width=True):
                st.session_state["quiz_rows"] = None
                st.session_state["i"] = 0
                st.session_state["answers"] = {}
                st.session_state["submitted"] = {}
                st.session_state["quiz_title"] = ""
                st.session_state["score_saved"] = False
                st.session_state["quiz_meta"] = {}
                st.rerun()

    if not st.session_state["quiz_rows"]:
        st.info("Clique sur **Générer un nouveau QCM** dans la barre de gauche.")
        st.stop()

    rows: List[Dict[str, Any]] = st.session_state["quiz_rows"]
    i = st.session_state["i"]
    total = len(rows)
    row = current_row()

    st.caption(st.session_state.get("quiz_title", ""))
    st.write(f"## Question {i + 1} / {total}")
    st.write(row["question_display"])

    qid = row["n°identifiant"]

    options = ["A", "B", "C"]
    labels = {
        "A": f"A — {row['Choix_A']}",
        "B": f"B — {row['Choix_B']}",
        "C": f"C — {row['Choix_C']}",
    }

    default_letter = st.session_state["answers"].get(qid, "A")
    default_index = options.index(default_letter) if default_letter in options else 0

    selected = st.radio(
        "Ta réponse",
        options=options,
        format_func=lambda x: labels[x],
        index=default_index,
        key=f"radio_{qid}",
    )
    st.session_state["answers"][qid] = selected

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("✅ Valider", use_container_width=True):
            st.session_state["submitted"][qid] = True
            st.rerun()

    with col2:
        if st.button("➡️ Suivante", use_container_width=True, disabled=(i >= total - 1)):
            st.session_state["i"] += 1
            st.rerun()

    with col3:
        if st.button("⬅️ Précédente", use_container_width=True, disabled=(i <= 0)):
            st.session_state["i"] -= 1
            st.rerun()

    if st.session_state["submitted"].get(qid, False):
        correct = row["answer_letter"]
        chosen = st.session_state["answers"][qid]

        def colored_line(letter: str) -> str:
            txt = labels[letter]
            if letter == correct:
                return (
                    "<div style='padding:10px;border-radius:10px;"
                    "background:#e9f7ef;border:1px solid #27ae60'>"
                    f"<b>{txt}</b></div>"
                )
            if letter == chosen and chosen != correct:
                return (
                    "<div style='padding:10px;border-radius:10px;"
                    "background:#fdecea;border:1px solid #e74c3c'>"
                    f"<b>{txt}</b></div>"
                )
            return (
                "<div style='padding:10px;border-radius:10px;"
                "background:#f6f6f6;border:1px solid #ddd'>"
                f"{txt}</div>"
            )

        st.write("### Correction")
        st.markdown(colored_line("A"), unsafe_allow_html=True)
        st.markdown(colored_line("B"), unsafe_allow_html=True)
        st.markdown(colored_line("C"), unsafe_allow_html=True)

        justif = row.get("Document_justifiant_reponse", "")
        if isinstance(justif, str) and justif.strip():
            st.caption(f"📌 Justification : {justif}")

    st.divider()

    submitted_count = sum(
        1 for r in rows if st.session_state["submitted"].get(r["n°identifiant"], False)
    )
    st.write(f"Validées : **{submitted_count}/{total}**")

    if submitted_count == total:
        score = 0
        wrong = []
        for r in rows:
            qid2 = r["n°identifiant"]
            chosen2 = st.session_state["answers"].get(qid2)
            if chosen2 == r["answer_letter"]:
                score += 1
            else:
                wrong.append((r, chosen2))

        st.success(f"🎯 Score final : **{score}/{total}** ({round(100 * score / total, 1)}%)")

        if not st.session_state["score_saved"]:
            meta = st.session_state.get("quiz_meta", {})

            # sécurité : on normalise encore le sous-thème en lvl1 avant sauvegarde
            sub_to_save = meta.get("subtheme")
            sub_to_save = subtheme_to_level1(sub_to_save) if sub_to_save else None

            save_attempt(
                user_id=user_id,
                user_name=name_norm,
                mode=meta.get("mode", "unknown"),
                theme=meta.get("theme"),
                theme_label=meta.get("theme_label"),
                subtheme=sub_to_save,
                n_questions=total,
                score=score,
            )
            st.session_state["score_saved"] = True
            st.toast("Score enregistré ✅")

        if wrong:
            st.write("### Récap de tes erreurs")
            for r, chosen2 in wrong:
                st.markdown(f"**Q :** {r['question_display']}")
                if chosen2 in ["A", "B", "C"]:
                    st.write(f"Ta réponse : **{chosen2}** — {answer_to_text(r, chosen2)}")
                else:
                    st.write("Ta réponse : (non renseignée)")
                st.write(
                    f"Bonne réponse : **{r['answer_letter']}** — {answer_to_text(r, r['answer_letter'])}"
                )

                justif = r.get("Document_justifiant_reponse", "")
                if isinstance(justif, str) and justif.strip():
                    st.caption(f"📌 {justif}")
                st.divider()
        else:
            st.balloons()


# ------------------------------------------------------------
# PAGE: Suivi (moyennes + évolution) par utilisateur
# ------------------------------------------------------------
if page == "Suivi":
    st.title("Suivi des scores")

    df_scores = load_attempts(user_id=user_id)
    if df_scores.empty:
        st.info("Aucun score enregistré pour le moment. Fais un QCM puis reviens ici.")
        st.stop()

    # Normalisation : si d'anciens scores avaient "1.2.1", on les affiche au niveau 1.2
    df_scores["subtheme"] = df_scores["subtheme"].apply(lambda x: subtheme_to_level1(x) if pd.notna(x) else x)

    with st.expander("Exporter mes scores"):
        csv = df_scores.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger CSV",
            data=csv,
            file_name=f"scores_{name_norm}.csv",
            mime="text/csv",
        )

    # --- SECTION 1 : Gros test (100) ---
    st.subheader("Gros test (100 questions)")
    g100 = df_scores[df_scores["mode"] == "global_100"].copy()
    if g100.empty:
        st.write("Pas encore de gros test enregistré.")
    else:
        st.metric("Moyenne", f"{g100['percent'].mean():.1f}%")
        score_chart(g100[["ts", "percent"]], title="Évolution — Gros test (100 questions)")

    st.divider()

    # --- SECTION 2 : Thèmes ---
    st.subheader("Thèmes (entraînements par thème/sous-thème)")
    theme_scores = df_scores.dropna(subset=["theme"]).copy()
    if theme_scores.empty:
        st.write("Pas encore de scores par thème/sous-thème.")
    else:
        theme_scores["theme"] = theme_scores["theme"].astype(str)
        theme_scores["theme_label"] = theme_scores["theme"].map(THEME_LABELS).fillna(
            theme_scores["theme_label"]
        )

        theme_avg = (
            theme_scores.groupby(["theme", "theme_label"], as_index=False)["percent"]
            .mean()
            .sort_values("theme", key=lambda s: s.astype(int))
        )
        st.dataframe(theme_avg, use_container_width=True, hide_index=True)

        theme_choices = theme_avg["theme"].tolist()
        t = st.selectbox(
            "Voir l'évolution d'un thème",
            theme_choices,
            format_func=lambda x: THEME_LABELS.get(str(x), f"Thème {x}"),
        )
        tdf = theme_scores[theme_scores["theme"] == str(t)].copy()
        score_chart(
            tdf[["ts", "percent"]],
            title=f"Évolution — {THEME_LABELS.get(str(t), f'Thème {t}')}",
        )

    st.divider()

    # --- SECTION 3 : Sous-thèmes (lvl1) ---
    st.subheader("Sous-thèmes")
    subs = df_scores.dropna(subset=["subtheme"]).copy()
    if subs.empty:
        st.write("Pas encore de scores par sous-thème.")
    else:
        subs["theme"] = subs["theme"].astype(str)
        subs["theme_label"] = subs["theme"].map(THEME_LABELS).fillna(subs["theme_label"])

        sub_avg = subs.groupby(["theme", "theme_label", "subtheme"], as_index=False)["percent"].mean()
        sub_avg = sub_avg.sort_values(["theme", "subtheme"], key=lambda col: col.map(sub_sort_key) if col.name == "subtheme" else col.astype(int))

        st.dataframe(sub_avg, use_container_width=True, hide_index=True)

        sub_choices = sorted(subs["subtheme"].unique(), key=sub_sort_key)
        s = st.selectbox("Voir l'évolution d'un sous-thème", sub_choices)

        sdf = subs[subs["subtheme"] == str(s)].copy()
        theme_of_sub = sdf["theme"].iloc[0] if not sdf.empty else ""
        score_chart(
            sdf[["ts", "percent"]],
            title=f"Évolution — {THEME_LABELS.get(str(theme_of_sub), f'Thème {theme_of_sub}')} — Sous-thème {s}",
        )
