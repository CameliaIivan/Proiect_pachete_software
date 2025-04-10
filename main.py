import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import geopandas as gpd
from streamlit_folium import st_folium
import folium
from shapely.geometry import Point
import seaborn as sns
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve


# Titlul aplicației
st.markdown("<h1 style='text-align: center;'>PROIECT - ANALIZA SPERANTEI DE VIATA INFANTILE</h1>",
            unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .custom-title {
        color: red !important;
        font-size: 40px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Încărcarea datelor din fișierul CSV (Date.csv)
file_path = "Date.csv"
df_raw = pd.read_csv(file_path)
# Copiem pentru a aplica curățarea pe un alt DataFrame
df = df_raw.copy()

# Curățarea datelor:
# - Înlocuim ".." cu NaN
# - Eliminăm rândurile cu valori lipsă
df.replace("..", np.nan, inplace=True)
df.dropna(inplace=True)

# Conversia coloanelor:
# Presupunem că prima coloană este "Țara" (de tip string),
# iar restul de coloane sunt indicatori numerici
numeric_columns = df.columns[1:]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df["Țara"] = df["Țara"].astype(str)

# Bara laterală pentru navigare între secțiuni
section = st.sidebar.radio("Navigați la:",
                           ["Prezentare Date", "Analiza Exploratorie", "Analiza Avansată", "Analiză Geospațială" ,"Regresie Multiplă", "Clasificare", "Funcții Suplimentare"])

#########################################
# Secțiunea: Prezentare Date
#########################################
if section == "Prezentare Date":
    st.markdown('<h1 class="custom-title">Prezentarea Datelor</h1>', unsafe_allow_html=True)

    # Afișarea dataset-ului curățat
    st.dataframe(df)
    st.subheader("Descrierea Setului de Date")
    st.write("Dimensiunea dataset-ului:", df.shape)
    num_rows = st.slider("Selectează numărul de rânduri de afișat:", min_value=5, max_value=len(df), value=5)
    st.write(f"Primele {num_rows} rânduri:")
    st.dataframe(df.head(num_rows))
    st.write("Statistici descriptive:")
    st.dataframe(df.describe())

    st.markdown("### Opțiuni de Descărcare")
    # Descărcare set raw
    csv_raw = df_raw.to_csv(index=False).encode('utf-8')
    st.download_button("Descărcați setul de date raw", data=csv_raw, file_name="dataset_raw.csv", mime="text/csv")
    # Descărcare set curățat
    csv_clean = df.to_csv(index=False).encode('utf-8')
    st.download_button("Descărcați setul de date curățat", data=csv_clean, file_name="dataset_clean.csv",
                       mime="text/csv")
    # Descărcare subset
    st.markdown("#### Descărcați un subset al datelor")
    subset_cols = st.multiselect("Selectați coloanele dorite:", options=df.columns.tolist(),
                                 default=df.columns.tolist())
    subset_rows = st.number_input("Selectați numărul de rânduri (de la început):", min_value=1, max_value=len(df),
                                  value=5)
    if subset_cols:
        df_subset = df[subset_cols].head(subset_rows)
        csv_subset = df_subset.to_csv(index=False).encode('utf-8')
        st.download_button("Descărcați subsetul de date", data=csv_subset, file_name="dataset_subset.csv",
                           mime="text/csv")
    else:
        st.write("Selectați cel puțin o coloană pentru a descărca subsetul.")

    # Vizualizare distribuție pentru o coloană numerică
    col_names = df.columns[1:]
    selected_col = st.selectbox("Selectează o coloană pentru vizualizare:", col_names)
    if selected_col:
        st.markdown(f"<h5 style='text-align: left;'>Distribuția valorilor pentru {selected_col}:</h5>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], bins=30, kde=True, ax=ax)
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Numărul țărilor")
        ax.set_title(f"Distribuția valorilor pentru {selected_col}")
        st.pyplot(fig)

        # Calcul și interpretare a asimetriei
        skew_val = df[selected_col].skew()
        st.write(f"Coeficient de asimetrie (skewness): {skew_val:.2f}")
        if abs(skew_val) < 0.5:
            interpretation = (
                "Distribuția este aproape normală, ceea ce sugerează o distribuție uniformă a indicatorului între țări. "
                "Majoritatea țărilor au valori apropiate de medie, indicând o variabilitate relativ scăzută."
            )
        elif skew_val > 0:
            interpretation = (
                "Distribuția este asimetrică spre dreapta, indicând că majoritatea țărilor au valori mai mici decât media, "
                "iar câteva țări prezintă valori semnificativ ridicate. Acest fenomen poate reflecta inegalități economice între țări."
            )
        else:
            interpretation = (
                "Distribuția este asimetrică spre stânga, indicând că majoritatea țărilor au valori ridicate, cu câteva excepții sub media generală. "
                "Acest lucru sugerează că indicatorul este, în general, favorabil, însă anumite țări se abat de la tendința generală."
            )
        st.markdown("### Interpretare Economică")
        st.write(interpretation)

#########################################
# Secțiunea: Analiza Exploratorie
#########################################
elif section == "Analiza Exploratorie":
    st.subheader("Analiza Exploratorie a Datelor")

    st.markdown("<h4 style='text-align: left;'>Matricea de corelație:</h4>", unsafe_allow_html=True)
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("<h4 style='text-align: left;'>Relația între variabile:</h4>", unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x_col = st.selectbox("Selectează variabila X:", numeric_cols)
    y_col = st.selectbox("Selectează variabila Y:", numeric_cols)
    if x_col and y_col:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### Boxplot pentru Outlier Detection")
    if x_col in numeric_cols and y_col in numeric_cols:
        subset_df = df[[x_col, y_col]]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=subset_df, ax=ax, width=0.6, linewidth=2, fliersize=5)
        ax.set_title("Boxplot pentru Outlier Detection", pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        fig.tight_layout()
        st.pyplot(fig)
        # Detectăm outlieri pentru fiecare variabilă selectată
        for col in [x_col, y_col]:
            st.markdown(f"#### Outlieri pentru {col}")
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            st.write(f"- Valoarea minimă: {lower_bound:.2f}")
            st.write(f"- Valoarea maximă: {upper_bound:.2f}")
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            st.write(f"Numărul de outlieri: {outliers.shape[0]}")
            if not outliers.empty:
                st.dataframe(outliers)
        st.markdown("""
        Interpretare Boxplot:  
        - Linia mediană reprezintă mediana valorilor.  
        - Caseta arată valorile la 25% și 75%.  
        - Mustățile se extind până la 1.5*IQR.  
        - Punctele individuale sunt outlieri.
        """)
    else:
        st.write("Selectați două variabile numerice pentru boxplot.")

#########################################
# Secțiunea: Analiza Avansata
#########################################
elif section == "Analiza Avansată":
    st.subheader("Analiza Avansata a Datelor")

    # ---- PREPROCESSARE: ENCODING ----
    st.markdown("## Preprocesare: Encoding")
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Transformăm 'Țara' în valori numerice
    le = LabelEncoder()
    df["Țara_encoded"] = le.fit_transform(df["Țara"])

    st.write("Valori unice pentru 'Țara' și codificarea lor:")
    st.dataframe(df[["Țara", "Țara_encoded"]].drop_duplicates().sort_values("Țara"))

    st.markdown("*Explicații suplimentare:*")
    st.markdown(
        "- *Encoding:* Transformă valorile text (ex. 'Romania', 'Franta') în numere (ex. 0, 1, ...), "
        "facilitând prelucrarea datelor de către algoritmii de machine learning."
    )

    # ---- PREPROCESSARE: SCALING ----
    st.markdown("## Preprocesare: Scaling")

    # Obținem lista coloanelor numerice
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    df_scaled = pd.DataFrame(scaled_data, columns=numeric_cols)

    st.write("Toate datele scalate:")
    st.dataframe(df_scaled)

    st.markdown("*Explicații suplimentare:*")
    st.markdown(
        "- *Scaling:* Standardizează valorile numerice pentru a elimina diferențele de scară, "
        "ajutând la obținerea unor rezultate mai robuste în analize precum clusteringul sau regresia."
    )

    st.markdown("---")

    # Filtrare date pe baza unei coloane numerice
    st.markdown("### Filtrare date")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_filter_col = st.selectbox("Selectați coloana numerică pentru filtrare:", numeric_cols)
        default_threshold = float(df[selected_filter_col].mean())
        threshold = st.number_input(
            f"Introduceți valoarea de filtrare pentru {selected_filter_col} (implicit: {default_threshold:.2f}):",
            value=default_threshold)
        filtered_df = df[df[selected_filter_col] > threshold]
        st.write(f"Date filtrate unde {selected_filter_col} > {threshold}:")
        st.dataframe(filtered_df)
    else:
        st.write("Nu există coloane numerice pentru filtrare.")

    # Agregări pe o coloană numerică
    st.markdown("### Agregări pe o coloană numerică")
    if numeric_cols:
        agg_col = st.selectbox("Selectați coloana numerică pentru agregare:", numeric_cols, key="agg_col")
        agg_function = st.selectbox("Selectați funcția de agregare:", ["sum", "mean", "max", "min"])
        if agg_function == "sum":
            agg_value = df[agg_col].sum()
        elif agg_function == "mean":
            agg_value = df[agg_col].mean()
        elif agg_function == "max":
            agg_value = df[agg_col].max()
        elif agg_function == "min":
            agg_value = df[agg_col].min()
        st.write(f"Valoarea {agg_function} a coloanei {agg_col}: {agg_value}")
    else:
        st.write("Nu există coloane numerice pentru agregare.")

    # Bar Chart comparativ: Top N țări după valoarea unei coloane selectate
    st.markdown("### Bar Chart comparativ")
    if "Țara" in df.columns and numeric_cols:
        top_n = st.slider("Selectați numărul de țări de afișat (top N):", min_value=5, max_value=len(df), value=10)
        sorted_df = df.sort_values(by=agg_col, ascending=False).head(top_n)
        fig, ax = plt.subplots()
        ax.bar(sorted_df["Țara"], sorted_df[agg_col])
        ax.set_title(f"Top {top_n} țări după {agg_col}")
        ax.set_xlabel("Țara")
        ax.set_ylabel(agg_col)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("Datele nu conțin coloana 'Țara' sau nu există coloane numerice pentru afișare.")

    # KMeans Clustering
    st.markdown("### KMeans Clustering")
    numeric_cols_km = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols_km) < 2:
        st.write("Nu sunt suficiente coloane numerice pentru a efectua clustering cu KMeans.")
    else:
        st.write("Selectați cele două coloane pentru clustering:")
        km_col1 = st.selectbox("Coloana X pentru clustering:", options=numeric_cols_km, key="km_col1")
        km_col2_options = [col for col in numeric_cols_km if col != km_col1]
        km_col2 = st.selectbox("Coloana Y pentru clustering:", options=km_col2_options, key="km_col2")
        X_km = df[[km_col1, km_col2]].values
        k = st.slider("Selectați numărul de clustere (K):", min_value=2, max_value=10, value=3)
        if st.button("Rulează KMeans"):
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
            y_km = kmeans.fit_predict(X_km)
            fig_km, ax_km = plt.subplots()
            colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "black", "orange", "purple", "brown"]
            for cluster in range(k):
                ax_km.scatter(X_km[y_km == cluster, 0], X_km[y_km == cluster, 1],
                              s=50, label=f"Cluster {cluster + 1}", color=colors[cluster])
            ax_km.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                          s=200, c="black", marker="X", label="Centroids")
            ax_km.set_title("KMeans Clustering")
            ax_km.set_xlabel(km_col1)
            ax_km.set_ylabel(km_col2)
            ax_km.legend()
            st.pyplot(fig_km)

            sil_score = silhouette_score(X_km, y_km)
            st.write(f"Silhouette Score: {sil_score:.4f}")
            # Calculul Silhouette Score
            # Silhouette Score măsoară cât de bine se potrivește fiecare punct în clusterul său, comparativ cu cele din alte clustere.
            # Scorul pentru fiecare punct este apoi calculat și variază între -1(incadrat gresit) și +1(bine incadrat)
        if st.checkbox("Afișează Metoda Elbow"):
            wcss = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, init="k-means++", random_state=42)
                km.fit(X_km)
                wcss.append(km.inertia_)
            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(range(1, 11), wcss, marker="o", color="red")
            ax_elbow.set_title("Metoda Elbow")
            ax_elbow.set_xlabel("Numărul de clustere")
            ax_elbow.set_ylabel("WCSS")
            st.pyplot(fig_elbow)
            # reprezintă suma distanțelor pătratice între fiecare punct și centrul (centroidul) clusterului său. Cu alte cuvinte, măsoară cât de compact sunt clusterele:
            # valori mici = punctele din fiecare cluster sunt foarte apropiate între ele = clustere compacte
            # valori mari = unctele sunt dispersate, indicând clustere mai puțin bine definite

#########################################
# Secțiunea: ANALIZA GEOSPATIALA
#########################################
elif section == "Analiză Geospațială":
    st.subheader("Analiză Geospațială")
    world = None
    # Încărcăm datasetul Natural Earth Low Resolution din GeoPandas,
    # care conține frontierele tuturor țărilor la nivel global.
    try:
        world = gpd.read_file("geodata/ne_110m_admin_0_countries.shp")
        #st.success("Shapefile încărcat cu succes!")
    except Exception as e:
        st.error(f"Eroare: {e}")
    #st.write(world.columns)

    if world is not None:
        # Pentru merge este esențial să avem o coloană comună.
        # Inspectează world.columns pentru a identifica câmpul de denumire – de obicei este "ADMIN"
        #st.write("Coloanele din shapefile:", world.columns.tolist())
        if "Țara" in df.columns:
            # Efectuăm merge-ul pe baza numelui țării
            merged = world.merge(df, left_on="ADMIN", right_on="Țara", how="left")
            #st.write("Datele au fost combinate pe baza câmpului 'ADMIN' (din shapefile) cu 'Țara' din datele tale.")

            # Selectarea indicatorului pentru harta tematică (indicator numeric din df)
            numeric_cols_df = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols_df:
                indicator = st.selectbox("Selectați indicatorul pentru harta interactivă:", numeric_cols_df)

                # Creăm o hartă folium centrată global
                m = folium.Map(location=[20, 0], zoom_start=2)
                folium.Choropleth(
                    geo_data=merged,
                    name="choropleth",
                    data=merged,
                    columns=["ADMIN", indicator],
                    key_on="feature.properties.ADMIN",
                    fill_color="OrRd",
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name=indicator
                ).add_to(m)
                folium.LayerControl().add_to(m)

                st.markdown("### Hartă interactivă")
                st_data = st_folium(m, width=700, height=500)

                # Dacă utilizatorul face click pe hartă, preluăm coordonatele click-ului
                if st_data is not None and 'last_clicked' in st_data and st_data['last_clicked'] is not None:
                    coords = st_data['last_clicked']  # Dict: {"lat": ..., "lng": ...}
                    pt = Point(coords['lng'], coords['lat'])
                    # Găsim țara care conține punctul
                    selected_country = merged[merged.contains(pt)]
                    if not selected_country.empty:
                        country_name = selected_country.iloc[0]["ADMIN"]
                        indicator_val = selected_country.iloc[0][indicator]
                        st.markdown("### Țara selectată")
                        st.write(f"Țara: **{country_name}**")
                        st.write(f"Valoarea indicatorului **{indicator}**: **{indicator_val}**")
                    else:
                        st.write("Nu s-a putut identifica nicio țară la clic.")
            else:
                st.write("Nu există indicatori numerici disponibili în datele tale.")
        else:
            st.error("Coloana 'Țara' nu există în setul tău de date.")
#########################################
# Secțiunea: REGRESIE MULTIPLA
#########################################
elif section == "Regresie Multiplă":
    st.subheader("Regresie Multipla - Modelare cu StatsModels")
    st.write("Selectați variabila dependentă și variabilele independente pentru a rula un model de regresie multiplă.")

    # Excludem variabila "Țara" din opțiuni
    cols_options = [col for col in df.columns if col != "Țara"]
    dep_var = st.selectbox("Selectați variabila dependentă:", cols_options)
    indep_vars = st.multiselect("Selectați variabilele independente:", [col for col in cols_options if col != dep_var])

    if st.button("Rulează modelul"):
        if dep_var and indep_vars:
            # Pregătim datele: adăugăm o coloană constantă
            X = df[indep_vars]
            X = sm.add_constant(X)
            y = df[dep_var]

            # Ajustăm modelul OLS
            model = sm.OLS(y, X).fit()

            # Extragem tabelul de coeficienți folosind summary2()
            summary_df = model.summary2().tables[1]
            st.markdown("#### Tabelul de Coeficienți")
            st.table(summary_df)

            # Afișăm statistici globale
            metrics = {
                "R-squared": model.rsquared,
                "Adj. R-squared": model.rsquared_adj,
                "F-statistic": model.fvalue,
                "Prob (F-statistic)": model.f_pvalue,
                "No. Observations": model.nobs
            }
            metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Valoare"])
            st.markdown("#### Statistici Globale ale Modelului")
            st.table(metrics_df)
        else:
            st.error("Vă rugăm să selectați variabila dependentă și cel puțin o variabilă independentă.")
#########################################
# Secțiunea: Funcții Suplimentare
#########################################
elif section == "Funcții Suplimentare":
    st.subheader("Funcții Suplimentare")
    func_opt = st.selectbox("Selectați funcția dorită:", [
        "Vizualizare Randuri (head/tail)",
        "Informații de bază și Statistici Descriptive",
        "Selectare Rânduri și Coloane",
        "Actualizare Valoare",
        "Slicing",
        "Operatii Group By (după prima literă a țării)"
    ])

    if func_opt == "Vizualizare Randuri (head/tail)":
        st.markdown("### Vizualizare Randuri")
        head_n = st.number_input("Număr de rânduri pentru head:", min_value=1, max_value=len(df), value=5)
        tail_n = st.number_input("Număr de rânduri pentru tail:", min_value=1, max_value=len(df), value=5, key="tail")
        st.write("Head:")
        st.dataframe(df.head(head_n))
        st.write("Tail:")
        st.dataframe(df.tail(tail_n))

    elif func_opt == "Informații de bază și Statistici Descriptive":
        st.markdown("### Informații de bază")
        # 1. Construim un DataFrame cu informațiile despre coloane
        info_data = {
            "Column": df.columns,
            "Non-Null Count": [df[col].notnull().sum() for col in df.columns],
            "Dtype": [df[col].dtype for col in df.columns]
        }
        st.table(info_data)

        # 2. Afișăm informații despre memoria utilizată
        mem_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
        st.write(f"Memory usage: {mem_usage:.2f} MB")

        # 3. Dimensiunea DataFrame-ului și numele coloanelor
        st.write("Dimensiunea dataset-ului:", df.shape)
        st.write("Coloane:", list(df.columns))

        # 4. Statistici descriptive
        st.markdown("### Statistici Descriptive")
        st.dataframe(df.describe())

    elif func_opt == "Selectare Rânduri și Coloane":
        st.markdown("### Selectare Rânduri și Coloane")
        if "Țara" in df.columns:
            selected_country = st.text_input("Introduceți numele țării pentru selectare (ex: Romania):",
                                             value="Romania")
            st.write("Rândurile pentru țara:", selected_country)
            st.dataframe(df.loc[df["Țara"].str.contains(selected_country, case=False)])
        else:
            st.write("Coloana 'Țara' nu este disponibilă în dataset.")
        col_select = st.text_input("Introduceți numele coloanei pentru selecție (ex: o coloană numerică):",
                                   value=df.columns[1])
        if col_select in df.columns:
            st.write("Selectare pentru coloana", col_select, "folosind loc:")
            st.dataframe(df.loc[:, col_select])
            st.write("Selectare folosind iloc pentru rândul 2 (index 1):")
            st.dataframe(df.iloc[1])
        else:
            st.write(f"Coloana {col_select} nu există.")

    elif func_opt == "Actualizare Valoare":
        st.markdown("### Actualizare Valoare")
        if "Țara" in df.columns:
            country_to_update = st.text_input("Introduceți numele țării pentru actualizare (ex: Romania):",
                                              value="Romania")
            col_to_update = st.selectbox("Selectați coloana numerică pentru actualizare:", df.columns[1:])
            st.write("Valoarea actuală pentru țara", country_to_update, "în coloana", col_to_update, ":")
            st.write(df.loc[df["Țara"].str.contains(country_to_update, case=False), col_to_update])
            new_value = st.number_input(f"Introduceți noua valoare pentru {col_to_update}:",
                                        value=float(df[col_to_update].mean()))
            if st.button("Actualizează valoarea"):
                df.loc[df["Țara"].str.contains(country_to_update, case=False), col_to_update] = new_value
                st.write("Valoarea actualizată:")
                st.dataframe(df.loc[df["Țara"].str.contains(country_to_update, case=False), ["Țara", col_to_update]])
        else:
            st.write("Coloana 'Țara' nu este disponibilă.")

    elif func_opt == "Slicing":
        st.markdown("### Slicing Rânduri")
        start_idx = st.number_input("Indice de start pentru slicing (loc):", min_value=0, max_value=len(df) - 1,
                                    value=1)
        end_idx = st.number_input("Indice de final pentru slicing (loc, inclusiv):", min_value=0, max_value=len(df) - 1,
                                  value=3, key="slice_end")
        st.write("Slicing folosind loc:")
        st.dataframe(df.loc[start_idx:end_idx])
        st.write("Slicing folosind iloc:")
        st.dataframe(df.iloc[start_idx:end_idx + 1])

    elif func_opt == "Operatii Group By (după prima literă a țării)":
        st.markdown("### Operatii Group By")
        if "Țara" in df.columns:
            df["Initiala"] = df["Țara"].str[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_group_col = st.selectbox("Selectați coloana numerică pentru agregare:", numeric_cols,
                                                  key="groupby_col")
                groupby_result = df.groupby("Initiala")[selected_group_col].agg(['sum', 'mean', 'max'])
                st.write("Agregări pe baza primei litere a țării:")
                st.dataframe(groupby_result)
            else:
                st.write("Nu există coloane numerice pentru agregare.")
        else:
            st.write("Coloana 'Țara' nu este disponibilă.")

elif section == "Clasificare":
    st.subheader("Curba ROC și AUC ")
    st.write(
        "Această secțiune afișează curba ROC, scorul AUC și matricea de confuzie pentru un model de clasificare (Logistic Regression).")



    # Excludem variabila "Țara" și selectăm doar coloanele numerice
    cols_options = [col for col in df.columns if col != "Țara"]
    numeric_cols = df[cols_options].select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        # Folosim prima coloană numerică pentru a crea un target binar
        target_col = numeric_cols[0]
        st.write(f"Folosim coloana *{target_col}* pentru a crea un target binar (1 dacă valoarea > medie, altfel 0).")
        y = (df[target_col] > df[target_col].mean()).astype(int)

        # Folosim două coloane numerice (dacă sunt disponibile) ca features.
        if len(numeric_cols) >= 3:
            features = numeric_cols[1:3]
        else:
            features = numeric_cols[1:]
        X = df[features]

        st.write("Feature-urile folosite:", features)

        # Împărțim datele în seturi de antrenament și test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Antrenăm modelul de Logistic Regression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        #st.write("### Curba ROC și Scorul AUC pentru Logistic Regression")


        def roc_auc_curve_plot(model, X_test, y_test):
            # Probabilități pentru modelul "No Skill" (toate predicțiile 0)
            ns_probs = [0 for _ in range(len(y_test))]
            # Probabilitățile prezise pentru clasa pozitivă
            model_probs = model.predict_proba(X_test)[:, 1]

            ns_auc = roc_auc_score(y_test, ns_probs)
            model_auc = roc_auc_score(y_test, model_probs)

            st.write("No Skill: ROC AUC = {:.3f}".format(ns_auc))
            st.write("Model: ROC AUC = {:.3f}".format(model_auc))

            ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
            model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            ax.plot(model_fpr, model_tpr, marker='.', label='Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)


        roc_auc_curve_plot(clf, X_test, y_test)

        # Calculăm predicțiile pentru matricea de confuzie
        y_pred = clf.predict(X_test)

        st.subheader("Matricea de Confuzie ")


        def conf_mtrx(y_test, y_pred, model_name):
            cm = confusion_matrix(y_test, y_pred)
            f, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            ax.set_title("Confusion Matrix - " + model_name)
            st.pyplot(f)


        conf_mtrx(y_test, y_pred, "Logistic Regression")

        st.markdown("#### Raport de clasificare:")
        st.text(classification_report(y_test, y_pred))
        st.write("Acuratețe:", accuracy_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    else:
        st.write("Nu există coloane numerice disponibile pentru evaluare ROC AUC și Confusion Matrix.")

