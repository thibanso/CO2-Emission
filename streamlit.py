#region librairies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
#endregion

#region option
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')
#endregion

#region sidebar
st.sidebar.title("Sommaire")
pages= ['👨‍💻Contexte',"🖼️ Cadre de l'analyse des données",'🧹 Nettoyage des données','📈 Data Visualisation', '🏭 Pre-processing','🤖 Modélisation','📚 Conclusion']
page=st.sidebar.radio('Allez vers', pages)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #7AA95C; /* Couleur de fond personnalisée */
    }
    </style>
    """,
    unsafe_allow_html=True
)

#region members
# Sidebar title
st.sidebar.title("Membres du projet :")

# Liste des membres avec leurs liens GitHub et LinkedIn
members = [
    {"name": "Antoine BARBIER",
    "github": "https://github.com/Antoine-DA",
    "linkedin": "https://www.linkedin.com/in/antoine-barbier-83654415b/"},
    {"name": "Flora BREN",
    "github": "https://github.com/bobmartin",
    "linkedin": "https://www.linkedin.com/in/flora-b-68a80013a"},
    {"name": "Thibault EL MANSOURI",
    "github": "https://github.com/thibanso",
    "linkedin": "https://www.linkedin.com/in/el-mansouri-299932130/"},
]

# Générer la liste des membres dans la sidebar
for member in members:
    st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <div style="flex: 1;">{member['name']}</div>
        <a href="{member['github']}" target="_blank" style="margin-right: 8px;">
            <img src="https://cdn-icons-png.flaticon.com/32/25/25231.png" width="20" alt="GitHub">
        </a>
        <a href="{member['linkedin']}" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/32/174/174857.png" width="20" alt="LinkedIn">
        </a>
    </div>
    """, unsafe_allow_html=True)
#endregion
#endregion

if page == pages[0]: #Contexte
    st.title("Projet prédictions d'émission de CO2")
    st.write("### Introduction")
    st.write("""Le dioxyde de carbone, communément appelé CO2, est un composant vital de notre atmosphère qui
joue un rôle essentiel dans le soutien de la vie sur Terre. Cependant, au fil des ans, les activités
humaines ont considérablement augmenté les niveaux de CO2 dans l’atmosphère, entraînant de
graves conséquences pour l’environnement. En 2022, les émissions mondiales de CO2 ont atteint un
niveau record de plus de 36,8 gigatonnes.\n
Les émissions de CO2 ont des coûts économiques importants, et notamment en termes de santé
publique, de dégradation de l'environnement et de changement climatique.
En effet, les émissions de CO2 contribuent à la pollution de l'air, ce qui peut entraîner des problèmes
de santé tels que des maladies respiratoires et cardiovasculaires. Les coûts associés aux soins de
santé pour traiter ces maladies peuvent être considérables.\n
Elles contribuent aussi au changement climatique, ce qui peut entraîner des phénomènes
météorologiques extrêmes, tels que des tempêtes, des inondations et des sécheresses. Ces
événements peuvent causer des dommages importants aux infrastructures, aux cultures agricoles et
aux écosystèmes, entraînant des coûts économiques élevés.\n
Certains secteurs économiques, tels que l'agriculture, le tourisme et l'énergie, sont particulièrement
vulnérables aux impacts du changement climatique. Par exemple, les changements dans les régimes
de précipitations peuvent affecter les rendements agricoles, tandis que les destinations touristiques
peuvent être impactées par des conditions météorologiques extrêmes.\n
Le Groupe d'experts intergouvernemental sur l'évolution du climat (GIEC) a publié des rapports
détaillant les émissions de CO2 par secteur, y compris le secteur des transports. Selon le GIEC, environ
15% des émissions mondiales de gaz à effet de serre proviennent directement du secteur des
transports. Parmi ces émissions, environ 70% sont attribuées au transport routier, ce qui inclut les
voitures particulières.\n
Les gouvernements et les entreprises doivent investir dans des mesures d'adaptation et de mitigation
pour faire face aux impacts du changement climatique.\n
En identifiant les véhicules les plus polluants et en comprenant les caractéristiques techniques
responsables de la pollution, les constructeurs automobiles pourraient utiliser ces informations pour
améliorer la conception de leurs véhicules et investir dans l'innovation afin de réduire les émissions
de CO2.\n
Un modèle de prédiction des émissions de CO2 pourrait permettre de prévoir les niveaux de pollution
pour de nouveaux types de véhicules, aidant ainsi à réduire les coûts économiques et
environnementaux associés aux émissions de CO2.""")
        
if page == pages[1]: #Cadre analyse des données
    st.header("Cadre de l'analyse de données")
    url = "https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/"
    with st.expander('1.1 - Premier jeu de données'):
        st.markdown("""
        Le premier fichier porte sur les émissions de CO2, de polluants et les caractéristiques des véhicules 
        commercialisés en France relatif à l’année 2014. Ce fichier est disponible librement sur le site de
        [data.gouv.fr](%s) et s’intitule «mars-2014-complete.csv».
        
        Il est accompagné d’un dictionnaire des variables permettant d’avoir la définition des abréviations et
        les unités de mesures utilisées. Toutefois la seule lecture des légendes est apparue insuffisante afin
        de comprendre certaines caractéristiques techniques des véhicules, il a donc été procédé à des
        recherches complémentaires afin d’avoir une lecture optimale du jeu de données.
        
        Il a été renommé df_fr pour la première partie d’observation du fichier.
        
        Ce premier jeu de données comporte 55 044 lignes et 30 colonnes.
        Il contient les informations ci-dessous:
        """ % url)
        df=pd.read_excel('tableau.xlsx', sheet_name='df_fr')
        st.dataframe(df)
    url = "https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b"
    with st.expander('2.2 - Deuxième jeu de données'):
        st.markdown("""
        Un second jeu de données porte sur les émissions de CO2 des véhicules produits en Europe relatif à
        l’année 2023. Ce fichier est disponible librement sur le site [European Environment Agency3](%s) et
        s’intitule «data.csv».
        
        Il est accompagné d’un dictionnaire des variables permettant d’avoir la définition des abréviations et
        les unités de mesures utilisées. Sur le même principe que le premier jeu de données, ce dictionnaire a
        été complété par des définitions afin d’avoir une meilleure lecture du jeu de données.
        
        Il a été renommé df_eu pour la première partie d’observation du fichier.

        Il contient les informations suivantes :""" % url)
        df = pd.read_excel('tableau.xlsx', sheet_name='df_eu')
        st.dataframe(df)
    st.markdown("""
    Les deux fichiers contiennent des informations intéressantes et utiles pour alimenter le sujet.

    La qualité du premier jeu de données est bonne car il possède très peu de NaN. Mais nous avons noté
    toutefois que ce fichier contenait des informations datant de 2014 ce qui semble trop éloigné de la
    réalité des émissions de CO2 produites par les nouvelles générations de véhicules.

    Le second fichier df_eu contient une colonne intitulée «country» et nous permet de constater que
    les informations relatives à la France sont inclus dans ce fichier. Les informations sont relatives à
    l’année 2023, cela nous permet donc d’avoir des données plus récentes.
    Nous avons noté également la présence également de deux colonnes relatives à la mesure des
    émissions de CO2 :
    
    -Enedc (g/km) Specific CO2 Emissions (NEDC).\n
    -Ewltp (g/km) Specific CO2 Emissions (WLTP).
    
    En 2017, la norme de mesure des émissions de CO2 (NEDC4) est devenue obsolète et a été remplacée
    par la norme (WLTP5). Cette norme offre une mesure plus précise et réaliste des émissions de CO2 et
    de la consommation de carburant des véhicules, reflétant mieux les conditions de conduite réelles.
    C’est une donnée importante pour notre étude dont la variable cible est l’émission de CO2.
    
    La récence des données, la multiplicité des types de véhicule et la mesure utilisée pour les émissions
    de CO2 contenu dans le fichier df_eu sont des paramètres qui nous sont apparus plus pertinent pour
    notre sujet.\n
    Le jeu de données conservé pour cette étude est donc le fichier df_eu.
    """)

if page == pages[2]: #Nettoyage des données
    with st.expander('2.1 Suppression des colonnes vides'):
        st.markdown(""" 
Nous avons dans un premier temps supprimé toutes les colonnes vides, colonnes qui contenaient
toujours les mêmes valeurs, ou les colonnes qui n'apportent aucune information en rapport avec
l’objet de notre étude.

Nous avons donc supprimé les colonnes : MMS, r , Enedc (g/km), W (mm), At1 (mm), At2 (mm),
Ernedc (g/km), De, Vf, Status, year, Date of registration. """)
        st.code("""#Suppression des colonnes inutiles pour le modèle
drop_col = ['Country','VFN','Mp','Mh','Man','Tan','T','Va','Ve','Mk','Cn','IT','ech','RLFI']
""")
        
    with st.expander('2.2 Suppression des doublons avant nettoyage'):
        st.markdown("""Nous avons par la suite identifié et supprimé les doublons sur les colonnes restantes. Cela nous a
permis de réduire drastiquement la taille de notre jeu de données.""")
        df = pd.DataFrame({'Nombre colonne':[39,27],'Nombre ligne':['10 734 656','1 804 116'],'Taille du fichier (Mo)':['2 450 Mo','388 Mo']}, index=['Tableau avant supression','Tableau après suppression'])
        df
        st.code("""df_eu = df_eu.drop(columns=drop_col)
                """)

if page == pages[5]: #Modélisation
    #region Titre formulaire
    st.markdown(f"""
              <style>
              .centered {{
                  display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 12vh;
                    font-size: 40px;
                    font-weight: bold;
                    background-color: #7AA95C; 
                    color: white;  
                    border-radius: 10px;
                    padding: 20px;
              }}
              </style>
              <div class="centered"> Prédiction des émissions de CO2 </div>
              """,unsafe_allow_html=True)
    #endregion
    
    #region Formulaire
    with st.form(key='prediction_form'):
        #category = st.selectbox("Catégorie de véhicule",('Tourisme','Utilitaire'),index=None,placeholder="Selectionner la catégorie de véhicule")
        category = st.segmented_control(label= 'Catégorie', options=['Tourisme','Utilitaire'])
        masse = st.number_input("Poids du véhicule (en Kg)",min_value=500.0, max_value= 5000.0, value = 1500.0, step=50.0)
        #fuel = st.selectbox("Motorisation",['Essence','Diesel','Essence hybride non-rechargeable','Essence hybride rechargeable','Diesel hybride rechargeable','Diesel hybride rechargeable','GPL','Gaz naturel','E85 non-rechargeable','E85 FlexiFuel'], index=None, placeholder='Sélectionner la motorisation')
        fuel = st.segmented_control(label="Motorisation",options=['Essence','Diesel','Essence hybride non-rechargeable','Essence hybride rechargeable','Diesel hybride rechargeable','Diesel hybride rechargeable','GPL','Gaz naturel','E85 non-rechargeable','E85 FlexiFuel'])

        autonomie_electrique = st.number_input("Autonomie électrique (km)", min_value=0.0, max_value=700.0, value=0.0, step=5.0)
        capacite_moteur = st.number_input("Capacité du moteur (cm3)",min_value=500.0, max_value=8000.0,value=1500.0,step=50.0)
        puiss_moteur = st.number_input("Puissance du moteur(KW)", min_value=5.0, max_value=1200.0, value=110.0, step=10.0)
        reduc_emission = st.number_input("Technologie de réduction d'émission (g/km)",min_value=0.0, max_value=10.0, value=0.0, step = 0.1)
        consommation = st.number_input("Consommation du moteur (l/100)",min_value=0.0, max_value=30.0, value=6.0, step=0.1)
        
        submit_button = st.form_submit_button(label="Prédire")
    #endregion
    
    #region encodage, chargement du fichier
    df = pd.read_csv('data_cleaned.csv', index_col=0)
    X = df.drop(columns='Ewltp (g/km)')
    y = df['Ewltp (g/km)'] 

    #Différenciation des colonnes numériques et catégorielles 
    num_col_NA = ['m (kg)','ec (cm3)','ep (KW)','Fuel consumption ']
    num_col = ['m (kg)','ec (cm3)','ep (KW)','Fuel consumption ','Electric range (km)','Erwltp (g/km)']
    cat_col = ['Ct','fuel_type']

        
    #Gestion des valeurs manquantes
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X.loc[:,num_col_NA] = imputer.fit_transform(X[num_col_NA])

    #Encodage des variables catégorielles
    oneh = OneHotEncoder(drop = 'first', sparse_output=False)
    X_encoded = oneh.fit_transform(X[cat_col])

    #Conversion en DataFrame
    noms_colonnes_cat = oneh.get_feature_names_out(cat_col)
    X_encoded = pd.DataFrame(X_encoded, columns=noms_colonnes_cat, index = X.index)

    #Standardisation des variables numériques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[num_col])

    #Conversion en DataFrame
    noms_colonnes_num = scaler.get_feature_names_out(num_col)
    X_scaled = pd.DataFrame(X_scaled, columns=noms_colonnes_num, index = X.index)

    #Reconstition du tableau après encodage
    X.drop(columns=cat_col)
    X = pd.concat([X_encoded,X_scaled], axis = 1)
    #endregion

    #region prédictions 
    if submit_button:
        # Transformation des données pour correspondre aux colonnes du modèle
        data = {
            "Ct_M1G": 1 if category == "Utilitaire" else 0,  # Exemple de transformation binaire
            "fuel_type_DHNR": 1 if fuel == "Diesel hybride non-rechargeable" else 0,
            "fuel_type_DHR": 1 if fuel == "Diesel hybride rechargeable" else 0,
            "fuel_type_E85F": 1 if fuel == "E85 FlexiFuel" else 0,
            "fuel_type_E85NR": 1 if fuel == "E85 non-rechargeable" else 0,
            "fuel_type_GN": 1 if fuel == "Gaz naturel" else 0,
            "fuel_type_GPL": 1 if fuel == "GPL" else 0,
            "fuel_type_P": 1 if fuel == "Essence" else 0,
            "fuel_type_PHNR": 1 if fuel == "Essence hybride non-rechargeable" else 0,
            "fuel_type_PHR": 1 if fuel == "Essence hybride rechargeable" else 0,
            "m (kg)": masse,
            "ec (cm3)": capacite_moteur,
            "ep (KW)": puiss_moteur,
            "Fuel consumption ": consommation,
            "Electric range (km)": autonomie_electrique,
            "Erwltp (g/km)": reduc_emission}
        
        input_df = pd.DataFrame([data])
        input_df[num_col]=scaler.transform(input_df[num_col])
        random_forest = joblib.load('model')
        pred = random_forest.predict(input_df)

     #region couleur résultat
        if pred <= 101:
            color = '#048f18'
        elif pred <= 121:
            color = '#6cb71e'
        elif pred <= 141:
            color = '#c0d717'
        elif pred <= 161:
            color = '#fff11c'
        elif pred <= 201:
            color = '#f8bd19'
        elif pred <= 251:
            color = '#f0750d'
        else:
            color = '#d2000b'
    #endregion
        st.markdown(
                f"""
                <style>
                .centered {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 10vh;
                    font-size: 50px;
                    font-weight: bold;
                    background-color: {color}; 
                    color: black;  
                    border-radius: 10px;
                    padding: 20px;  /* pour ajouter du padding autour du texte */
                }}
                </style>
                <div class="centered">{np.round(pred[0], 1)} g / km</div>
                """,
                unsafe_allow_html=True
            )

    #endregion