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
import zipfile
# Fonction pour charger les données une seule fois
@st.cache_data
def load_data(): # Charger le fichier CSV
    return pd.read_csv("data_cleaned.csv", index_col=0)

#endregion

#region option
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')
#endregion

#region sidebar
st.sidebar.title("Sommaire")
pages= ['👨‍💻Contexte',"🖼️ Cadre de l'analyse des données",'🧹 Nettoyage des données','📈 Data Visualisation', '🏭 Pre-processing','🤖 Modélisation','📚 Conclusion','🚘 Démo']
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
st.sidebar.title("Owner du projet :")

# Liste des membres avec leurs liens GitHub et LinkedIn
members = [
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
    
    st.markdown(
    """
    <div style="text-align: right;">
        <a href="https://datascientest.com" target="_blank">
            <img src="https://cdn.cookielaw.org/logos/08311578-d3d8-46d7-b6a2-30d80a44185a/56ecc322-2e5c-4e76-aa3b-cb09756551bc/32c1ae27-5e01-4d0e-88f0-963c0342ec9a/logo-2021.png" alt="Logo" width="250">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
    st.title("Prédictions d'émission de CO₂")
    st.divider()

    st.image('image/image_intro.webp', width = 400, use_container_width=1 )
    st.write("### Introduction")
    st.write("""Le dioxyde de carbone, communément appelé CO2, est un composant vital de notre atmosphère qui
joue un rôle essentiel dans le soutien de la vie sur Terre. Cependant, au fil des ans, les activités
humaines ont considérablement augmenté les niveaux de CO2 dans l’atmosphère, entraînant de
graves conséquences pour l’environnement. En 2022, les émissions mondiales de CO2 ont atteint un
niveau record de plus de **36,8 gigatonnes**.\n
Les émissions de CO2 ont des coûts économiques importants, et notamment en termes de santé
publique, de **dégradation de l'environnement et de changement climatique**.
En effet, les émissions de CO2 contribuent à la pollution de l'air, ce qui peut entraîner des problèmes
de santé tels que des maladies respiratoires et cardiovasculaires. Les coûts associés aux soins de
santé pour traiter ces maladies peuvent être considérables.\n
Elles contribuent aussi au changement climatique, ce qui peut entraîner des phénomènes
météorologiques extrêmes, tels que des tempêtes, des inondations et des sécheresses. Ces
événements peuvent causer des dommages importants aux infrastructures, aux cultures agricoles et
aux écosystèmes, entraînant des coûts économiques élevés.\n
Le Groupe d'experts intergouvernemental sur l'évolution du climat (GIEC) a publié des rapports
détaillant les émissions de CO2 par secteur, y compris le secteur des transports. Selon le GIEC, environ
**15% des émissions mondiales** de gaz à effet de serre proviennent directement du secteur des
transports. Parmi ces émissions, environ **70%** sont attribuées au **transport routier**, ce qui inclut les
voitures particulières.""")
    st.image('image/GES_transport.jpg', width=500)
    st.markdown(
    """
    <div style="text-align: left;">
        <p style="font-size: smaller; color: lightgrey; font-style: italic; margin-top: 5px;">
            Source: <a href="https://www.europarl.europa.eu/topics/fr/article/20190313STO31218/emissions-de-co2-des-voitures-faits-et-chiffres-infographie" target="_blank" style="color: lightgrey; text-decoration: none;">
            Parlement Européen, 17 Février 2023</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

    st.write("""
Les gouvernements et les entreprises doivent investir dans des mesures d'adaptation et de mitigation
pour faire face aux impacts du changement climatique.\n
**En identifiant les véhicules les plus polluants et en comprenant les caractéristiques techniques
responsables de la pollution, les constructeurs automobiles pourraient utiliser ces informations pour
améliorer la conception de leurs véhicules et investir dans l'innovation afin de réduire les émissions
de CO2.**\n
Un modèle de prédiction des émissions de CO2 pourrait permettre de prévoir les niveaux de pollution
pour de nouveaux types de véhicules, aidant ainsi à réduire les coûts économiques et
environnementaux associés aux émissions de CO2.""")
        
if page == pages[1]: #Cadre analyse des données
    st.header("1 - Cadre de l'analyse de données")
    st.divider()

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
        
        st.dataframe(df)
    st.markdown("""
    Les deux fichiers contiennent des informations intéressantes et utiles pour alimenter le sujet.

    La qualité du premier jeu de données est bonne car il possède très peu de NaN. Mais nous avons noté
    toutefois que ce fichier contenait des informations datant de 2014 ce qui semble trop éloigné de la
    réalité des émissions de CO2 produites par les nouvelles générations de véhicules.

    Le second fichier df_eu contient une colonne intitulée «country» et nous permet de constater que
    les informations relatives à la France sont inclus dans ce fichier. Les informations sont relatives à
    l’année 2023, cela nous permet donc d’avoir des données plus récentes.
    La récence des données, la multiplicité des types de véhicule et la mesure utilisée pour les émissions
    de CO2 contenu dans le fichier df_eu sont des paramètres qui nous sont apparus plus pertinent pour
    notre sujet.\n
    **Le jeu de données conservé pour cette étude est donc le fichier df_eu.**
    """)

if page == pages[2]: #Nettoyage des données
    st.header('2 - Étapes du nettoyage de données :')
    st.divider()

#region étape nettoyage
    with st.expander('2.1 Suppression des colonnes vides'): #2.1
        st.markdown(""" 
Nous avons dans un premier temps supprimé toutes les **colonnes vides**, colonnes qui contenaient
toujours les mêmes valeurs, ou les colonnes qui n'apportent aucune information en rapport avec
l’objet de notre étude.

Nous avons donc supprimé les colonnes : MMS, r , Enedc (g/km), W (mm), At1 (mm), At2 (mm),
Ernedc (g/km), De, Vf, Status, year, Date of registration. """)
        st.code("""#Suppression des colonnes inutiles pour le modèle
drop_col = ['Country','VFN','Mp','Mh','Man','Tan','T','Va','Ve','Mk','Cn','IT','ech','RLFI']
""")
        
    with st.expander('2.2 Suppression des doublons avant nettoyage'): #2.2
        st.markdown("""Nous avons par la suite identifié et **supprimé les doublons** sur les colonnes restantes. Cela nous a
permis de réduire drastiquement la taille de notre jeu de données.""")
        df = pd.DataFrame({'Nombre colonne':[39,27],'Nombre ligne':['10 734 656','1 804 116'],'Taille du fichier (Mo)':['2 450 Mo','388 Mo']}, index=['Tableau avant supression','Tableau après suppression'])
        df
        st.code("""df_eu = df_eu.drop(columns=drop_col)
                """)
        
    with st.expander('2.3 Suppression des colonnes inutiles pour le modèle de Machine Learning'): #2.3
        st.markdown("""Nous avons ensuite réalisé un tableau permettant d’identifier les colonnes qui nous seraient utiles ou non pour exploiter notre jeu de données.\n
Pour rappel, l’objectif étant in fine d’établir un modèle permettant à un constructeur de renseigner les caractéristiques techniques d’un futur véhicule, et que celui-ci lui 
retourne une prédiction sur les émissions de CO2, nous nous sommes, pour chaque colonne poser la question :\n
<span style="color:red;"><i>Est-ce que l’information dans cette colonne est pertinente pour prédire l’émission de CO2?</i></span>
""",
    unsafe_allow_html=True
)  
        df=pd.read_excel('tableau.xlsx', sheet_name='tableau_pertinence')
        st.dataframe(df)
        st.markdown("""Le résultat de cette étape a permis passer d’un jeu de données de **27 colonnes** à un jeu de données de **13 colonnes**.
""")
        st.code("""#Suppression des colonnes inutiles pour le modèle
drop_col = ['Country','VFN','Mp','Mh','Man','Tan','T','Va','Ve','Mk','Cn','IT','ech','RLFI']
df_eu = df_eu.drop(columns=drop_col)""")
        
    with st.expander("2.4.1 Création de colonnes 'intermédiaires'"): #2.4.1
        st.markdown("""
                    Nous décidons de créer une nouvelle colonnes qui combine l'information des colonnes <span style="color : red;"><i>Fuel Type (Ft)</i></span> et <span style="color : red;"><i>Fuel Mode (Fm)</i></span>"""
                    ,unsafe_allow_html=True)
        df=pd.read_excel('tableau.xlsx', sheet_name='fuel_type')
        st.dataframe(df)
        
        st.code("""
               
               #On convertit les valeurs de Ft en minuscule
df_eu['Ft'] = df_eu['Ft'].str.lower()

#Création d'une nouvelle colonne plus détaillées pour le type de moteur
#qui synthétise Ft (fuel type) et Fm (fuel mode)
#Au final il y a donc 10 catégories
df_eu['fuel_type'] = ''
df_eu.loc[(df_eu['Ft']=='petrol')&(df_eu['Fm']=='H'), 'fuel_type'] = 'PHNR' 
df_eu.loc[(df_eu['Ft']=='petrol')&(df_eu['Fm']=='M'), 'fuel_type'] = 'P' 
df_eu.loc[df_eu['Ft']=='petrol/electric', 'fuel_type'] ='PHR' 
df_eu.loc[(df_eu['Ft']=='diesel')&(df_eu['Fm']=='H'), 'fuel_type'] = 'DHNR' 
df_eu.loc[(df_eu['Ft']=='diesel')&(df_eu['Fm']=='M'), 'fuel_type'] = 'D' 
df_eu.loc[df_eu['Ft']=='diesel/electric', 'fuel_type'] ='DHR' 
df_eu.loc[df_eu['Ft']=='hydrogen', 'fuel_type'] = 'H' 
df_eu.loc[df_eu['Ft']=='lpg', 'fuel_type'] = 'GPL' 
df_eu.loc[(df_eu['Ft']=='e85')&(df_eu['Fm']=='H'), 'fuel_type'] = 'E85NR' 
df_eu.loc[(df_eu['Ft']=='e85')&(df_eu['Fm']=='F'), 'fuel_type'] = 'E85F'
df_eu.loc[df_eu['Ft']=='ng', 'fuel_type'] = 'GN' """
)
    
    with st.expander('2.4.2 Les catégories de véhicules'):#2.4.2
        st.markdown("""Nous avons deux colonnes <span style="color : red;"><i>Cr</i></span> et <span style="color : red;"><i>Ct</i></span> qui nous informe sur les catégories des véhicules suivantes :
""", unsafe_allow_html=True)
        df=pd.read_excel('tableau.xlsx', sheet_name='categorie_vehicule')
        st.dataframe(df)
        st.markdown("""La colonne Ct, contient les 6 catégories, mais contient **1109 valeurs manquantes**.
Nous décidons donc de conserver la colonne Ct – qui semble plus fiable – et dans les cas où les valeurs sont manquantes, nous lui **attribuons la valeur qui était contenu dans Cr**.
""")
        st.code("""df_eu['Ct'] = df_eu['Ct'].fillna(df_eu['Cr'])
""")
        st.markdown("""Étant donné le faible volume de véhicules des catégories N1,N1G, N2, N2G (254 véhicules) nous décidons de **supprimer** les lignes pour lesquelles Ct inclut une de ces quatre catégories.
 
De plus, nous supprimons Cr qui ne nous servira plus.
""")
        st.code("""#On supprime les valeurs peu fréquentes, à savoir N1, N1G, N2 et N2G
df_eu = df_eu.drop(df_eu[df_eu['Ct'] =='N1'].index)
df_eu = df_eu.drop(df_eu[df_eu['Ct'] =='N1G'].index)
df_eu = df_eu.drop(df_eu[df_eu['Ct'] =='N2'].index)
df_eu = df_eu.drop(df_eu[df_eu['Ct'] =='N2G'].index)

#Suppression de Cr
df_eu = df_eu.drop(columns=['Cr'])""")
    
    with st.expander('2.5 Suppression de lignes et de colonnes jugées non-pertinentes pour le modèle de machine learning') : #2.5
        st.markdown("""L’étape décrite en 2.4.1 nous a permis d’identifier des véhicules qui n’émettent pas de CO2.
Cela représente les véhicules qui ont un moteur à **hydrogène** ou **électrique**. 
Nous décidons donc de supprimer tous les véhicules qui ont un de ces deux types de moteur.\n
Nous décidons également de supprimer les colonnes <span style="color : red;"><i>Ft</i></span> et <span style="color : red;"><i>Fm</i></span> qui ne nous serviront plus.
""", unsafe_allow_html=True)
        st.code("""#Suppression des véhicules qui n'émettent pas de CO2. 
df_eu = df_eu.drop(df_eu[df_eu['Ft'] =='electric'].index)
df_eu = df_eu.drop(df_eu[df_eu['Ft'] =='hydrogen'].index)
df_eu = df_eu.drop(df_eu[df_eu['Ft'] =='unknown'].index)

#Suppression des colonnes Ft Fm,
df_eu = df_eu.drop(columns=['Ft','Fm'])""")
        
    with st.expander('2.6 Remplacement des valeurs manquantes pour les véhicules non-hybrides'): #2.6
        st.markdown("""Les colonnes <span style="color : red;"><i>z (Wh/km)</i></span>, <span style="color : red;"><i>Electric range (km)</i></span> sont des colonnes qui donnent des informations sur la consommation électrique et l’autonomie en mode électrique des véhicules hybrides.
 
Nous considérons que ce sont des **données intéressantes** pour notre modèle de Machine Learning.

Toutefois, pour les véhicules non-hybrides, ces valeurs sont manquantes. Nous décidons de les remplacer par des 0, car ces véhicules n’ont ni consommation électrique, ni autonomie en mode électrique, et cela ne faussera pas la performance du modèle.
 
La colonne <span style="color : red;"><i>Erwltp (g/km)</i></span> correspond à la réduction d’émission de CO2 mise en place par **certains constructeurs**.
Les véhicules ne disposant pas d’une telle technologie ont une valeur manquante dans cette colonne.
Nous décidons également de remplacer ces valeurs manquantes par des 0.
""", unsafe_allow_html=True)
        st.code("""#On remplace les NaN de la consommation électrique par des 0
df_eu['z (Wh/km)'] = df_eu['z (Wh/km)'].fillna(0)

#On remplace les NaN de la réduction d'émission de CO2 par des 0
df_eu['Erwltp (g/km)'] = df_eu['Erwltp (g/km)'].fillna(0)

#On remplace les NaN de l'autonomie électrique par des 0
df_eu['Electric range (km)'] = df_eu['Electric range (km)'].fillna(0)""")
        
    with st.expander('2.7 Suppression des lignes dont la variable cible est manquante'): #2.7
        st.markdown("""Nous supprimons les lignes dont la variable cible <span style="color : red;"><i>Ewltp (g/km)</i></span> n'est pas renseignée. 
""",unsafe_allow_html=True)
        st.code("""#Suppression des lignes où la variable cible est manquante
df_eu = df_eu.dropna(subset=['Ewltp (g/km)'])""")
    
    with st.expander("""2.8 Réduction de dimension pour la variable cible"""): #2.8
        st.markdown("""Après ces premières étapes, nous remarquons que certains véhicules ont des émissions de CO2 relativement faibles (< 10 g/km).

Après vérification de certains modèles sur Internet, nous doutons de la fiabilité des données contenues dans la colonne <span style="color : red;"><i>Ewltp (g/km)</i></span>.

Nous décidons donc de fixer un seuil de gramme de CO2 émis par km, en dessous duquel il nous paraît très improbable que l’information soit fiable.
 
Ce seuil a été fixé à 15 g/km
""",unsafe_allow_html=True)
        st.code("""#Suppression des lignes dans lesquelles la variable cible est inférieur 
#à un seuil (ici inférieur ou égal à 15)
df_eu = df_eu.drop(df_eu[df_eu['Ewltp (g/km)'] <=15].index)""")
     
    with st.expander('2.9 Identification des problèmes de multi-colinéarité'): #2.9
        st.markdown("""Nous souhaitons maintenant nous intéresser au problème de multicolinéarité.""")
        st.image('image/heatmap.png')
        st.markdown("""Cette matrice, nous permet d’identifier deux multi-colinéarités :
- Entre <span style="color : red;"><i>Mt</i></span> – le poids total d’un véhicule en conditions de test – et <span style="color : red;"><i>m (kg)</i></span>, la Masse (𝑟 = 0.99)
- Entre <span style="color : red;"><i>Electric range (km)</i></span> et <span style="color : red;"><i>z (Wh/km)</i></span> (𝑟 = 0.93)""", unsafe_allow_html=True)
        st.code("""
                #Suite à l'analyse de la heatmap, on décide de supprimer la colonne Mt, 
#qui correspondait au 'poids total d'un véhicule dans les conditions de test'
#car cette colonne contenait plus de NaN que la colonne m (kg)
df_eu = df_eu.drop(columns='Mt')

#Suite à l'analyse de la heatmap, on décide de supprimer la colonne 'z (Wh/km)' 
#qui correspondait à la consommation électrique
df_eu = df_eu.drop(columns='z (Wh/km)')""")
        
    with st.expander('2.10 Suppression des doublons après nettoyage'): #2.10
        st.markdown("""Une fois toutes ces étapes de nettoyage finalisées, nous supprimons une deuxième fois les doublons dans le jeu de données restant.
""")
        st.code("""#On réenlève les derniers doublons
df_eu.drop_duplicates(inplace=True)""")
        df=pd.read_excel('tableau.xlsx', sheet_name='duplicate_delete')
        st.dataframe(df)
#endregion

if page == pages[3]:  #Data Visualisation
    
    # Charger les données une seule fois
    data_cleaned = load_data()
    st.header("3 - Modélisation")
    st.divider()

    st.subheader("3.1 Distribution de la variable cible : Emissions CO2 (g/km)")
    st.write("""
    La distribution de la variable cible nous permet de visualiser comment les émissions de CO2 (en g/km) sont réparties dans notre jeu de données.
    Cela peut nous aider à identifier des tendances générales, des valeurs extrêmes ou des comportements inhabituels dans les données.
    """)
    # Créer le graphique de distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data_cleaned['Ewltp (g/km)'], kde=True, bins=30, color='blue', edgecolor='black')
    plt.title('Distribution des émissions de CO2 (g/km)', fontsize=16)
    plt.xlabel('Emissions CO2 (g/km)', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
    
    # Affichage de la heatmap
    st.subheader("3.2 Heatmap des corrélations entre les variables numériques :")
    st.write("""
    Cette heatmap permet d'analyser les corrélations entre les différentes variables numériques de notre jeu de données. 
    Cela nous permet de comprendre comment les variables se lient entre elles et de mieux saisir l'impact de certaines caractéristiques sur les émissions de CO2.
    """)
    
    # Sélectionner les colonnes numériques
    data_numeric = data_cleaned[['m (kg)', 'Ewltp (g/km)', 'ec (cm3)', 'ep (KW)', 'Fuel consumption ']]
    
    # Calculer la matrice de corrélation
    correlation_matrix = data_numeric.corr()
    
    # Afficher la heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de corrélation des variables numériques')
    st.pyplot(plt)  # Affiche la heatmap dans Streamlit
    st.subheader("3.3 Analyse de l'impact des paramètres sur les émissions de CO2")
    st.write("""
    Les graphiques ci-dessous montrent la relation entre les émissions de CO2 et différents paramètres des véhicules, ceci sera affiché toujours en fonction des types de carburants. 
    L'objectif est de visualiser l'impact de la masse, de la capacité du moteur, de sa puissance et de la consommation de carburant 
    sur ces émissions CO2 afin de mieux comprendre comment ces caractéristiques influencent l'empreinte écologique des véhicules.
    """)
    # Dictionnaire pour associer les colonnes aux noms plus clairs uniquement pour l'affichage
    colonnes_renommees = {
        'm (kg)': 'Masse du véhicule (kg)',
        'ec (cm3)': 'Capacité moteur (cm3)',
        'ep (KW)': 'Puissance moteur (KW)',
        'Fuel consumption ': 'Consommation de carburant (L/100km)',
        'Ewltp (g/km)': 'Emissions CO2 (g/km)'  # Emissions CO2 est l'axe Y fixe
    }
    # Liste des colonnes disponibles pour l'axe X
    colonnes_disponibles = ['m (kg)', 'ec (cm3)', 'ep (KW)', 'Fuel consumption ']
    # Renommer les colonnes disponibles en affichage lisible (utilisation du dictionnaire)
    colonnes_disponibles_renommees = [colonnes_renommees[col] for col in colonnes_disponibles]
    # Liste déroulante pour choisir la colonne pour l'axe X avec les noms clairs
    axe_x_choisi = st.selectbox('Choisissez la colonne pour l\'axe X', colonnes_disponibles_renommees)
    # On récupère la colonne d'origine en fonction du choix fait par l'utilisateur
    axe_x = [col for col, name in colonnes_renommees.items() if name == axe_x_choisi][0]
    # L'axe Y est fixe sur 'Ewltp (g/km)' pour les émissions CO2
    axe_y = 'Ewltp (g/km)'
    axe_y_renomme = colonnes_renommees[axe_y]  # Nom affiché pour l'axe Y
    # Créer le graphique de relation entre les colonnes choisies
    st.write("Voici le graphique :")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=axe_x, y=axe_y, hue='fuel_type', data=data_cleaned)
    plt.title(f'Relation entre la variable {axe_x_choisi} et les émissions CO2 en g/km')
    plt.xlabel(axe_x_choisi)
    plt.ylabel(axe_y_renomme)
    plt.legend(title='Type de carburant')
    st.pyplot(plt)  # Affiche le graphique dans Streamlit
    # Texte explicatif pour le scatterplot
    if axe_x == 'm (kg)':  # Masse du véhicule
        st.markdown("""
        *Ce graphique montre la relation entre la masse du véhicule (en kg) et les émissions de CO2 (en g/km).* 
        *Les véhicules plus lourds ont tendance à produire davantage d’émissions de CO2.* 
        *Nous pouvons supposer que cela s’explique par le fait que les véhicules plus lourds nécessitent plus d'énergie pour être déplacés, ce qui entraîne une combustion accrue de carburant.*
        *La masse d’un véhicule est un facteur clé influençant les émissions de CO2, mais l’impact varie selon le type de carburant utilisé.*
        """)
    elif axe_x == 'ec (cm3)':  # Capa moteur
        st.markdown("""
        *Ce graphique montre la relation entre la capacité du moteur (en cm³) et les émissions de CO2.* 
        *Les véhicules avec un moteur plus gros (plus de cm³) émettent davantage de CO2, car ces moteurs nécessitent plus de carburant pour fonctionner,* 
        *ce qui entraîne une combustion plus importante et une production accrue de gaz à effet de serre, notamment du dioxyde de carbone.*
        """)
    elif axe_x == 'ep (KW)':  # Puissance du moteur
        st.markdown("""
        *Ce graphique montre la relation entre la puissance du moteur (en kW) et les émissions de CO2.* 
        *Une puissance plus élevée est associée à des émissions de CO2 plus élevées, car un moteur plus puissant consomme davantage de carburant pour générer cette énergie supplémentaire,* 
        *ce qui entraîne une augmentation des rejets de gaz à effet de serre dans l'atmosphère.*
        """)
    elif axe_x == 'Fuel consumption ':  # Consommation de carburant
        st.markdown("""
        *Ce graphique montre la relation entre la consommation de carburant (en litres aux 100 km) et les émissions de CO2.* 
        *Nous observons un lien entre la consommation de carburant et les émissions de CO2. Lorsque la consommation augmente, les émissions tendent également à augmenter.* 
        *Nous pouvons supposer qu’une plus grande consommation entraîne une combustion accrue de carburant, entraînant plus d'émissions de CO2.*
        """)

if page == pages[4]: #Pre-processing
    st.header('4 - Pré-processing')
    st.divider()
    st.subheader("4.1 Création du jeu d'entraînement et du jeu de test")
    st.markdown("""Une fois le jeu de données bien nettoyé, nous créons nos jeux d'entraînement et de test.
Le jeu d'entraînement (X_train, y_train) représente 80% des données.
Le jeu d'entraînement (X_test, y_test) représente 20% des données.""")
    st.code("""feats = df_eu.drop(columns='Ewltp (g/km)')
target = df_eu['Ewltp (g/km)']
X_train, X_test, y_train, y_test = train_test_split(
    feats, target, random_state = 41, test_size = 0.2)""")
    
    st.subheader("4.2 Gestion des valeurs manquantes")
    st.markdown("""Pour les valeurs manquantes des colonnes m (kg), ec (cm3), ep (KW) et Fuel consumption, nous choisissons de les remplacer par la médiane, 
qui est résistante aux valeurs extrêmes contenues dans certaines colonnes.""")
    st.code("""#Gestion des valeurs manquantes
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
num_col_NA = ['m (kg)','ec (cm3)','ep (KW)','Fuel consumption ']
num_col = ['m (kg)','ec (cm3)','ep (KW)','Fuel consumption ','Electric range (km)','Erwltp (g/km)']

X_train.loc[:,num_col_NA] = imputer.fit_transform(X_train[num_col_NA])
X_test.loc[:,num_col_NA] = imputer.transform(X_test[num_col_NA])""")
    
    st.subheader("4.3  Encodage ")
    st.markdown("""Pour l’encodage des variables catégorielles Ct et fuel_type, nous optons pour OneHotEncoder, car ces variables ne présentent pas de 
relation d'ordre entre leurs différentes catégories.""")
    st.code("""#Encodage des variables catégorielles
oneh = OneHotEncoder(drop = 'first', sparse_output=False)
cat_col = ['Ct','fuel_type']
X_train_encoded = oneh.fit_transform(X_train[cat_col])
X_test_encoded = oneh.transform(X_test[cat_col])

#Conversion en DataFrame
noms_colonnes_cat = oneh.get_feature_names_out(cat_col)
X_train_encoded = pd.DataFrame(X_train_encoded, columns=noms_colonnes_cat, index = X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=noms_colonnes_cat, index = X_test.index)""")
    
    st.subheader("4.4  Standardisation")
    st.markdown("""Enfin, pour la Standardisation, nous utilisons StandardScaler pour uniformiser les valeurs 
des variables numériques afin de pouvoir entraîner des modèles de familles différentes.""")
    st.code("""#Standardisation des variables numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_col])
X_test_scaled = scaler.transform(X_test[num_col])

#Conversion en DataFrame
noms_colonnes_num = scaler.get_feature_names_out(num_col)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=noms_colonnes_num, index = X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=noms_colonnes_num, index = X_test.index)""")
    
    st.subheader("4.5 Reconstition de DataFrame")
    st.markdown("""Enfin, nous reconstituons le DataFrame pour la modélisation.""")
    st.code("""
#Reconstition du tableau après encodage
X_train.drop(columns=cat_col)
X_test.drop(columns=cat_col)
X_train = pd.concat([X_train_encoded,X_train_scaled], axis = 1)
X_test = pd.concat([X_test_encoded,X_test_scaled], axis = 1)""")

if page == pages[5]: #Modélisation
    #Titre de la page
    st.header("5 - Modélisation")
    st.divider()
    #region regression
    st.subheader("5.1 Modèles de régression")
    
    #Option pour les modèles et les métriques
    regression_model_option = ["Régression Linéaire","Arbre de décision","Forêt aléatoire","XGB Regressor"]
    regression_metric_option = ["Score Train","R²","MAE","MSE","RMSE"]

    #Sélection par l'utilisateur du contenu à afficher
    model_selection = st.pills("Modèle(s)", regression_model_option, selection_mode="multi")
    metric_selection = st.pills("Métrique(s)", regression_metric_option, selection_mode="multi")
    
    #Chargement du fichier
    df = pd.read_excel('tableau.xlsx', sheet_name='regression')

    if model_selection and metric_selection:
        # Filtrer les modèles choisis
        df_filtered = df[df["Modèle"].isin(model_selection)]

        # Garder uniquement les colonnes sélectionnées
        columns_to_display = ["Modèle"] + metric_selection
        df_display = df_filtered[columns_to_display]

        # Arrondir les résultats numériques à deux chiffres après la virgule
        df_display = df_display.round(2)

        # Afficher le tableau filtré
        st.write("### Résultats :")
        st.dataframe(df_display, use_container_width=True)

    st.subheader("5.2 Optimisation du modèle de Régression")
    st.markdown("""GridSearchCv est une technique utilisée en apprentissage automatique pour optimiser les hyper paramètres d'un modèle. Nous allons donc chercher les meilleurs hyperparamètres permettant d’améliorer les performances de notre modèle Random Forest.

Cela nous donne les résultats suivants : """)
    st.code("""Meilleurs paramètres : {
    'max_depth': 30, #Profondeur maximale des arbres
    'max_features': 'sqrt', #Combien de caractéristiques sont prises en compte à chaque noeud
    'min_samples_leaf': 1, #Nombre minimal d'échantillon dans une feuille terminal
    'min_samples_split': 10, #Nombre minimal d'échantillon pour la division d'un noeud
    'n_estimators': 300 #Nombre d'arbres dans la forêt
    }
""")
    st.markdown("""Cette optimisation permet d'améliorer le modèle :""")
    df_opti = df.loc[(df['MSE']==198.22)|(df['MSE']==173.70)]
    st.dataframe(df_opti, use_container_width=True)
    
    st.markdown("""Nous avons ensuite étudié les résidus, c'est à dire la différence entre les valeurs réelles et les valeurs prédites. 
                Plus une valeur est proche de l'axe centrale, plus le modèle l'a bien prédite. """)
    st.image('image/residu.png')
    st.markdown("""On observe une ligne au niveau du 112 g/km qui pose problème. 
                Nous revenons donc au niveau du nettoyage des données et nous supprimons les lignes pour lequelles la colonne Ewltp (g/km) vaut 112.""")
    st.code("df_eu = df_eu.drop(df_eu[df_eu['Ewltp (g/km)']==112].index)")
    st.markdown("""De plus, on remarque que beaucoup de valeurs 'faibles' sont mals prédites. Ainsi, nous augmentons le seuil (fixé précédemment à 15g/km) à 35g/km. """)
    st.code("""df_eu = df_eu.drop(df_eu[df_eu['Ewltp (g/km)'] <=35].index)""")
    
    st.markdown("A la fin de toutes ces étapes, nous pouvons observer le résultat final avec les différentes améliorations.")
    df_opti2 = df.loc[(df['MSE']==198.22)|(df['MSE']==173.70)|(df['MSE']==79.78)]
    st.dataframe(df_opti2, use_container_width=True)
    #endregion
    
    
if page == pages[7]: #Démo
    #region Titre formulaire
    st.markdown(f"""
              <style>
              .title {{
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
              <div class="title"> Prédiction des émissions de CO2 </div>
              """,unsafe_allow_html=True)
    #endregion
    st.divider()

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
        
        zip_path = "model.zip"
        with zipfile.ZipFile(zip_path,'r') as zip_ref:
            zip_ref.extractall()
        model_path= 'model'
        random_forest = joblib.load(model_path)
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
    st.image('image/etiquette_CO2.png')
    #endregion
