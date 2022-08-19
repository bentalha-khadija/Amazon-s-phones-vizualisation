#Importation
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash_bootstrap_templates import ThemeSwitchAIO
import dash.dependencies as dd
from wordcloud import WordCloud
import base64
from io import BytesIO
import warnings
import nltk
import spacy
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
nlp = spacy.load('fr_core_news_md')
warnings.filterwarnings("ignore")
nltk.download('stopwords')
stopwords = list(fr_stop)+stopwords.words('french')+["non","j'avais","c'est","ca","autres","d'autres","j'en","j'ai","jour","qu'il","faut","disant","j'en","d'a","faire","jours","oui","qu'on","tient","journée","disait","chose","ici","d'en","rien","contre","pourrez","sait","mise","d'un","trouve"]

#select the Bootstrap stylesheet2 and figure template2 for the theme toggle
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.1/dbc.min.css"
template_theme1 = "quartz"
template_theme2 = "vapor"
url_theme1 = dbc.themes.QUARTZ
url_theme2 = dbc.themes.VAPOR

#navbar
navbar = html.Div(
    dbc.Row(
        [
            dbc.Col(html.H2("Tableau De Bord"), md=2),
            dbc.Col(dbc.Input(type="search", placeholder="Search here"), md=8),
            # dbc.Col("", md=2),
            dbc.Col(
                ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2], ), md=2
            ),
        ],
    ), className="Nav"
)

#Data
df=pd.read_csv('data/dataset/data.csv')

df_copy = df.copy()
for i in df_copy.index:
        if df_copy["evaluations"][i]<=1: df_copy["evaluations"][i]=1
        elif (df_copy["evaluations"][i]>1 and df_copy["evaluations"][i]<=2) or df_copy["evaluations"][i]==1.7: df["evaluations"][i]=2
        elif df_copy["evaluations"][i]>2 and df_copy["evaluations"][i]<=3: df_copy["evaluations"][i]=3
        elif df_copy["evaluations"][i]>3 and df_copy["evaluations"][i]<=4: df_copy["evaluations"][i]=4
        else:  df_copy["evaluations"][i]=5

for i in df_copy.index:
    if df_copy["evaluations"][i]==1.7 : df_copy["evaluations"][i]=2

for i in df.index:
    if df["sentiment"][i]==0: df["sentiment"][i]='Négatif'
    else: df["sentiment"][i]='Positif'
sentiment = df.sentiment.unique()

n=len(df["titre_produit"])
tab = df.nlargest(6, ['prix'])
tab = tab.loc[:,["titre_produit","prix"]]
dataframe = pd.DataFrame(tab)

mean_df = pd.DataFrame(df['evaluations'].groupby(df['marque']).mean().map('{:.2f}'.format).reset_index(name='mean'))
std_df = pd.DataFrame(df['evaluations'].groupby(df['marque']).std().map('{:.2f}'.format).reset_index(name='std')).drop(['marque'], axis=1)
stat=pd.concat([mean_df, std_df], axis = 1).sort_values(by=['mean'], ascending=False)

df_copy1=df.copy()
df_copy1=df_copy1.drop(['sentiment'], axis=1)
df_corr = df_copy1.corr()

#Objects
markdown_text = '''
Ce tableau de bord s'inscrit dans le cadre d'un projet
du stage académique sur les données des smartphones 
et téléphones portables débloquése du AMAZON.
Ce projet vise à effectuer une analyse exploratoire 
des prix et des avis des smartphones et téléphones 
portables débloquése collectés à partir [Amazon.com](https://www.amazon.fr/s?i=electronics&rh=n%3A218193031&fs=true)
pour découvrir les tendances, les chiffres et la
relation entre les différents attributs, afin d'aider
le consommateur a prendre une décision d'achat éclairée en toute confiance.
'''
markdown = html.Div([dcc.Markdown(children=markdown_text)])
box_plot = html.Div([html.H5(children='La distribution des prix(en Euro) selon la marque du téléphone ',
                style={
                'textAlign': 'center',
                'color': 'white'}),
            dcc.Graph(id='box-plot')], className="m-4")
value = html.Div([
    html.H5(children='Nombre totale des téléphones',
            style={
                'textAlign': 'center',
                'color': 'white'}
            ),
    html.P(f"{n}",
            style={
                'textAlign': 'center',
                'color': 'purple',
                'fontSize': 30}
            ),

    ], className="card_container",
)
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], style={'marginLeft': 'auto', 'marginRight': 'auto', 'fontSize': 15,'textAlign': 'left'}, className="table")
table1 = html.Div([
    html.H5(children='Le top 10 des téléphones les plus chèrs (en Euro)',
            style={
                'textAlign': 'center',
                'color': 'white'}),
    generate_table(dataframe)
], className="card_container")
table2 = html.Div([
    html.H5(children='Top marques selon leurs moyennes evaluations',
            style={
                'textAlign': 'center',
                'color': 'white'}),
    generate_table(stat)
])
pie_chart = html.Div([html.H5(children='La répartition des évaluations des consommateurs',
                style={
                'textAlign': 'center',
                'color': 'white'}),dcc.Graph(id='pie-chart')], className="m-4")
histogram=html.Div([html.H5(children='Sentiment Analysis sur les avis des téléphones en utilisant le réseau neuronal BERT',
                style={
                'textAlign': 'center',
                'color': 'white'}),
                dcc.Graph(id='histogram')], className="m-4")
def plot_wordcloud(text):
    wc = WordCloud(background_color="white",stopwords=stopwords, max_words=90).generate(text)
    return wc.to_image()
dropdown = html.Div(dbc.Row(dbc.Col(
    dcc.Dropdown(
        id="ticker",
        options=[{"label": x, "value": x} for x in sentiment],
        value='Positif',
        clearable=False,
    ), width={"size": 4, "order": 1, "offset": 0}, )
))
img=html.Div([
    html.Img(id="image_wc"),
])
corr= html.Div([html.H5(children='La relation entre les prix et les evaluations des télephones',
            style={
            'textAlign': 'center',
            'color': 'white'}),
            dcc.Graph(id='corr')], className="m-4")
bubble_chart=html.Div([html.H5(children='La répartition des prix selon les evaluations pour chaque marque',
        style={
            'textAlign': 'center',
            'color': 'white'}),
        dcc.Graph(id='bubble-chart')], className="m-4")

#app 
app = dash.Dash(__name__,
                external_stylesheets=[url_theme2, dbc_css],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}])

#layout
app.layout = dbc.Container([
    dbc.Row(
        [
            dbc.Col(
                [
                    navbar,
                    html.Br(),
                    html.H2("Smartphones et téléphones portables débloquées -AMAZON-", style={'textAlign': 'center'}),
                ]
            )
        ]
    ),
    html.Br(),
    markdown,
    html.Hr(),
    dbc.Row(
        [dbc.Col([value, table1], md=4, sm=12,className="my_card"),
        dbc.Col(box_plot, md=6, sm=12, className="my_card"),
        ],className="justify-content-center"
    ),
    dbc.Row(
        [dbc.Col(pie_chart, md=6, sm=12, className="my_card"),
        dbc.Col(table2, md=4, sm=12, className="my_card")
        ],className="justify-content-center"
    ),
    dbc.Row(
        [dbc.Col(histogram, md=4, sm=12, className="my_card"),
        dbc.Col([html.H5(children='Word Clouds pour positifs et négatifs avis',style={'textAlign': 'center','color': 'white'}),dropdown,img], md=6,sm=12,className="my_card" )
        ],className="justify-content-center"
    ),
    dbc.Row(
        [dbc.Col(bubble_chart, md=6, sm=12, className="my_card"),
        dbc.Col(corr, md=4, sm=12, className="my_card")
        ],className="justify-content-center"
    ),
    ],
    fluid=True,
)

#callbacks
@app.callback(
    Output("box-plot", "figure"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def display_box_plot(toggle):
    template = template_theme1 if toggle else template_theme2
    fig = px.box(df, x="marque", y="prix", points="all", template=template)
    return fig

@app.callback(
    Output("pie-chart", "figure"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def display_pie_chart(toggle):
    template = template_theme1 if toggle else template_theme2
    fig = px.pie(df_copy, values='evaluations', names='evaluations',template=template)
    return fig

@app.callback(
    Output("histogram", "figure"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def display_histogram(toggle):
    template = template_theme1 if toggle else template_theme2
    fig = px.histogram(df, x="sentiment", template=template)
    fig.update_layout(bargap=0.2)
    return fig

@app.callback(dd.Output('image_wc', 'src'), [dd.Input('image_wc', 'id'), Input("ticker", "value")])
def make_image(b, feeling):
    text=''
    df_mask = df['sentiment'] == feeling
    filtered_df = df[df_mask]
    for i in filtered_df.index:
        string= str(df["avis_text"][i])
        doc = nlp(string)
        tokens = []
        for token in doc:
            tokens.append(token)
        lemmatized_sentence = " ".join([token.lemma_ for token in doc])
        text+= lemmatized_sentence
    img = BytesIO()
    plot_wordcloud(text).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(
    Output("corr", "figure"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def display_heatmap(toggle):
    template = template_theme1 if toggle else template_theme2
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = df_corr.columns,
            y = df_corr.index,
            z = np.array(df_corr),
            text=df_corr.values,
            texttemplate='%{text:.2f}'
        )
    )
    fig.update_layout(template=template)
    return fig

@app.callback(
    Output("bubble-chart", "figure"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def display_bubble(toggle):
    template = template_theme1 if toggle else template_theme2
    fig = px.scatter(df, x="prix", y="evaluations",
	size="evaluations", color="marque", template=template,
    )
    return fig

#running
if __name__ == "__main__":
    app.run_server(debug=True)