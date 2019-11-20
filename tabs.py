import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import dash_table
from tab1.view import import_tab1
from tab4.view import import_tab4
from tab2.view import import_tab2
from tab3.view import import_tab3
from tab5.view import import_tab5
from dash.dependencies import Input, Output, State
import numpy as np
import pickle


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
yelp = pd.read_csv('yelp_data_arizona_state_topcity_restaurant_sampling_FINAL_DATA.csv')
df_business = pd.read_csv('yelp_business_arizona_topcity.csv')

f = open('yelp_recommendation_model_5.pkl', 'rb')
P, Q, userid_vectorizer = pickle.load(f), pickle.load(f), pickle.load(f)

import string
from nltk.corpus import stopwords
stop = []
for word in stopwords.words('english'):
    s = [char for char in word if char not in string.punctuation]
    stop.append(''.join(s))

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return " ".join([word for word in nopunc.split() if word.lower() not in stop])

global prediksi_rest
prediksi_rest = []

def hasil_prediksi(city,words):
    
    test_df= pd.DataFrame([words], columns=['text'])
    test_df['text'] = test_df['text'].apply(text_process)
    test_vectors = userid_vectorizer.transform(test_df['text'])
    test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())

    predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])
    topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])
    topRecommendations=topRecommendations.merge(df_business[['business_id','city']], on='business_id')
    if city=='All':
        topRecommendations = topRecommendations[:5]
    else:
        topRecommendations=topRecommendations[topRecommendations['city']==city].reset_index().drop('index',axis=1)[:5]

    hasil=''
    for i in topRecommendations['business_id']:
        hasil += str(df_business[df_business['business_id']==i]['name'].iloc[0]) + '\n\n'
        hasil += str(df_business[df_business['business_id']==i]['categories'].iloc[0]) + '\n'
        hasil += 'Restaurant Ratings: ' + str(df_business[df_business['business_id']==i]['stars'].iloc[0])+ '\n'
        hasil += 'Total Ratings Count: ' + str(df_business[df_business['business_id']==i]['review_count'].iloc[0]) + '\n'
        hasil += '\n\n'
        hasil += '\n\n'
    return hasil

def restaurant_desc(city,rest):
    yelp_rest = yelp[yelp['name']==rest]
    yelp_rest = yelp_rest[yelp_rest['city']==city]
    b_id = yelp_rest['business_id'].iloc[0]
    output = ''
    yelp_b = df_business[df_business['business_id']==b_id]
    output += yelp_b['name'].iloc[0] + '\n'
    output += ' ' + '\n'
    output += ' ' + '\n'
    output += 'Categories: ' + yelp_b['categories'].iloc[0] + '\n'
    output += ' ' + '\n'
    output += 'Address: ' + yelp_b['address'].iloc[0] + '\n'
    output += ' ' + '\n'
    output += 'City: ' + yelp_b['city'].iloc[0] + '\n'
    output += ' ' + '\n'
    output += 'Restaurant Ratings: ' + str(yelp_b['stars'].iloc[0]) + '\n'
    output += 'Total Review Count: ' + str(yelp_b['review_count'].iloc[0]) + '\n'
    output += ' ' + '\n'
    output += ' ' + '\n'
    output += 'Top Review: ' + '\n'
    output +=  yelp_rest.sort_values(by=['stars','useful'],ascending=False)['text'].iloc[0] + '\n'
    output += ' ' + '\n'
    output += ' ' + '\n'
    output += 'Worst Review: ' + '\n'
    output +=  yelp_rest.sort_values(by=['stars','useful'],ascending=True)['text'].iloc[0] + '\n'
    return output



def all_city():
    a = [{'label' : i, 'value' : i} for i in df_business['city'].unique()]
    a.append({'label' : 'All', 'value' : 'All'})
    return a

def all_rest_name():
    b = [{'label' : i, 'value' : i} for i in df_business['name'].unique()]
    b.append({'label' : 'All', 'value' : 'All'})
    return b


app.layout = html.Div(children = [
    html.H1('Yelp Arizona Restaurant Recommendation'),
    html.P('Created by: Theo Jeremiah'),
    dcc.Tabs(value = 'tabs', id = 'tabs-1', children = [

        # dcc.Tab(label = 'Exploring Arizona Restaurant Data', id = 'tab-satu', children = [
        # html.Div(children = [
        #     html.P('City:'),
        #     dcc.Dropdown(id = 'x-axis-1',
        #     options = all_city(),
        #     value = 'All')                
        #     ],className = 'col-3'),

        # html.Div(children = [
        #     html.P('Max Rows: '),
        #     dcc.Input(
        #     id='x-axis-2',
        #     type='number',
        #     value = 10,
        #     placeholder='Input number'),                
        #     ],className = 'col-3'),

        # html.Br(),
        # html.Div(html.Button('Search'), id = 'search2', className = 'col-3'),

        # html.Br(),

        # html.Div([
        #     dash_table.DataTable(
        #         id='table',
        #         columns=[{"name": i, "id": i} for i in df_business.columns],
        #         data=df_business.to_dict('records'),
        #         page_action = "native",
        #         page_current = 0,
        #         page_size = 10,
        #     )
        #     ])
        # ]),

        #tab1
        dcc.Tab(label = 'Restaurant Recommendation', id = 'tab-dua', children = [
        html.Div(children = [
            html.P('City:'),
            dcc.Dropdown(id = 'x-axis-3',
            options = all_city(),
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Input a Sentence:'),
            dcc.Input(
            id='input-text',
            type='text',
            placeholder='input text',)                
            ],className = 'col-6'),

        html.Br(),
        html.Div(html.Button('Search'), id = 'search3', className = 'col-3'),

        html.Br(),

        html.Div(id = 'predict', children= [
            html.H1('Fill in the Parameters')
            ],style={'text-align':'center'})


        ]),#penutup tab1

        #tab 2
        dcc.Tab(label = 'Restaurant Detailed Review', id = 'tab-tiga', children = [

        html.Div(children = [
            html.P('City:'),
            dcc.Dropdown(id = 'x-axis-4',
            options = all_city(),
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Restaurant:'),
            dcc.Dropdown(id = 'x-axis-5',
            options = all_rest_name(),
            value = 'kosong')                
            ],className = 'col-3'),

        html.Br(),
        html.Div(html.Button('Search'), id = 'search4', className = 'col-3'),

        html.Br(),
        html.Br(),
        html.Div(id = 'predict2', children= [
            html.H1('Fill in the Parameters')
            ],style={'text-align':'left'})





        ]), #penutup tab2

    ],
    content_style = {
        'fontFamily' : 'Arial',
        'borderBottom' : '1px solid #d6d6d6',
        'borderLeft' : '1px solid #d6d6d6',
        'borderRight' : '1px solid #d6d6d6',
        'padding' : '44px'
    })
],style={'maxWidth': '1200px', 'margin': '0 auto'})

# @app.callback(
#     [Output(component_id = 'table', component_property = 'data'),
#     Output(component_id = 'table', component_property = 'page_size')],
#     [Input(component_id = 'search2', component_property = 'n_clicks')],
#     [State(component_id = 'x-axis-1', component_property = 'value'),
#     State(component_id = 'x-axis-2', component_property = 'value')]
# )

# def create_data_frame(n_clicks, x1, x2):
#     if x1 == 'All':
#         data = df_business.to_dict('records')
#     else:
#         data = df_business[df_business['city'] == x1].to_dict('records')
#     page_size = x2
#     return data, page_size

@app.callback(
    [Output(component_id = 'predict', component_property = 'children')],
    [Input(component_id = 'search3', component_property = 'n_clicks')],
    [State(component_id = 'x-axis-3', component_property = 'value'),
    State(component_id = 'input-text', component_property = 'value')]
)

def create_predictions(n_clicks, x1, x2):
    if x1=='kosong':
        children = [
            html.H1('Fill in the Parameters')
        ]
    else:
        children = [
            dcc.Markdown(
            hasil_prediksi(x1,x2),
            style={"white-space": "pre"}),
                ]
    return children

@app.callback(
    [Output(component_id = 'predict2', component_property = 'children')],
    [Input(component_id = 'search4', component_property = 'n_clicks')],
    [State(component_id = 'x-axis-4', component_property = 'value'),
    State(component_id = 'x-axis-5', component_property = 'value')]
)

def create_predictions2(n_clicks, x1, x2):
    if x1=='kosong':
        children = [
            html.H1('Fill in the Parameters')
        ]
    else:
        children = [
            # html.H1(x2),
            dcc.Markdown(
                restaurant_desc(x1,x2),
                style={'marginLeft': 10, 'marginRight': 10, 'marginTop': 10, 
               'backgroundColor':'#F7FBFE',
               'border': 'thin lightgrey dashed', 'padding': '6px 0px 0px 8px'}),
                ]
    return children

if __name__ == '__main__':
    app.run_server(debug=True)