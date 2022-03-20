# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go


app = dash.Dash(__name__)
server = app.server

# Patrick
df_merged_quality = pd.read_csv("Input/merged_data_cleaned.csv")
df_merged_quality.rename(columns={"Unnamed: 0": "Id"}, inplace=True)
df_merged_quality.set_index("Id", inplace=True)

# -----Arabica-----
df_origin_arabica = df_merged_quality[df_merged_quality["Species"] == "Arabica"]
df_origin_arabica = df_origin_arabica[df_origin_arabica["Country.of.Origin"].notna()]
df_origin_arabica.groupby(["Country.of.Origin"]).count()
df_origin_arabica_count = df_origin_arabica.groupby(["Country.of.Origin"]).count()
df_origin_arabica_count["Number of Company"] = df_origin_arabica_count["Species"]
df_origin_arabica_count["Types"] = "Arabica"
df_origin_arabica_count = df_origin_arabica_count[["Number of Company", "Types"]].sort_values("Number of Company",
                                                                                              ascending=False)
df_origin_arabica_count = df_origin_arabica_count.iloc[:10, :]

# -----Robusta-----
df_origin_robusta = df_merged_quality[df_merged_quality["Species"] == "Robusta"]
df_origin_robusta = df_origin_robusta[df_origin_robusta["Country.of.Origin"].notna()]
df_origin_robusta_count = df_origin_robusta.groupby(["Country.of.Origin"]).count()
df_origin_robusta_count["Number of Company"] = df_origin_robusta_count["Species"]
df_origin_robusta_count["Types"] = "Robusta"
df_origin_robusta_count = df_origin_robusta_count[["Number of Company", "Types"]].sort_values("Number of Company",
                                                                                              ascending=False)
# ---- Stack 2 dataframes
df_ctry_origin_company_count = pd.concat([df_origin_arabica_count, df_origin_robusta_count])

# ---- Production
df_production_number = pd.read_csv("input/total-production.csv")
df_production_number_2018 = df_production_number[["total_production", "2018"]]
df_production_number_2018 = df_production_number_2018.sort_values(by=["2018"], ascending=False)

fig2 = px.bar(df_production_number_2018[:10], y="2018", x="total_production", color="total_production")
fig2.update_xaxes(tickangle=90)

df_consumption = pd.read_csv("Input/psd_coffee.csv")
df_consumption = df_consumption[df_consumption["Market_Year"] == 2018]
df_consumption = df_consumption[df_consumption["Attribute_Description"] == "Domestic Consumption"]
df_consumption = df_consumption[df_consumption["Value"] > 2000]
df_consumption = df_consumption.sort_values("Value", ascending=False)
fig6 = px.bar(df_consumption, y="Value", x="Country_Name", color="Country_Name")
fig6.update_xaxes(tickangle=90)
# ------Exports
df_exports = pd.read_csv("Input/exports-calendar-year.csv")
df_exports.rename(columns={"exports": "country"}, inplace=True)

df_exports = df_exports.melt("country")
df_continents = pd.read_csv('Input/Continents.csv')
df_continents = df_continents[["Entity", "Continent"]]
df_exports = df_exports.join(df_continents.set_index('Entity'), on='country')
df_exports = df_exports.dropna()

list_continents = list(df_exports["Continent"].unique())
list_year = list(df_exports.iloc[:, 1:-2].columns)
df_exports = df_exports.rename(columns={"variable": "year", "value": "export"})
df_exports = df_exports.sort_values(["country", "year"])
# Import data
data = pd.read_csv("Input/directory.csv")

Starbucks = data[data['Brand'] == 'Starbucks']['Country'].value_counts().to_frame()
Starbucks['id'] = Starbucks.index

isoCode = pd.read_csv("Input/countries_codes_and_coordinates.csv")

Country = isoCode.merge(Starbucks, left_on='A2', right_on='id')
Country = Country.rename(columns={"Country": "Starbucks"})

pop = pd.read_csv("Input/population_by_country_2020.csv", delimiter=',')

df = pop.merge(Country, left_on='Country (or dependency)', right_on='Name')
# Extract data
Urb = df["Urban Pop %"]
df["U"] = Urb
U = []
for i in range(len(Urb)):
    if str(Urb[i])[0] != 'N':
        U.append(int(str(Urb[i])[:2]))
    else:
        U.append(49)

df['U'] = U
df['U Pop'] = df["Population (2020)"] * df['U'] / 100
df['r'] = df['U Pop'] / df['Starbucks']
df['Popularity'] = 20 + np.log(df['Starbucks'] / df['U Pop'])

df1 = df[(df['U Pop'] > 1000000) & (df['Starbucks'] > 100) & (df['U'] > 49)]
fig4 = px.scatter(df1, x='Popularity', y='Starbucks', color='U Pop',
                  size='U', hover_name="Name",
                  # color_continuous_scale=px.colors.sequential.YlOrBr,
                  title='Starbucks Popularity in big country')

# Layout
df2 = df[(df['U'] > 0)]

# Should I buy an Espresso Machine

# Edit the layout

# Layout

fig5 = px.choropleth(df2, locations="A3",
                     color="U",  # lifeExp is a column of gapminder
                     hover_name="Name",  # column to add to hover information
                     color_continuous_scale=px.colors.sequential.Plasma)

# ----Taste
ind = np.arange(19, 31)
l = [2]
for i in ind:
    l.append(i)
df_origin_arabica_taste = df_origin_arabica.iloc[:, l]
df_origin_arabica_taste_avg = df_origin_arabica_taste.groupby(["Country.of.Origin"]).mean()
df_origin_arabica_taste_avg = df_origin_arabica_taste_avg.iloc[:, :-3]
df_origin_arabica_taste_avg = df_origin_arabica_taste_avg.loc[["Brazil", "India", "Japan", "Vietnam"]]
theta = ["Aroma", "Flavor", "Aftertase", "Acidity", "Sweetness"]

###
cpp = pd.read_csv("Input/coffeePriceParis.csv")
cpp = cpp.groupby(["Arrondissement"]).mean()
cpp['A'] = cpp.index

figcpp = px.bar(cpp.sort_values(by=['Price']), x='Price', y='A')

app.layout = html.Div(children=[

    html.Div([
        html.H2('Types of companies and number of companies that produce it around the world'),
        html.H3('''This chart is a small representation of the types coffee and in which country they are produce, \n
                    but as we will see in the next charts, the number of companies that produces coffee isn't a good \n
                    representation of the coffee market and production '''),

        dcc.Dropdown(
            df_ctry_origin_company_count['Types'].unique(),
            "Robusta",
            id='coffee_type',

        ),

        dcc.Graph(
            id="coffe-graphic"
        )

    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.H2('''Coffee production'''),
        html.H3('''
        Knowing the number of company by  country produces which type of coffee is cool,\n
         but what is more interesting is knowing the total number of coffee by country (bags of coffee 1000kg)
         '''),
        dcc.Graph(id='production_coffee',
                  figure=fig2)
    ]),

    html.H2('Exports of coffee per continent and country'),
    html.H3(''' As we saw in the previous chart, the country that produces the most coffee is Brazil and consequently, 
    the country that export the most coffee across all continents is ... BRAZIL
    '''),
    html.H4('''We decided to filter the chart by country and continent to see who is the most exporting country in 
    each continent and overall
    '''),
    dcc.Graph(id="export-graph"),
    dcc.Checklist(
        id="checklist",
        options=list_continents,
        value=["Asia"],
        inline=True
    ),
    html.Div([
        html.H2('''Coffee consumption around the world'''),
        html.H3('''After finding out which country produces the more coffee, 
            we can see that only one of them is included in the consumption chart (10 most)
             but the country that drink the most coffee are USA and the EU
             '''),
        html.H4('''( EU isn't a country but the consumption is doubled so we can imagine how much they like coffee =) 
        '''),
        dcc.Graph(id='consumption_coffe',
                  figure=fig6)
    ]),
    html.Div([
        html.H2('''Taste of coffee in 4 different countries '''),
        html.H3('''We wanted to know what makes brazilian coffee so special, by comparing coffee taste on different 
    criteria, overall the brazilian coffee is perfectly while maximizing everything that makes the coffee so good 
    '''),
        dcc.Dropdown(
            df_origin_arabica_taste_avg.index,
            "Brazil",

            id='coffee_taste',

        ),

        dcc.Graph(
            id="taste-graphic"
        )

    ], style={'width': '48%', 'display': 'inline-block'}),
    html.H2(children='Starbucks Coffee Exploration'),
    html.H3('''We decided to dive into some small details about coffee and talk about one of the most loved coffee 
    shop in the world : Starbucks'''),


    dcc.Graph(
        id='Figure 1',
        figure=fig4
    ),
    dcc.Graph(
        id='The Coffee Map',
        figure=fig5
    ),


    dcc.Graph(
        id='Price of Espresso in Paris',
        figure=figcpp
    ),
    html.H2("Buy an Espresso machine or Coffee at a shop ?"),
    html.H3("After analyzing some of the boring stuff about coffee we wanted to show a statistical feature that will "
            "be useful for coffee lovers; In short, if you don't intend to drink coffee everyday the machine is "
            "useless, but if you do drink coffee everyday whatever the cost of the machine, at some point buying "
            "coffee at a shop everyday will be much more expensive "),
    dcc.Graph(
        id='Should I buy an Espresso Machine?',
    ),
    html.H6("The number of Espresso you drink per week"),
    dcc.Slider(
        0,
        20,
        step=1,
        value=2,
        id='Frequency'
    ),
    html.H6("The price of your Machine"),
    dcc.Slider(
        0,
        2000,
        step=1,
        value=600,
        id='Price'
    )
])


@app.callback(
    Output('coffe-graphic', 'figure'),
    Input('coffee_type', 'value'))
def update_graph(coffee_type):
    df_ctry_origin_company_count_countries=df_ctry_origin_company_count[df_ctry_origin_company_count["Types"] == coffee_type]
    fig=px.bar(df_ctry_origin_company_count_countries,x=df_ctry_origin_company_count_countries.index,y="Number of Company",color="Types")
    return fig


@app.callback(
        Output("export-graph", "figure"),
        Input("checklist", "value"))
def update_line_chart(continents):
    mask = df_exports["Continent"].isin(continents)
    fig = px.line(df_exports[mask],
                  x="year", y="export", color='country')
    return fig

# Taste Graphic
@app.callback(
    Output('taste-graphic', 'figure'),
    Input('coffee_taste', 'value'))
def update_graph(coffee_taste):
    df_origin_arabica_taste_avg_ctry = df_origin_arabica_taste_avg[df_origin_arabica_taste_avg.index == coffee_taste]
    fig = go.Figure(data=go.Scatterpolar(
        r=df_origin_arabica_taste_avg_ctry.loc[coffee_taste].tolist(),
        theta=theta,
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
        ),
        showlegend=False
    )
    return fig

@app.callback(
    Output('Should I buy an Espresso Machine?', 'figure'),
    Input('Frequency', 'value'),
    Input('Price', 'value'))
def update_cpp(Frequency,Price):
    save_amount = 2.5
    price = Price
    maintenance = 0.1
    warranty = 365
    drink_rate = Frequency/7
    x = np.linspace(1, warranty, warranty)
    x1 = x * drink_rate
    y1 = save_amount * x1
    y2 = price + x1 * maintenance
    y3 = y1 - y2
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=x, y=y1, name='Money to buy Espresso at Store',
                           line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=x, y=y2, name='Price of Equipment and maintenance',
                           line=dict(color='firebrick', width=4)))

# Edit the layout
    fig.update_layout(title='Should you buy an Espresso Machine? Check the Warranty!',
                    xaxis_title='Days',
                    yaxis_title='Euro')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
