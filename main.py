import os
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px

DATA_PATH_RETAIL = os.path.join(os.getcwd(), 'online_retail_II.xlsx')
MAX_SAMPLES = 1_000_000

def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext not in ['.xls', '.xlsx']:
        raise ValueError(f"Unsupported file type: {ext}")

    df = pd.read_excel(path)

    df = df.rename(columns={
        'Customer ID': 'userId',
        'StockCode': 'itemId',
        'InvoiceDate': 'timestamp',
        'Quantity': 'Quantity',
        'Price': 'Price',
        'Description': 'Description',
        'Country': 'Country'
    })

    df = df.dropna(subset=['userId', 'itemId'])
    # Normalize types
    df['userId'] = df['userId'].astype(float).astype(int).astype(str)
    df['itemId'] = df['itemId'].astype(str)

    df = df[df['Quantity'] > 0]

    df['rating'] = df['Quantity'].clip(lower=1, upper=5).astype(float)

    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)

    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    return df.reset_index(drop=True)

def compute_basic_stats(df: pd.DataFrame) -> dict:
    return {
        'num_samples': len(df),
        'num_users': df['userId'].nunique(),
        'num_items': df['itemId'].nunique(),
        'mean_rating': df['rating'].mean()
    }


def create_eda_figures(df: pd.DataFrame) -> dict:
    rating_counts = df['rating'].value_counts().sort_index()
    fig_rating_dist = px.bar(x=rating_counts.index, y=rating_counts.values,
                              labels={'x': 'Rating', 'y': 'Count'},
                              title='Rating Distribution', template='plotly_dark')
    fig_rating_dist.update_traces(marker_color='teal', marker_line_color='navy', marker_line_width=1.5)

    top_users = df['userId'].value_counts().nlargest(10)
    fig_top_users = px.bar(x=top_users.values, y=top_users.index,
                             orientation='h', labels={'x': 'Purchases', 'y': 'User ID'},
                             title='Top 10 Users by Purchases', template='ggplot2')
    fig_top_users.update_traces(marker_color='orchid', marker_line_width=1)

    top_items = df['itemId'].value_counts().nlargest(10)
    fig_top_items = px.bar(x=top_items.values, y=top_items.index,
                             orientation='h', labels={'x': 'Purchases', 'y': 'Item ID'},
                             title='Top 10 Items by Purchases', template='seaborn')
    fig_top_items.update_traces(marker_color='olive', marker_line_color='black', marker_line_width=1)

    time_df = df.groupby('year_month').agg(total_ratings=('rating', 'count')).reset_index()
    fig_time_count = px.line(time_df, x='year_month', y='total_ratings',
                              labels={'year_month': 'Year-Month', 'total_ratings': 'Total Ratings'},
                              title='Total Ratings Over Time', template='plotly')
    fig_time_count.update_traces(mode='lines+markers', line=dict(dash='dashdot', width=2), marker=dict(symbol='square', size=8))

    time_avg_df = df.groupby('year_month').agg(average_rating=('rating', 'mean')).reset_index()
    fig_time_avg = px.area(time_avg_df, x='year_month', y='average_rating',
                            labels={'year_month': 'Year-Month', 'average_rating': 'Average Rating'},
                            title='Average Rating Over Time', template='presentation')
    fig_time_avg.update_traces(line=dict(color='teal', width=2), fillcolor='lightcyan', opacity=0.6)

    return {
        'rating_dist': fig_rating_dist,
        'top_users': fig_top_users,
        'top_items': fig_top_items,
        'time_count': fig_time_count,
        'time_avg': fig_time_avg
    }

def train_recommender(df: pd.DataFrame, n_factors: int = 50) -> tuple:
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userId', 'itemId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=n_factors, random_state=42)
    algo.fit(trainset)
    preds = algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)
    return algo, rmse, mae

def create_dash_app(stats, figs, algo, df):
    app = dash.Dash(__name__)
    user_ids = df['userId'].unique()

    app.layout = html.Div([
        html.H1("Recommender System Dashboard", style={'textAlign': 'center'}),
        # EDA section
        html.Section([
            html.H2("Exploratory Analysis"),
            html.Div([
                html.Div(dcc.Graph(figure=figs['rating_dist']), style={'gridColumn': '1'}),
                html.Div(dcc.Graph(figure=figs['top_users']), style={'gridColumn': '2'}),
                html.Div(dcc.Graph(figure=figs['top_items']), style={'gridColumn': '1'}),
                html.Div(dcc.Graph(figure=figs['time_count']), style={'gridColumn': '2'}),
                html.Div(dcc.Graph(figure=figs['time_avg']), style={'gridColumn': '1 / span 2'})
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'})
        ], style={'marginTop': '20px'}),
        # Purchase history section
        html.Section([
            html.H2("Purchase History for Customer ID"),
            dcc.Input(id='user-input', type='text', placeholder='Enter Customer ID...', style={'marginRight': '10px'}),
            html.Button('Show History', id='btn-history', n_clicks=0),
            html.Div(id='history-output', style={'marginTop': '20px'})
        ], style={'marginTop': '40px', 'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '8px'})
    ], style={'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px'})

    @app.callback(
        Output('history-output', 'children'),
        Input('btn-history', 'n_clicks'),
        State('user-input', 'value')
    )
    def show_history(n_clicks, user_val):
        if n_clicks and user_val:
            uid = str(user_val)
            if uid not in user_ids:
                return html.P(f"Customer ID {uid} not found.", style={'color': 'red'})
            user_df = df[df['userId'] == uid]
            # Build table header
            header = [html.Tr([html.Th(col) for col in ['InvoiceDate', 'Price', 'Country', 'Quantity', 'StockCode', 'Description']])]
            # Build rows
            rows = []
            for _, row in user_df.sort_values('date', ascending=False).iterrows():
                rows.append(html.Tr([
                    html.Td(row['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(row['timestamp']) else ''),
                    html.Td(f"{row['Price']:.2f}"),
                    html.Td(row['Country']),
                    html.Td(int(row['Quantity'])),
                    html.Td(row['itemId']),
                    html.Td(row['Description'])
                ]))
            table = html.Table(header + rows, style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'})
            return table
        return ''

    return app

if __name__ == '__main__':
    df = load_data(DATA_PATH_RETAIL)
    algo, rmse, mae = train_recommender(df)
    figs = create_eda_figures(df)
    app = create_dash_app({}, figs, algo, df)
    app.run(debug=True)
