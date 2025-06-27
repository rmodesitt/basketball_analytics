import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# Baseketball web app title and description
st.title('NBA Player Stats Explorer')

st.markdown('''
This app performs simple webscraping of NBA player stats data and provides analysis
on key basketball insights!
\n\n**Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
'''
)

# Sidebar - Header
st.sidebar.header('Input features')

# Sidebar - Year
years_list = list(reversed(range(2015, 2026)))
selected_year = st.sidebar.selectbox('Year', years_list)

# Web scraping of NBA player stats
@st.cache_data
def load_data(year):
    url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_game.html'
    html = pd.read_html(url, header = 0)
    bb_df_reg = html[0]
    bb_df_playoffs = html[1]
    # Data preprocessing
    bb_df_reg = bb_df_reg.fillna(0)
    bb_df_playoffs = bb_df_playoffs.fillna(0)
    playerstats_reg = bb_df_reg.drop(['Rk', 'Awards'], axis=1)
    playerstats_reg = playerstats_reg.drop(playerstats_reg.index[-1])
    playerstats_playoffs = bb_df_playoffs.drop(['Rk', 'Awards'], axis=1)
    playerstats_playoffs = playerstats_playoffs.drop(playerstats_playoffs.index[-1])
    return playerstats_reg, playerstats_playoffs

playerstats_reg, playerstats_playoffs = load_data(selected_year)

# Sidebar - Type
type_options = {
    'Regular season': playerstats_reg,
    'Playoffs': playerstats_playoffs
}
selected_type = st.sidebar.selectbox('Type', list(type_options.keys()))

playerstats_df = type_options[selected_type]

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats_df.Team.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
st.sidebar.markdown('''
                    **2TM**: Player has played on two teams  
                    **3TM**: Player has played on three teams
                    '''
)

# Define team background colors for multiselect options
st.markdown('''
<style>
span[data-baseweb='tag'] {
  color: white;
  background-color: red;
}

span[data-baseweb='tag']:has(span[title='ATL']) {
  color: white;
  background-color: crimson;
}

span[data-baseweb='tag']:has(span[title='BOS']) {
  color: white;
  background-color: forestgreen;
}

span[data-baseweb='tag']:has(span[title='BRK']) {
  color: white;
  background-color: dimgray;
}

span[data-baseweb='tag']:has(span[title='CHI']) {
  color: white;
  background-color: red;
}

span[data-baseweb='tag']:has(span[title='CHO']) {
  color: black;
  background-color: mediumturquoise;
}

span[data-baseweb='tag']:has(span[title='CLE']) {
  color: white;
  background-color: maroon;
}

span[data-baseweb='tag']:has(span[title='DAL']) {
  color: white;
  background-color: mediumblue;
}

span[data-baseweb='tag']:has(span[title='DEN']) {
  color: white;
  background-color: navy;
}

span[data-baseweb='tag']:has(span[title='DET']) {
  color: white;
  background-color: orangered;
}

span[data-baseweb='tag']:has(span[title='GSW']) {
  color: black;
  background-color: gold;
}

span[data-baseweb='tag']:has(span[title='HOU']) {
  color: white;
  background-color: red;
}

span[data-baseweb='tag']:has(span[title='IND']) {
  color: black;
  background-color: gold;
}

span[data-baseweb='tag']:has(span[title='LAC']) {
  color: white;
  background-color: dodgerblue;
}

span[data-baseweb='tag']:has(span[title='LAL']) {
  color: white;
  background-color: darkorchid;
}

span[data-baseweb='tag']:has(span[title='MEM']) {
  color: white;
  background-color: navy;
}

span[data-baseweb='tag']:has(span[title='MIA']) {
  color: white;
  background-color: red;
}

span[data-baseweb='tag']:has(span[title='MIL']) {
  color: white;
  background-color: seagreen;
}

span[data-baseweb='tag']:has(span[title='MIN']) {
  color: white;
  background-color: midnightblue;
}

span[data-baseweb='tag']:has(span[title='NOP']) {
  color: white;
  background-color: mediumseagreen;
}

span[data-baseweb='tag']:has(span[title='NYK']) {
  color: white;
  background-color: royalblue;
}

span[data-baseweb='tag']:has(span[title='OKC']) {
  color: white;
  background-color: tomato;
}

span[data-baseweb='tag']:has(span[title='ORL']) {
  color: white;
  background-color: mediumblue;
}

span[data-baseweb='tag']:has(span[title='PHI']) {
  color: black;
  background-color: mintcream;
}

span[data-baseweb='tag']:has(span[title='PHO']) {
  color: white;
  background-color: orange;
}

span[data-baseweb='tag']:has(span[title='POR']) {
  color: white;
  background-color: red;
}

span[data-baseweb='tag']:has(span[title='SAC']) {
  color: white;
  background-color: mediumpurple;
}

span[data-baseweb='tag']:has(span[title='SAS']) {
  color: white;
  background-color: black;
}

span[data-baseweb='tag']:has(span[title='TOR']) {
  color: white;
  background-color: mediumorchid;
}

span[data-baseweb='tag']:has(span[title='UTA']) {
  color: white;
  background-color: rebeccapurple;
}

span[data-baseweb='tag']:has(span[title='WAS']) {
  color: white;
  background-color: slateblue;
}
</style>
''', unsafe_allow_html=True)

# Sidebar - Position selection
sorted_unique_pos = sorted(playerstats_df.Pos.unique())
selected_pos = st.sidebar.multiselect('Position', sorted_unique_pos, sorted_unique_pos)

# Filtering data
bb_selected_team = playerstats_df[(playerstats_df.Team.isin(selected_team)) & (playerstats_df.Pos.isin(selected_pos))]

# Add body description and DataFrame / table
st.header('Player stats of selected team(s)')
st.write('Data dimension: ' + str(bb_selected_team.shape[0]) + ' rows and ' + str(bb_selected_team.shape[1]) + ' columns.')
st.dataframe(bb_selected_team)

# Download NBA player stats data to csv file
csv = bb_selected_team.to_csv(index=False)
st.download_button(label='Download CSV file', data=csv, file_name='playerstats.csv', mime='text/csv')

# Heatmap
if st.button('Intercorrelation heatmap'):
    st.header('Intercorrelation matrix heatmap')
    corr = bb_selected_team.corr(numeric_only=True)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(fig)

# Shot distribution radius chart
st.markdown('''<hr style='border: 2px solid #333; margin-top: 25px; margin-bottom: 25px;' />''', unsafe_allow_html=True)
st.subheader('Player point type distribution')
st.markdown('''This is a table showing the distribution of points a player generates per game. 
The type of points consist of 2 point baskets, three-pointers, and free throws.
         \n\n Select the checkbox(es) in the <code>Select</code> column and press the <b>Shot radius chart</b>
         button to see a visual representation of the player point distribution. Multiple players can
         be selected.''', unsafe_allow_html=True)

point_dis_df = playerstats_df[['Player', 'Age', 'Team', 'Pos', '2P', '3P', 'FT', 'PTS']]
point_dis_selected_team = point_dis_df[(playerstats_df.Team.isin(selected_team)) & (point_dis_df.Pos.isin(selected_pos))]

if 'selected_players' not in st.session_state:
    st.session_state.selected_players = set()

search_query = st.text_input(':mag: Search for player name:', key='search1')
if search_query:
    filtered = point_dis_selected_team[point_dis_selected_team['Player'].str.contains(search_query, case=False, na=False)].copy()
else:
    filtered = point_dis_selected_team.copy()

filtered['Select'] = filtered['Player'].isin(st.session_state.selected_players)

point_dis_selected_team_edit = st.data_editor(filtered, use_container_width=True, num_rows='fixed', key='editor1')

for _, row in point_dis_selected_team_edit.iterrows():
    if row['Select']:
        st.session_state.selected_players.add(row['Player'])
    else:
        st.session_state.selected_players.discard(row['Player'])

selected_players = point_dis_selected_team[point_dis_selected_team['Player'].isin(st.session_state.selected_players)]
st.dataframe(selected_players)

if st.button('Shot radius chart', key='button_radius'):
    if selected_players.empty:
        st.error('Please select at least one player.')
    else:
        st.subheader('Point distribution radius chart')
        selected_players_long = selected_players.melt(id_vars='Player', value_vars=['2P', '3P', 'FT'], var_name='Point type', value_name='Attempts made')
        colors = ['#FF5733', '#1F77B4', '#a2ea00', '#618fe2', '#eb2f9b']
        point_fig = px.line_polar(selected_players_long, r='Attempts made', theta='Point type', color='Player', line_close=True, color_discrete_sequence=colors)
        point_fig.update_traces(fill='toself', mode='lines+markers')
        st.plotly_chart(point_fig, use_container_width=True)

# Pts, reb, ast bubble chart
st.markdown('''<hr style='border: 2px solid #333; margin-top: 25px; margin-bottom: 25px;' />''', unsafe_allow_html=True)
st.subheader('Player pts, reb, and assists contribution')
st.markdown('''This is a table showing a player's contributions of basic basketball statistics: points, rebounds, assists, steals, and blocks.
         \n\n Select the checkbox(es) in the <code>Select</code> column and press the <b>Statistics bubble chart</b>
         button to see a visual representation of the statistics overview of a player(s) distribution. Multiple players can
         be selected.''', unsafe_allow_html=True)

stats_dis_df = playerstats_df[['Player', 'Age', 'Team', 'Pos', 'PTS', 'TRB', 'AST', 'STL', 'BLK']]
stats_dis_selected_team = stats_dis_df[(playerstats_df.Team.isin(selected_team)) & (point_dis_df.Pos.isin(selected_pos))]

if 'stats_selected_players' not in st.session_state:
    st.session_state.stats_selected_players = set()

stats_search_query = st.text_input(':mag: Search for player name:', key='search2')
if stats_search_query:
    stats_filtered = stats_dis_selected_team[stats_dis_selected_team['Player'].str.contains(stats_search_query, case=False, na=False)].copy()
else:
    stats_filtered = stats_dis_selected_team.copy()

stats_filtered['Select'] = stats_filtered['Player'].isin(st.session_state.stats_selected_players)

stats_dis_selected_team_edit = st.data_editor(stats_filtered, use_container_width=True, num_rows='fixed', key='editor2')

for _, row in stats_dis_selected_team_edit.iterrows():
    if row['Select']:
        st.session_state.stats_selected_players.add(row['Player'])
    else:
        st.session_state.stats_selected_players.discard(row['Player'])

stats_selected_players = stats_dis_selected_team[stats_dis_selected_team['Player'].isin(st.session_state.stats_selected_players)]
st.dataframe(stats_selected_players)

if st.button('Statistics bubble chart', key='button_stats'):
    if stats_selected_players.empty:
        st.error('Please select at least one player.')
    else:
        st.subheader('Basic stats contribution bubble chart')
        colors = ['#FF5733', '#1F77B4', '#a2ea00', '#618fe2', '#eb2f9b']
        stats_dis_fig = px.scatter(stats_selected_players, x='AST', y='TRB', size='PTS', color='Player',
                                   hover_data=['Player', 'PTS', 'AST', 'TRB'], color_discrete_sequence=colors)
        stats_dis_fig.update_layout(xaxis_title='Assists', yaxis_title='Rebounds')
        stats_dis_fig.update_xaxes(title_font=dict(weight='bold', color='black'), showgrid=True)
        stats_dis_fig.update_yaxes(title_font=dict(weight='bold', color='black'), showgrid=True)
        st.plotly_chart(stats_dis_fig, use_container_width=True)

# Playing time comparison scatterplot #1
st.markdown('''<hr style='border: 2px solid #333; margin-top: 25px; margin-bottom: 25px;' />''', unsafe_allow_html=True)
st.subheader('Point production vs playing time comparison')
st.write('''This is a simplified table displaying the playing time and point production statistics of each player.
'There is data for each player if you hover over a point in the plot.''')
playing_time_df = playerstats_df[['Player', 'Age', 'Team', 'Pos', 'G', 'GS', 'MP', 'FG', 'FGA', 'PTS']]
playing_time_df['PTS/MP'] = (playing_time_df['PTS'] / playing_time_df['MP']).round(2)
playing_time_selected_team = playing_time_df[(playerstats_df.Team.isin(selected_team)) & (playing_time_df.Pos.isin(selected_pos))]
st.dataframe(playing_time_selected_team, use_container_width=True)

range_slider1 = st.slider('Minutes played range', min_value=0, max_value=40, value=(0, 40), key=1)
playing_time_filtered1 = playing_time_selected_team[(playing_time_selected_team['MP'] >= range_slider1[0]) & 
                                                   (playing_time_selected_team['MP'] <= range_slider1[1])]
playing_time_fig = px.scatter(playing_time_filtered1, x='PTS', y='MP', color='PTS', color_continuous_scale='Reds',
                              size='PTS', size_max=15, hover_data=['Player', 'PTS', 'MP', 'PTS/MP'])
playing_time_fig.update_layout(title='Points vs minutes played', title_x=0.5, title_xanchor='center', 
                               title_font=dict(size=20), xaxis_title='Points per game', yaxis_title='Minutes played')
playing_time_fig.update_xaxes(title_font=dict(weight='bold', color='black'), showgrid=True)
playing_time_fig.update_yaxes(title_font=dict(weight='bold', color='black'), showgrid=True)
st.plotly_chart(playing_time_fig, use_container_width=True)

# Playing time comparison scatterplot #2
st.markdown('''<hr style='border: 1px solid #333; margin-top: 25px; margin-bottom: 25px;' />''', unsafe_allow_html=True)
range_slider2 = st.slider('Minutes played range', min_value=0, max_value=40, value=(0, 40), key=2)
playing_time_filtered2 = playing_time_selected_team[(playing_time_selected_team['MP'] >= range_slider2[0]) & 
                                                   (playing_time_selected_team['MP'] <= range_slider2[1])]
games_started_fig = px.scatter(playing_time_filtered2, x='GS', y='MP', color='PTS', color_continuous_scale='Blues',
                              size='PTS', size_max=15, hover_data=['Player', 'PTS', 'MP', 'GS', 'G'])
games_started_fig.update_layout(title='Starter vs bench scoring comparison', title_x=0.5, title_xanchor='center', 
                               title_font=dict(size=20), xaxis_title='Games started', yaxis_title='Minutes played')
games_started_fig.update_xaxes(title_font=dict(weight='bold', color='black'), showgrid=True)
games_started_fig.update_yaxes(title_font=dict(weight='bold', color='black'), showgrid=True)
st.plotly_chart(games_started_fig, use_container_width=True)



