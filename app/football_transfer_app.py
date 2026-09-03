import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Page configuration
st.set_page_config(page_title="Football Transfer Analytics", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data with error handling
@st.cache_data
def load_data():
    # Transfer datasets (required)
    try:
        goalkeepers_transfers = pd.read_csv('Goalkeepers_Transfers.csv')
        finalized_transfers = pd.read_csv('Finalized_Transfers_Dataset.csv')
        attackers_transfers = pd.read_csv('Attackers_Transfers.csv')
        midfielders_transfers = pd.read_csv('Midfielders_Transfers.csv')
        defenders_transfers = pd.read_csv('Defenders_Transfers.csv')
        goalkeepers_seasons = pd.read_csv('goalkeepers_all_seasons_raw.csv')
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}")
        st.stop()
    
    # Bootstrap and model datasets (optional)
    try:
        gk_bootstrap = pd.read_csv('GK_Performance_Bootstrap_Weights.csv')
    except:
        gk_bootstrap = pd.DataFrame()
    
    try:
        defenders_bootstrap = pd.read_csv('Defenders_Bootstrap_Weights1.csv')
    except:
        defenders_bootstrap = pd.DataFrame()
    
    try:
        midfielders_bootstrap = pd.read_csv('Midfielders_Bootstrap_Weights.csv')
    except:
        midfielders_bootstrap = pd.DataFrame()
    
    try:
        attackers_bootstrap = pd.read_csv('Attackers_Global_Model_Bootstrap_Results.csv')
    except:
        attackers_bootstrap = pd.DataFrame()
    
    try:
        model_comparison = pd.read_csv('Model_Comparison_Results.csv')
    except:
        model_comparison = pd.DataFrame()
    
    try:
        feature_formula = pd.read_csv('Feature-Formula-RawComponents-Type.csv')
    except:
        feature_formula = pd.DataFrame()
    
    return (goalkeepers_transfers, finalized_transfers, attackers_transfers, 
            midfielders_transfers, defenders_transfers, goalkeepers_seasons,
            gk_bootstrap, defenders_bootstrap, midfielders_bootstrap, 
            attackers_bootstrap, model_comparison, feature_formula)

(gk_transfers, finalized, attackers, midfielders, defenders, gk_seasons,
 gk_bootstrap, def_bootstrap, mid_bootstrap, att_bootstrap, model_comp, feature_formula) = load_data()

# Title and description
st.title("Football Transfer Market Analytics Dashboard")
st.markdown("### Comprehensive analysis of player transfers and performance across positions")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a view:", 
                        ["Overview", "Transfer Market Analysis", "Position Comparison", 
                         "Goalkeeper Analysis", "Defender Analysis", 
                         "Midfielder Analysis", "Attacker Analysis"])

# ============== OVERVIEW PAGE ==============
if page == "Overview":
    st.header("Transfer Market Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Transfers", f"{len(finalized):,}")
    with col2:
        avg_fee = finalized[finalized['Fee'] > 0]['Fee'].mean()
        st.metric("Avg Transfer Fee", f"€{avg_fee/1e6:.1f}M")
    with col3:
        success_rate = (finalized['Transfer_Success'].sum() / len(finalized)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        st.metric("Goalkeepers", len(gk_transfers))
        st.metric("Defenders", len(defenders))
    with col5:
        st.metric("Midfielders", len(midfielders))
        st.metric("Attackers", len(attackers))
    
    st.markdown("---")
    
    # Position distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transfer Distribution by Position")
        position_counts = finalized['Position'].value_counts()
        fig = px.pie(values=position_counts.values, names=position_counts.index,
                     title="Transfer Share by Position",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Success Rate by Position")
        success_by_pos = finalized.groupby('Position').agg({
            'Transfer_Success': 'mean',
            'PlayerName': 'count'
        }).reset_index()
        success_by_pos.columns = ['Position', 'Success_Rate', 'Count']
        success_by_pos['Success_Rate'] = success_by_pos['Success_Rate'] * 100
        
        fig = px.bar(success_by_pos.nlargest(8, 'Count'), 
                     x='Position', y='Success_Rate',
                     title="Transfer Success Rate (%)",
                     color='Success_Rate',
                     color_continuous_scale='RdYlGn')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Transfer fees over seasons
    st.subheader("Transfer Market Trends Over Seasons")
    season_analysis = finalized[finalized['Fee'] > 0].groupby('Season').agg({
        'Fee': ['sum', 'mean', 'count'],
        'Transfer_Success': 'mean'
    }).reset_index()
    season_analysis.columns = ['Season', 'Total_Fees', 'Avg_Fee', 'Num_Transfers', 'Success_Rate']
    season_analysis['Success_Rate'] = season_analysis['Success_Rate'] * 100
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Total Transfer Spending", "Average Transfer Fee"))
    
    fig.add_trace(go.Bar(x=season_analysis['Season'], y=season_analysis['Total_Fees']/1e6,
                         name='Total Fees (M EUR)', marker_color='indianred'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=season_analysis['Season'], y=season_analysis['Avg_Fee']/1e6,
                            name='Avg Fee (M EUR)', mode='lines+markers', marker_color='lightseagreen'),
                  row=1, col=2)
    
    fig.update_xaxes(title_text="Season", row=1, col=1)
    fig.update_xaxes(title_text="Season", row=1, col=2)
    fig.update_yaxes(title_text="Million EUR", row=1, col=1)
    fig.update_yaxes(title_text="Million EUR", row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Definitions Section (only if data available)
    if not feature_formula.empty:
        st.markdown("---")
        st.subheader("Performance Metrics Definitions")
        st.markdown("Understanding how player performance is calculated across different metrics:")
        
        # Display feature formula table
        feature_display = feature_formula.copy()
        st.dataframe(feature_display, use_container_width=True, hide_index=True)
        
        # Visual breakdown by type
        col1, col2 = st.columns(2)
        
        with col1:
            type_counts = feature_formula['Type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title="Metrics by Category",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(feature_formula, x='Feature', y='Raw Components',
                        title="Components per Metric",
                        color='Type',
                        labels={'Raw Components': 'Number of Components'},
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

# ============== TRANSFER MARKET ANALYSIS PAGE ==============
elif page == "Transfer Market Analysis":
    st.header("Transfer Market Analysis")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_season = st.selectbox("Select Season", 
                                      ['All'] + sorted(finalized['Season'].unique().tolist()))
    with col2:
        min_fee = st.slider("Minimum Transfer Fee (M EUR)", 0, 100, 0)
    with col3:
        positions = st.multiselect("Filter Positions", 
                                   finalized['Position'].unique().tolist(),
                                   default=finalized['Position'].unique().tolist())
    
    # Filter data
    filtered_data = finalized.copy()
    if selected_season != 'All':
        filtered_data = filtered_data[filtered_data['Season'] == selected_season]
    filtered_data = filtered_data[filtered_data['Fee'] >= min_fee * 1e6]
    filtered_data = filtered_data[filtered_data['Position'].isin(positions)]
    
    st.markdown(f"**Showing {len(filtered_data)} transfers**")
    
    # Transfer fee vs performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transfer Fee vs Performance")
        fee_perf_data = filtered_data[filtered_data['Fee'] > 0].copy()
        fee_perf_data['Fee_M'] = fee_perf_data['Fee'] / 1e6
        
        fig = px.scatter(fee_perf_data, x='Fee_M', y='Performance_Index',
                        color='Transfer_Success', size='Age',
                        hover_data=['PlayerName', 'Position', 'LeftClub', 'JoinedClub'],
                        title="Does Higher Fee Mean Better Performance?",
                        labels={'Fee_M': 'Transfer Fee (M EUR)', 'Transfer_Success': 'Success'},
                        color_continuous_scale='RdYlGn')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution of Transfers")
        fig = px.histogram(filtered_data, x='Age', color='Transfer_Success',
                          title="Transfer Success by Player Age",
                          labels={'Transfer_Success': 'Success'},
                          nbins=20, barmode='group',
                          color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # Club rank changes
    st.subheader("Club Ranking Impact on Transfer Success")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(filtered_data, x='Transfer_Success', y='Delta_Club_Rank',
                    title="Club Rank Change Distribution",
                    labels={'Delta_Club_Rank': 'Change in Club Ranking', 
                           'Transfer_Success': 'Transfer Success'},
                    color='Transfer_Success',
                    color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Success rate by club movement
        filtered_data['Movement'] = filtered_data['Delta_Club_Rank'].apply(
            lambda x: 'Moving Up' if x < 0 else ('Moving Down' if x > 0 else 'Same Level'))
        movement_success = filtered_data.groupby('Movement')['Transfer_Success'].agg(['mean', 'count']).reset_index()
        movement_success.columns = ['Movement', 'Success_Rate', 'Count']
        movement_success['Success_Rate'] = movement_success['Success_Rate'] * 100
        
        fig = px.bar(movement_success, x='Movement', y='Success_Rate',
                    title="Success Rate by Club Movement",
                    text='Count', color='Success_Rate',
                    color_continuous_scale='RdYlGn')
        fig.update_traces(texttemplate='%{text} transfers', textposition='outside')
        fig.update_layout(yaxis_title="Success Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top transfers table
    st.subheader("Most Expensive Transfers")
    top_transfers = filtered_data.nlargest(10, 'Fee')[
        ['PlayerName', 'Position', 'Age', 'Season', 'LeftClub', 'JoinedClub', 
         'Fee', 'Performance_Index', 'Transfer_Success']
    ].copy()
    top_transfers['Fee'] = top_transfers['Fee'].apply(lambda x: f"€{x/1e6:.1f}M")
    top_transfers['Transfer_Success'] = top_transfers['Transfer_Success'].map({0: 'Failed', 1: 'Success'})
    st.dataframe(top_transfers, use_container_width=True, hide_index=True)

# ============== POSITION COMPARISON PAGE ==============
elif page == "Position Comparison":
    st.header("Position Comparison")
    
    # Prepare data by position
    gk_transfers['Position_Group'] = 'Goalkeeper'
    attackers['Position_Group'] = 'Attacker'
    midfielders['Position_Group'] = 'Midfielder'
    defenders['Position_Group'] = 'Defender'
    
    # Key metrics comparison
    st.subheader("Key Metrics by Position")
    
    metrics_data = []
    for df, pos in [(gk_transfers, 'Goalkeeper'), (defenders, 'Defender'), 
                    (midfielders, 'Midfielder'), (attackers, 'Attacker')]:
        metrics_data.append({
            'Position': pos,
            'Avg Age': df['Age'].mean(),
            'Avg Fee (M EUR)': df[df['Fee'] > 0]['Fee'].mean() / 1e6,
            'Success Rate (%)': df['Transfer_Success'].mean() * 100,
            'Avg Performance': df['Performance_Index'].mean(),
            'Transfer Count': len(df)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        for metric in ['Avg Age', 'Success Rate (%)']:
            fig.add_trace(go.Bar(name=metric, x=metrics_df['Position'], 
                                y=metrics_df[metric]))
        fig.update_layout(title="Age and Success Rate Comparison", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        for metric in ['Avg Fee (M EUR)', 'Avg Performance']:
            fig.add_trace(go.Bar(name=metric, x=metrics_df['Position'], 
                                y=metrics_df[metric]))
        fig.update_layout(title="Fee and Performance Comparison", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance Index Distribution
    st.subheader("Performance Distribution Across Positions")
    
    combined_perf = pd.concat([
        gk_transfers[['Performance_Index', 'Position_Group', 'Transfer_Success']],
        defenders[['Performance_Index', 'Position_Group', 'Transfer_Success']],
        midfielders[['Performance_Index', 'Position_Group', 'Transfer_Success']],
        attackers[['Performance_Index', 'Position_Group', 'Transfer_Success']]
    ])
    
    fig = px.violin(combined_perf, x='Position_Group', y='Performance_Index',
                   color='Transfer_Success', box=True,
                   title="Performance Index Distribution by Position and Success",
                   labels={'Position_Group': 'Position', 'Transfer_Success': 'Success'},
                   color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Injury and Loan impact
    st.subheader("Risk Factors: Injuries and Loans")
    col1, col2 = st.columns(2)
    
    with col1:
        combined_injury = pd.concat([
            gk_transfers[['Injury_Index', 'Transfer_Success', 'Position_Group']],
            defenders[['Injury_Index', 'Transfer_Success', 'Position_Group']],
            midfielders[['Injury_Index', 'Transfer_Success', 'Position_Group']],
            attackers[['Injury_Index', 'Transfer_Success', 'Position_Group']]
        ])
        
        fig = px.box(combined_injury, x='Position_Group', y='Injury_Index',
                    color='Transfer_Success',
                    title="Injury Index by Position",
                    labels={'Position_Group': 'Position', 'Transfer_Success': 'Success'},
                    color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        combined_loan = pd.concat([
            gk_transfers[['Loan_Index', 'Transfer_Success', 'Position_Group']],
            defenders[['Loan_Index', 'Transfer_Success', 'Position_Group']],
            midfielders[['Loan_Index', 'Transfer_Success', 'Position_Group']],
            attackers[['Loan_Index', 'Transfer_Success', 'Position_Group']]
        ])
        
        fig = px.box(combined_loan, x='Position_Group', y='Loan_Index',
                    color='Transfer_Success',
                    title="Loan Index by Position",
                    labels={'Position_Group': 'Position', 'Transfer_Success': 'Success'},
                    color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)

# ============== GOALKEEPER ANALYSIS ==============
elif page == "Goalkeeper Analysis":
    st.header("Goalkeeper Analysis")
    
    # Goalkeeper Transfer Overview
    st.subheader("Goalkeeper Transfer Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Goalkeepers", len(gk_transfers))
    with col2:
        avg_fee_gk = gk_transfers[gk_transfers['Fee'] > 0]['Fee'].mean()
        st.metric("Avg Transfer Fee", f"€{avg_fee_gk/1e6:.1f}M")
    with col3:
        success_rate_gk = (gk_transfers['Transfer_Success'].sum() / len(gk_transfers)) * 100
        st.metric("Success Rate", f"{success_rate_gk:.1f}%")
    with col4:
        avg_age_gk = gk_transfers['Age'].mean()
        st.metric("Avg Age", f"{avg_age_gk:.1f}")
    
    st.markdown("---")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Save percentage - Filter out NaN values
        gk_clean = gk_transfers[gk_transfers['Saves_Pct'].notna() & gk_transfers['Performance_Index'].notna()].copy()
        fig = px.scatter(gk_clean, x='Saves', y='Saves_Pct',
                       color='Transfer_Success', size='Performance_Index',
                       hover_data=['PlayerName', 'LeftClub', 'JoinedClub'],
                       title="Saves vs Save Percentage",
                       labels={'Transfer_Success': 'Success'},
                       color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Clean sheets
        fig = px.scatter(gk_clean[gk_clean['Clean_Sheets'].notna()], 
                       x='Clean_Sheets', y='Clean_Sheets_Pct',
                       color='Transfer_Success',
                       hover_data=['PlayerName', 'LeftClub', 'JoinedClub'],
                       title="Clean Sheets vs Clean Sheet Percentage",
                       labels={'Transfer_Success': 'Success'},
                       color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Season performance trends
    st.subheader("Goalkeeper Performance Trends Across Seasons")
    
    gk_seasons_clean = gk_seasons.copy()
    gk_seasons_clean['Performance Save%'] = pd.to_numeric(gk_seasons_clean['Performance Save%'], errors='coerce')
    
    season_avg = gk_seasons_clean.groupby('Season').agg({
        'Performance Save%': 'mean',
        'Performance CS%': 'mean',
        'Playing Time MP': 'sum',
        'Player': 'count'
    }).reset_index()
    season_avg.columns = ['Season', 'Avg_Save_Pct', 'Avg_CS_Pct', 'Total_Matches', 'Num_Goalkeepers']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=season_avg['Season'], y=season_avg['Avg_Save_Pct'],
                                mode='lines+markers', name='Avg Save %',
                                line=dict(color='royalblue', width=3)))
        fig.update_layout(title="Average Save Percentage Over Seasons",
                         xaxis_title="Season", yaxis_title="Save %", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=season_avg['Season'], y=season_avg['Avg_CS_Pct'],
                                mode='lines+markers', name='Avg Clean Sheet %',
                                line=dict(color='seagreen', width=3)))
        fig.update_layout(title="Average Clean Sheet Percentage Over Seasons",
                         xaxis_title="Season", yaxis_title="Clean Sheet %", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bootstrap Weights (if available) - FIXED COLUMN NAMES
    if not gk_bootstrap.empty:
        st.subheader("Performance Feature Importance")
        fig = px.bar(gk_bootstrap, x='Mean_Weight', y='Feature',
                    title="Goalkeeper Performance Weights",
                    orientation='h',
                    error_x='Std_Weight',
                    color='Mean_Weight',
                    color_continuous_scale='Blues')
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performers
    st.subheader("Top Goalkeeper Performers")
    top_gks = gk_transfers.nlargest(10, 'Performance_Index')[
        ['PlayerName', 'Age', 'LeftClub', 'JoinedClub', 'Saves', 'Saves_Pct',
         'Clean_Sheets', 'Clean_Sheets_Pct', 'Performance_Index', 'Transfer_Success']
    ].copy()
    top_gks['Transfer_Success'] = top_gks['Transfer_Success'].map({0: 'Failed', 1: 'Success'})
    st.dataframe(top_gks, use_container_width=True, hide_index=True)

# ============== DEFENDER ANALYSIS ==============
elif page == "Defender Analysis":
    st.header("Defender Analysis")
    
    st.subheader("Defender Transfer Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Defenders", len(defenders))
    with col2:
        avg_fee_def = defenders[defenders['Fee'] > 0]['Fee'].mean()
        st.metric("Avg Transfer Fee", f"€{avg_fee_def/1e6:.1f}M")
    with col3:
        success_rate_def = (defenders['Transfer_Success'].sum() / len(defenders)) * 100
        st.metric("Success Rate", f"{success_rate_def:.1f}%")
    with col4:
        avg_age_def = defenders['Age'].mean()
        st.metric("Avg Age", f"{avg_age_def:.1f}")
    
    st.markdown("---")
    
    # Position breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Defender Position Distribution")
        position_counts = defenders['Position'].value_counts()
        fig = px.pie(values=position_counts.values, names=position_counts.index,
                    title="Defender Types",
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Success Rate by Defender Type")
        success_by_def_pos = defenders.groupby('Position').agg({
            'Transfer_Success': 'mean',
            'PlayerName': 'count'
        }).reset_index()
        success_by_def_pos.columns = ['Position', 'Success_Rate', 'Count']
        success_by_def_pos['Success_Rate'] = success_by_def_pos['Success_Rate'] * 100
        
        fig = px.bar(success_by_def_pos, 
                     x='Position', y='Success_Rate',
                     title="Success Rate by Position",
                     color='Success_Rate',
                     text='Count',
                     color_continuous_scale='RdYlGn')
        fig.update_traces(texttemplate='%{text} transfers', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("Defender Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Defensive Actions Distribution
        fig = px.histogram(defenders, x='Defensive_Actions', color='Transfer_Success',
                          title="Defensive Actions Distribution",
                          labels={'Transfer_Success': 'Success'},
                          nbins=20, barmode='overlay',
                          color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mistake Rate Analysis
        fig = px.box(defenders, x='Transfer_Success', y='Mistake_Rate',
                    title="Mistake Rate by Transfer Success",
                    labels={'Mistake_Rate': 'Mistake Rate', 
                           'Transfer_Success': 'Transfer Success'},
                    color='Transfer_Success',
                    color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Performance Indicators
    st.subheader("Key Performance Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(defenders, x='Total_Carry_Distance', y='Performance_Index',
                        color='Transfer_Success',
                        hover_data=['PlayerName', 'Position'],
                        title="Carry Distance vs Performance",
                        labels={'Transfer_Success': 'Success'},
                        color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # FIX: Filter out NaN values from Performance_Index before using as size
        def_clean = defenders[defenders['Performance_Index'].notna()].copy()
        fig = px.scatter(def_clean, x='Attacking_Contributions', y='Defensive_Actions',
                        color='Transfer_Success', size='Performance_Index',
                        hover_data=['PlayerName', 'Position'],
                        title="Attacking vs Defensive Contributions",
                        labels={'Transfer_Success': 'Success'},
                        color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Bootstrap Weights (if available) - Uses 'Weight' column
    if not def_bootstrap.empty:
        st.subheader("Performance Feature Importance by Role")
        # Group by role and show top features
        for role in def_bootstrap['Role_Cohort'].unique():
            role_data = def_bootstrap[def_bootstrap['Role_Cohort'] == role].nlargest(10, 'Weight')
            fig = px.bar(role_data, x='Weight', y='Feature',
                        title=f"{role} - Feature Weights",
                        orientation='h',
                        color='Weight',
                        color_continuous_scale='Reds')
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Top performers
    st.subheader("Top Defender Performers")
    top_defenders_all = defenders.nlargest(15, 'Performance_Index')[
        ['PlayerName', 'Age', 'Position', 'LeftClub', 'JoinedClub', 
         'Performance_Index', 'Defensive_Actions', 'Duel_Efficiency', 
         'Mistake_Rate', 'Transfer_Success']
    ].copy()
    top_defenders_all['Transfer_Success'] = top_defenders_all['Transfer_Success'].map({0: 'Failed', 1: 'Success'})
    top_defenders_all['Mistake_Rate'] = top_defenders_all['Mistake_Rate'].round(3)
    st.dataframe(top_defenders_all, use_container_width=True, hide_index=True)

# ============== MIDFIELDER ANALYSIS ==============
elif page == "Midfielder Analysis":
    st.header("Midfielder Analysis")
    
    st.subheader("Midfielder Transfer Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Midfielders", len(midfielders))
    with col2:
        avg_fee_mid = midfielders[midfielders['Fee'] > 0]['Fee'].mean()
        st.metric("Avg Transfer Fee", f"€{avg_fee_mid/1e6:.1f}M")
    with col3:
        success_rate_mid = (midfielders['Transfer_Success'].sum() / len(midfielders)) * 100
        st.metric("Success Rate", f"{success_rate_mid:.1f}%")
    with col4:
        avg_age_mid = midfielders['Age'].mean()
        st.metric("Avg Age", f"{avg_age_mid:.1f}")
    
    st.markdown("---")
    
    # Performance metrics
    st.subheader("Midfielder Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FIX: Filter out NaN values from Performance_Index before using as size
        mid_clean = midfielders[midfielders['Performance_Index'].notna()].copy()
        fig = px.scatter(mid_clean, x='Defensive_Actions', y='Attacking_Contributions',
                       color='Transfer_Success', size='Performance_Index',
                       hover_data=['PlayerName', 'Position'],
                       title="Defensive vs Attacking Contributions",
                       labels={'Transfer_Success': 'Success'},
                       color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(mid_clean, x='Ball_Progression', y='Performance_Index',
                       color='Transfer_Success',
                       hover_data=['PlayerName', 'Position'],
                       title="Ball Progression vs Performance",
                       labels={'Transfer_Success': 'Success'},
                       color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Role analysis
    st.subheader("Midfielder Role Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        role_counts = midfielders['Role_Cohort'].value_counts()
        fig = px.pie(values=role_counts.values, names=role_counts.index,
                    title="Midfielder Roles",
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Success by role
        success_by_role = midfielders.groupby('Role_Cohort').agg({
            'Transfer_Success': 'mean',
            'PlayerName': 'count'
        }).reset_index()
        success_by_role.columns = ['Role', 'Success_Rate', 'Count']
        success_by_role['Success_Rate'] = success_by_role['Success_Rate'] * 100
        
        fig = px.bar(success_by_role, 
                     x='Role', y='Success_Rate',
                     title="Success Rate by Role",
                     color='Success_Rate',
                     text='Count',
                     color_continuous_scale='RdYlGn')
        fig.update_traces(texttemplate='%{text} transfers', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Bootstrap Weights (if available) - Uses 'Weight' column
    if not mid_bootstrap.empty:
        st.subheader("Performance Feature Importance by Role")
        # Group by role and show top features
        for role in mid_bootstrap['Role_Cohort'].unique():
            role_data = mid_bootstrap[mid_bootstrap['Role_Cohort'] == role].nlargest(10, 'Weight')
            fig = px.bar(role_data, x='Weight', y='Feature',
                        title=f"{role} - Feature Weights",
                        orientation='h',
                        color='Weight',
                        color_continuous_scale='Greens')
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Top performers
    st.subheader("Top Midfielder Performers")
    top_mids = midfielders.nlargest(15, 'Performance_Index')[
        ['PlayerName', 'Age', 'Position', 'LeftClub', 'JoinedClub',
         'Performance_Index', 'Attacking_Contributions', 'Defensive_Actions', 
         'Ball_Progression', 'Transfer_Success']
    ].copy()
    top_mids['Transfer_Success'] = top_mids['Transfer_Success'].map({0: 'Failed', 1: 'Success'})
    st.dataframe(top_mids, use_container_width=True, hide_index=True)

# ============== ATTACKER ANALYSIS ==============
elif page == "Attacker Analysis":
    st.header("Attacker Analysis")
    
    st.subheader("Attacker Transfer Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Attackers", len(attackers))
    with col2:
        avg_fee_att = attackers[attackers['Fee'] > 0]['Fee'].mean()
        st.metric("Avg Transfer Fee", f"€{avg_fee_att/1e6:.1f}M")
    with col3:
        success_rate_att = (attackers['Transfer_Success'].sum() / len(attackers)) * 100
        st.metric("Success Rate", f"{success_rate_att:.1f}%")
    with col4:
        avg_age_att = attackers['Age'].mean()
        st.metric("Avg Age", f"{avg_age_att:.1f}")
    
    st.markdown("---")
    
    # Performance metrics
    st.subheader("Attacker Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FIX: Filter out NaN values from Performance_Index before using as size
        att_clean = attackers[attackers['Goals'].notna() & attackers['Performance_Index'].notna()].copy()
        fig = px.scatter(att_clean, x='Goals', y='Assists',
                       color='Transfer_Success', size='Performance_Index',
                       hover_data=['PlayerName', 'LeftClub', 'JoinedClub'],
                       title="Goals vs Assists",
                       labels={'Transfer_Success': 'Success'},
                       color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(att_clean[att_clean['Key_Passes'].notna()], 
                       x='Key_Passes', y='Performance_Index',
                       color='Transfer_Success',
                       hover_data=['PlayerName', 'Goals', 'Assists'],
                       title="Key Passes vs Performance",
                       labels={'Transfer_Success': 'Success'},
                       color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    st.subheader("Advanced Attacker Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Progressive passes
        fig = px.histogram(attackers, x='Progressive_Passes', color='Transfer_Success',
                          title="Progressive Passes Distribution",
                          labels={'Transfer_Success': 'Success'},
                          nbins=20, barmode='overlay',
                          color_discrete_map={0: '#ff6b6b', 1: '#51cf66'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Nationality distribution
        nat_counts = attackers['Nationality'].value_counts().head(15)
        fig = px.bar(x=nat_counts.index, y=nat_counts.values,
                    title="Top 15 Nationalities",
                    labels={'x': 'Nationality', 'y': 'Number of Transfers'},
                    color=nat_counts.values, color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)
    
    # Bootstrap Results (if available) - FIXED: Shows model comparison
    if not att_bootstrap.empty:
        st.subheader("Model Performance Comparison")
        st.markdown("Comparison of different machine learning models for predicting attacker transfer success:")
        
        # Display model comparison table
        st.dataframe(att_bootstrap, use_container_width=True, hide_index=True)
        
        # Visualize model accuracy
        fig = px.bar(att_bootstrap, x='Model', y='Accuracy_mean',
                    title="Model Accuracy Comparison",
                    error_y='Accuracy_std',
                    color='Accuracy_mean',
                    color_continuous_scale='Oranges',
                    labels={'Accuracy_mean': 'Mean Accuracy', 'Accuracy_std': 'Std Deviation'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performers
    st.subheader("Top Attacker Performers")
    top_attackers = attackers.nlargest(15, 'Performance_Index')[
        ['PlayerName', 'Age', 'LeftClub', 'JoinedClub', 'Goals', 'Assists', 
         'Key_Passes', 'Performance_Index', 'Transfer_Success']
    ].copy()
    top_attackers['Transfer_Success'] = top_attackers['Transfer_Success'].map({0: 'Failed', 1: 'Success'})
    st.dataframe(top_attackers, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("**Data Source:** Football Transfer Market Analysis | **Dashboard Created with:** Streamlit & Plotly")