import streamlit as st
import requests
import json
import time
import ast
import datetime
from datetime import datetime
import pandas as pd
import altair as alt
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gspread_dataframe as gd
import logging
from datetime import datetime
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from chatbot import chat_request
import streamlit_option_menu
from streamlit_option_menu import option_menu
from pandasql import sqldf


# Set up logging to console and logs.log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('logs.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

global scope
scope = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file'
    ]
def get_data():

    creds_dict = {"your cred"}

    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict,scope)
        client = gspread.authorize(creds)
        #Fetch the sheet
        logger.info("Opening the sheet")
        sh = client.open('Eblock Transportjobs')
        logger.info("Fetching the sheet")
        sheet_deliveries=sh.worksheet("Deliveries")
        logger.info("Fetching the data")
        db_deliveries=sheet_deliveries.get_all_values()
        deliveries_db=pd.DataFrame(db_deliveries[1:],columns=db_deliveries[0])
        logger.info("Data Retieved")
        return deliveries_db
    except Exception as e:
        logger.error("Error occured while fetching data")
        logger.error(e)
        return None
       

# Getting Recommendations
def convert_date_format(input_date):
    # Convert the input date string to a datetime object
    date_object = datetime.strptime(input_date, "%B %d, %Y")

    # Format the datetime object as a string in "yyyy-mm-dd" format
    formatted_date = date_object.strftime("%Y-%m-%d")

    return formatted_date

def to_date(df):
    date_columns = ['Estimated Pickup Date', 'Adjusted Pickup', 'Estimated Delivery', 'Adjusted Delivery', 'Final Delivery']
    for c in date_columns:
        df[c] =df[c].apply(lambda x: pd.to_datetime(x.split(" - ")[0], errors='coerce'))
    df['Estimated Pickup/Delivery']=df['Estimated Delivery']-df['Estimated Pickup Date']
    df['Final Pickup/Delivery']=df['Final Delivery']-df['Estimated Pickup Date']
    df['Time Difference']=df['Final Pickup/Delivery']-df['Estimated Pickup/Delivery']
    df['Time Difference']=df['Time Difference'].apply(lambda x: x.days)
    return df

def carrier_routes(df):
    # Step 1: Remove non-numeric characters from price values
    df['Final Amount'] = df['Final Amount'].str.replace('[^\d.]', '', regex=True)

    # Step 2: Convert price values to numeric format
    df['Final Amount'] = pd.to_numeric(df['Final Amount'])

    # Step 3: Group by carriers and calculate average price
    carrier_prices = df
    return carrier_prices

def custom_std(series):
    # Return 0 if there's only one data point, else calculate std
    return series.std() if len(series) > 1 else 0

def count_trips(df, carrier, pickup_loc, delivery_loc):
    return len(df[(df["Carrier"] == carrier) & (df["Pickup Location(s)"] == pickup_loc) & (df["Delivery Location(s)"] == delivery_loc)])


def count_unique_routes(df, carrier):
    
    t = df.copy()
    t["route"] = t["Pickup Location(s)"] + " " + t["Delivery Location(s)"]
    return len(np.unique(t[t["Carrier"] == carrier]["route"].values))

# Assuming df is your DataFrame with the provided columns

# Route and Buyer Analysis
def route_and_buyer_analysis(dataframe):
    # Extracting relevant columns
    buyers_route = dataframe[['Pickup Location(s)', 'Delivery Location(s)', 'Buyer']]  
    # Most frequently used routes
    frequent_routes = dataframe.groupby(['Pickup Location(s)', 'Delivery Location(s)']).size().reset_index(name='Route Frequency')
    # Buyers frequently using routes
    frequent_buyers = buyers_route.groupby(['Buyer', 'Pickup Location(s)', 'Delivery Location(s)']).size().reset_index(name='Buyers Frequency')
    
    return frequent_routes, frequent_buyers

# Carrier Performance Assessment
def carrier_performance_assessment(dataframe):
    # Extracting relevant columns
    performance_data = dataframe[['Carrier', 'Estimated Pickup Date', 'Estimated Delivery', 'Final Delivery']]

    #carrier popularity
    # Calculate Total Trips per Carrier
    carrier_popularity = dataframe.groupby('Carrier').size().reset_index(name='Total Trips')
    total_unique_carriers = dataframe['Carrier'].nunique()
    # Calculate Carrier Popularity
    carrier_popularity['Carrier Popularity Index'] = carrier_popularity['Total Trips'] / total_unique_carriers
    # Specify datetime format for parsing
    datetime_format = '%Y-%m-%d'  # Adjust the format based on your data
    
    # Convert datetime columns to datetime format
    performance_data['Pickup Time'] = pd.to_datetime(performance_data['Estimated Pickup Date'], format=datetime_format) - pd.to_datetime(dataframe['Adjusted Pickup'], errors='coerce')
    performance_data['Delivery Time'] = pd.to_datetime(performance_data['Final Delivery'], format=datetime_format) - pd.to_datetime(performance_data['Estimated Delivery'], errors='coerce')
    
    # Evaluating carriers based on pickup and delivery times
    carrier_performance = performance_data.groupby('Carrier').agg({'Pickup Time': 'mean', 'Delivery Time': 'mean'})
    
    return carrier_performance,carrier_popularity


# Pricing and Quote Analysis
def pricing_and_quote_analysis(df):
    # Extracting relevant columns
    quotes_data = df[['Carrier', 'Quotes', 'Final Amount']]
    # Analyzing quote variability for different routes
    quote_variability = quotes_data.groupby('Carrier')['Quotes'].nunique().reset_index(name='Quote Variability')
    
    # Identifying carriers who typically charge lower prices
    lower_price_carriers=quotes_data.groupby('Carrier')['Final Amount'].mean().nsmallest(3).reset_index(name='Minimum Mean Final Amount')
    # Calculating cost per vehicle for multi-vehicle jobs
    df['Cost per Vehicle'] =df['Final Amount'] / df['Vehicle Count']
    
    return quote_variability, lower_price_carriers, df[['Carrier', 'Cost per Vehicle']]

# Carrier Comparison and Niche Opportunities
def carrier_comparison_and_niche_opportunities(dataframe):
    # Comparing carriers based on efficiency, reliability, and pricing
    comparison_data = dataframe[['Carrier', 'Estimated Pickup Date', 'Estimated Delivery', 'Final Delivery', 'Quotes', 'Final Amount']]
    
    # Identifying under-served routes or services
    under_served_routes = dataframe.groupby(['Carrier', 'Pickup Location(s)', 'Delivery Location(s)']).size().reset_index(name='Under-served routes Frequency')
    under_served_routes['Under-served routes Frequency'] = under_served_routes['Under-served routes Frequency'].apply(lambda x : 1/x)
    return comparison_data, under_served_routes

# Geographical Insights
def geographical_insights(dataframe):
    # Analyzing regional demand and carrier density
    region_data = dataframe[['Pickup Location(s)', 'Delivery Location(s)', 'Carrier']]
    
    # Identifying areas with high demand and insufficient carrier coverage
    high_demand_areas = region_data.groupby('Delivery Location(s)').size().reset_index(name="Service Demand frequency")
    insufficient_coverage_areas = region_data.groupby('Pickup Location(s)')['Carrier'].nunique().reset_index(name="Carrier Coverage Density")
    
    return high_demand_areas, insufficient_coverage_areas

def no_delay_ratio(df, carrier, delivery_loc, pickup_loc):
    filtered = df[(df['Delivery Location(s)'] == delivery_loc) & (df['Pickup Location(s)'] == pickup_loc) & (df['Carrier'] == carrier)]
    if len(filtered) == 0:
        return 0
    return len(filtered[filtered["Time Difference mean"] >= 0]) / len(filtered)


def get_current_date():
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y-%m-%d")  # Adjust the format as needed
    return formatted_date

def clean_vehicle_count(df):
    return pd.to_numeric(df['Vehicle Count'].astype('str').str.extract('(\d+)')[0])

def clean_pickup_delivery_location(df):
    df['Delivery Location(s)']=df['Delivery Location(s)'].apply(lambda x : x.replace(' ',''))
    df['Pickup Location(s)']=df['Pickup Location(s)'].apply(lambda x : x.replace(' ',''))
    return df

def update_features_into_google_sheet(data):
    file_name = 'agney-gsheets.json'#'/home/agneynalapat123/agney-gsheets.json'
    creds = ServiceAccountCredentials.from_json_keyfile_name(file_name,scope)
    client = gspread.authorize(creds)
    #Fetch the sheet
    sh = client.open('Eblock Transportjobs')
    #sheet=sh.add_worksheet(title="test9", rows="5000", cols="20")
    sheet_deliveries=sh.worksheet("features")

    work_df = sheet_deliveries.get_all_values()

    if len(work_df)-1==len(data):
        logger.info("NO update needed.")
        return 
    # Clear existing data in the worksheet
    sheet_deliveries.clear()
    # Add the DataFrame to the worksheet starting from cell A1
    data.dropna(subset=['Carrier'],inplace=True)
    data=data.sort_values("No Delay Ratio")
    #set the data frame for updated values
    gd.set_with_dataframe(sheet_deliveries,data)
    logger.info("Updated google sheet!")
    return 

def TransportAnalysis(df):
    df = df.fillna(0)
    df = to_date(df)
    df['Vehicle Count']=clean_vehicle_count(df)
    df = carrier_routes(df)
    ###############################################################
    #########added carrier performance 
    logger.info('Analysing Performance of carrier...')
    performance_data,carrier_popularity=carrier_performance_assessment(df)
    df=pd.merge(df , performance_data, on=["Carrier"])
    df=pd.merge(df , carrier_popularity, on=["Carrier"])
    #######################################################################
    df['Time Difference'] = pd.to_numeric(df['Time Difference'], errors='coerce')
    df["Total Trips"] = df[["Carrier", "Pickup Location(s)", "Delivery Location(s)"]].apply(lambda x:count_trips(df, x["Carrier"], x["Pickup Location(s)"], x["Delivery Location(s)"]), axis=1)
    df["Unique Routes"] = df["Carrier"].apply(lambda x:count_unique_routes(df, x))
    ################################################################
    #######added buyer and route analysis
    frequent_route,frequent_buyer=route_and_buyer_analysis(df)
    logger.info(frequent_route.sort_values('Route Frequency',ascending=False)[:5])
    #merge both result into the dataframe
    logger.info('Analysing route and buyers...')
    df=pd.merge(df,frequent_route,on=['Pickup Location(s)', 'Delivery Location(s)'])
    df=pd.merge(df,frequent_buyer,on=['Pickup Location(s)', 'Delivery Location(s)',"Buyer"])
    #############################################################
    ########pricing and quote analysis##########################
    logger.info('Analysing Quotes Variability and cost per vehicle...')
    quote_variability, lower_price_carrier, cost_per_vehicle_df=pricing_and_quote_analysis(df)
    df = pd.merge(df,quote_variability,on=['Carrier'])
    df['Cost per Vehicle'] = cost_per_vehicle_df['Cost per Vehicle']
    ###########################################################
    ########under-served route and comparision df#####################
    logger.info('Analysing niche opportunities...')
    comparision_df,under_served_route_df=carrier_comparison_and_niche_opportunities(df)
    df=pd.merge(df,under_served_route_df,on=['Carrier','Pickup Location(s)', 'Delivery Location(s)'])
    #########################################################################
    ########geological insights########################################
    logger.info('Analysing geological performance...')
    high_demand_area,insufficient_coverage_areas=geographical_insights(df)
    df=pd.merge(df,high_demand_area,on=['Delivery Location(s)'])
    df=pd.merge(df,insufficient_coverage_areas,on=['Pickup Location(s)'])
    ########################################################################
    ##Group the data
    grouped = df.groupby(['Carrier', 'Pickup Location(s)', 'Delivery Location(s)'])

    # Calculate the mean and std using custom std function
    result = grouped.agg({
                          'Final Amount': ['mean', custom_std, "min", "max"],
                          'Time Difference': ['mean', custom_std], 
                          'Total Trips': 'max', 'Unique Routes': 'max',
                          'Route Frequency':'max',
                          'Buyers Frequency':['max','min'],
                          'Carrier Popularity Index' : 'first',
                          ## leave the values as it is,performs no agg on it
                          'Pickup Time'   : 'first',
                          'Delivery Time' : 'first',
                          ## selecting max quote variablity 
                          'Quote Variability' : 'max',
                          'Cost per Vehicle' : ['min','max'],
                          'Under-served routes Frequency' : 'first',
                           'Service Demand frequency' : 'first',
                            'Carrier Coverage Density' : 'first'
                         }
                        ).reset_index()
    result.columns = [' '.join(col).strip() if col[1]!='first' else col[0] for col in result.columns.values ]
    # Sort the result
    result["No Delay Ratio"] = result.apply(lambda row: no_delay_ratio(result,row['Carrier'],row['Delivery Location(s)'], row['Pickup Location(s)']), axis=1)
    update_features_into_google_sheet(result)
    result = result.sort_values(by='Final Amount mean')
    
    #sort
    frequent_buyer=frequent_buyer.sort_values("Buyers Frequency", ascending=False)
    frequent_route=frequent_route.sort_values('Route Frequency',ascending=False)
    #stores data for current active session one time to overcome delays
    st.session_state.not_filter_carrier_data=result
    st.session_state.frequent_route =frequent_route
    st.session_state.frequent_buyer =frequent_buyer
    st.session_state.lower_price_carriers=lower_price_carrier
    # Clear existing data in the worksheet
    return result,frequent_buyer,frequent_route,lower_price_carrier

def get_carriers(pickup_loc, delivery_loc):
    df = get_data()
    df.dropna(subset=['Carrier'],inplace=True)
    df = clean_pickup_delivery_location(df)
    total_unique_combinations = len(df.groupby(['Pickup Location(s)', "Delivery Location(s)"]))
    if df is None:
        return pd.DataFrame({})
    if all(key in st.session_state for key in ['not_filter_carrier_data', 'frequent_buyer', 'frequent_route','lower_price_carriers']):   
        final_data=st.session_state.not_filter_carrier_data
        frequent_route=st.session_state.frequent_route
        frequent_buyer = st.session_state.frequent_buyer 

    else:
        ##Transport analysis 
        final_data, frequent_buyer,frequent_route,lower_price_carrier= TransportAnalysis(df)
    # ensures the flexibility of inputs
    delivery_loc = delivery_loc.replace(' ','')
    pickup_loc   =  pickup_loc.replace(' ','')
    
    ## filter the data for given locations 
    final_data=final_data[(final_data['Delivery Location(s)']==delivery_loc) & (final_data['Pickup Location(s)']==pickup_loc)]
    final_data["Trips / Total Unique Routes"] = final_data["Total Trips max"] / total_unique_combinations
    frequent_route['route']=frequent_route['Pickup Location(s)'] + '-' + frequent_route['Delivery Location(s)']
    # store filter data in current session
    st.session_state.carrier_data_filtered = final_data.sort_values("No Delay Ratio", ascending=False)
    #return 
    return final_data.sort_values("No Delay Ratio", ascending=False),frequent_buyer,frequent_route
            
# Getting Quotes

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return dict()

def create_quote_db():
    quotes = []
    df = get_data()
    if df is None:
        return pd.DataFrame({})
    df["Quotes"] = df["Quotes"].apply(lambda x : safe_literal_eval(x) if not x is np.nan else dict())
    for idx, row in df.iterrows():
        for key in row["Quotes"].keys():
            
            row["Quotes"][key] = pd.Series(row["Quotes"][key]).replace('[^\d.]', '', regex=True).values[0]
            row["Quotes"][key] = int(row["Quotes"][key])
            quotes.append({
                "FromZIP": row["FromZIP"],
                "ToZIP": row["ToZIP"],
                "Carrier": key,
                "Qoute Amount": row["Quotes"][key],
            })
    st.session_state.quotes_data = pd.DataFrame(quotes)
    return pd.DataFrame(quotes)

def execute_sql(sql):
    try:
        return sqldf(sql, globals())
    except Exception as e:
        logger.error("Error occured while executing sql")
        logger.error(e)
        return None

# UI 

def apply_filters(df):
    
    if st.session_state.get("enable_filtering", False):
        logger.info("Filtering Enabled")
        filtered_df = df.copy()
        for column in df.columns:
            if st.sidebar.checkbox(f"Enable {column} Filtering", key=f"enable_{column}_filtering"):
                if is_numeric_dtype(df[column]):
                    min_value, max_value = df[column].min(), df[column].max()
                    start, end = st.sidebar.slider(f"{column} Range", min_value, max_value, (min_value, max_value))
                    filtered_df = filtered_df[(filtered_df[column] >= start) & (filtered_df[column] <= end)]
                elif is_object_dtype(df[column]):
                    all_options = df[column].unique()
                    if st.sidebar.checkbox("Select All", key=f"select_all_{column}"):
                        default = all_options
                    else:
                        default = []
                    options = st.sidebar.multiselect(f"Select {column}", all_options, default=default)
                    filtered_df = filtered_df[filtered_df[column].isin(options)]  
                    
        return filtered_df
    else:
        return df


st.set_page_config(page_title="Transport Analysis", page_icon="📄", layout="centered")

if 'enable_filtering' not in st.session_state:
    st.session_state.enable_filtering = False

with st.sidebar:
    selected = option_menu(
    menu_title = "Transport Analysis",
    options = ["Quotes","Analysis", "findings", "Executive Summary",'Features Description','Ask Our Assistant'],
    icons = ["receipt","bar-chart" ,"lightbulb","receipt","lightbulb","chat"],
    menu_icon = "truck-front-fill",
    default_index = 0,
    orientation = "vertical",
)
st.sidebar.checkbox("Enable Filtering", key="enable_filtering")

if selected == "Ask Our Assistant":
    st.header("Transport Chatbot")
    with st.chat_message("assistant"):
        st.write("Hello,how may I help you?")

    if "message" not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("what is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.message.append({"role": "user", "content": prompt})

        with st.chat_message('assistant'):

            try:
                response = chat_request(prompt)
                st.markdown(response["data"])
            except Exception as e:
                logger.info(f'Error while sending request to gpt-4 API {"status": "error", "data": str(e)}')

        st.session_state.message.append({"role": "assistant", "content": response["data"]})

if selected == 'Features Description':
    Description="""# Features Definition

## Overall Analysis Features

### 🌐 Most Searched Routes/Frequent Routes
Calculated based on the entire dataset.

### 🕵️‍♂️ Most Frequent Buyers
Calculated based on the entire dataset.

## Grouped Analysis Features (Grouped by Carrier, FromZIP, ToZIP)

### 💰 Final Amount Mean
Represents the average final amount charged by the carrier for the specified route. A lower mean final amount may be desirable for cost-effectiveness.

### 📊 Final Amount Custom Standard Deviation
This custom standard deviation of the final amount provides insights into the variability of pricing. Lower variability may indicate more consistent pricing.

### 🕒 Time Difference Mean
The average time difference between estimated pickup/delivery and final pickup/delivery. A carrier with lower mean time differences is likely to be more reliable in terms of meeting estimated times.

### 🚌 Total Trips Max
The maximum total trips made by the carrier on the specified route. More trips may suggest a carrier's popularity or experience on that route.

### 🌍 Unique Routes Max
The maximum number of unique routes covered by the carrier. A carrier covering more diverse routes may offer more comprehensive services.

### 🔄 Route Frequency Max 
The Route Frequnecy indicate the traffic from the specified or how frequently the route is being selected.

### 🛍️ Buyers Frequency Max and Min
Grouped by Buyer, FromZIP, ToZIP, it indicates how frequent the buyer is on that route. The frequency of buyers can indicate the carrier's relationship with specific buyers. Higher frequency suggests a strong partnership.

### 🚚 Pickup Time and Delivery Time
Carriers with lower mean pickup and delivery times are generally more efficient and reliable in meeting estimated times. Users can use this information to assess carriers based on their historical performance in terms of timely pickups and deliveries.

### 💬 Quote Variability Max
Maximum quote variability can be an indicator of how much pricing may vary. Lower variability may be preferred for predictable costs.

### 💵 Cost per Vehicle Min and Max
The cost per vehicle gives an idea of the carrier's efficiency. Lower costs may indicate cost-effectiveness.

### 🌐 Under-served Routes Frequency
Indicates the carrier's involvement in under-served routes. A higher frequency suggests a carrier's commitment to less popular routes.

### 🌎 Regional Demand Frequency and Coverage Density
Indicates the carrier's presence in high-demand areas and coverage density. A carrier covering high-demand areas may be preferable.

## No Delay Ratio
- 🚚 Ratio = 1: The carrier is consistently delivering on time or with positive time differences on the specified route. This is generally considered favorable.
- ⏳ Ratio = 0: The carrier is consistently delivering late or with negative time differences on the specified route. This may be a cause for concern, especially if on-time delivery is crucial for the business.
"""
    st.markdown(Description)


if selected == "findings":
    if all(key in st.session_state for key in ['carrier_data_filtered', 'frequent_buyer', 'frequent_route','lower_price_carriers']):    
        result_data=st.session_state.carrier_data_filtered
        frequent_route=st.session_state.frequent_route
        frequent_buyer = st.session_state.frequent_buyer 
        lower_price_carriers= st.session_state.lower_price_carriers

        now=get_current_date()
        st.title("Report Findings")
        ##################################################################
        carrier_data=result_data["Carrier"].tolist()
        carrier_on_this_route_formate=f""" 
    #### carrier operating on this route
    ##### As of {now}, {" , ".join(carrier_data)} are carrier operating on this route.
    """
        st.info(carrier_on_this_route_formate)
        ##################################################################
        carrier_lower_charge=result_data.sort_values('Final Amount mean',ascending=False)['Carrier'].tolist()[0]
        carrier_lower_charge_formate=f"""
    #### Carrier that charge lower price on this route
    ##### As of {now}, {carrier_lower_charge} is the carriers that is charging lower price for its servvice on this route.
    """
        st.info(carrier_lower_charge_formate)
    
        carrier_with_flexibilty=result_data.sort_values('Quote Variability max',ascending=False)['Carrier'].tolist()[0]
        carrier_with_flexibilty_formate=f"""
    #### Carrier that provide flexibilty in pricing or provide variable services
    ##### As of {now}, {carrier_with_flexibilty} is a carrier that provide various shipping services on Quotes on this route.
    """
        st.info(carrier_with_flexibilty_formate)


    else:
        st.warning("Please Enter the details in Analysis page to get the Report for Respective route",icon="🤖")

if selected == "Executive Summary":
    with st.spinner('Generating Report...Please wait'):
        if all(key in st.session_state for key in ['not_filter_carrier_data', 'frequent_buyer', 'frequent_route','lower_price_carriers']): 
            result_data=st.session_state.not_filter_carrier_data
            frequent_route=st.session_state.frequent_route
            frequent_buyer = st.session_state.frequent_buyer 
            lower_price_carriers= st.session_state.lower_price_carriers
        else:
            logger.info('No session data stored,retrieving data...')
            dataframe=get_data()
            result_data,frequent_buyer,frequent_route,lower_price_carriers=TransportAnalysis(dataframe)

        st.success(f"Sucessfully Generated Report", icon="✅")
        # Executive Summary
        st.title("Executive Summary 📊")
        st.title("Brief overview of key findings.📈")
        st.subheader("Insights and key metrics summarized below.")

        # popular route Analysis
        st.subheader("Insights into popular routes.")
        frequent_route['route']=frequent_route['Pickup Location(s)'] + '-' + frequent_route['Delivery Location(s)']
        formatted_routes = '\n'.join([f"- {route} - {frequency}" for route,frequency in zip(frequent_route['route'][:5],frequent_route['Route Frequency'][:5])])
        frequent_route_formate=f"""These are the most frequently used Route AS of {get_current_date()} \n {
            formatted_routes}"""
        st.info(frequent_route_formate)

        # popular carrier
        st.subheader("Insight into most Used Carrier")
        carrier_popularity_grouped=result_data.groupby('Carrier').agg({'Carrier Popularity Index': 'first'}).reset_index()
        carrier_popularity_grouped.sort_values('Carrier Popularity Index',inplace=True,ascending=False)
        formatted_carrier_popularity = '\n'.join([f"- {carrier} - {index}" for carrier ,index in zip(carrier_popularity_grouped['Carrier'][:5],carrier_popularity_grouped['Carrier Popularity Index'][:5])])
        carrier_popularity_formate=f"""These are the most used Carrier AS of {get_current_date()} \n {
            formatted_carrier_popularity}"""
        st.info(carrier_popularity_formate)

        carrier_popularity_threshold=2   #filter out to get popular carriers only
        # most frequent buyer
        st.subheader("Insight into most frequent Buyers")
        formatted_frequent_buyers = '\n'.join([f"- {buyer} - {freq}" for buyer,freq in zip(frequent_buyer['Buyer'][:5],frequent_buyer['Buyers Frequency'][:5])])
        frequent_buyers_formate=f"""These are most frequent buyers AS of {get_current_date()} \n {
            formatted_frequent_buyers}"""
        st.info(frequent_buyers_formate)

        # Carrier lower price
        st.subheader("Insight into Carrier Providing Service at lower Price")
        formatted_low_price_carrier = '\n'.join([f"- {carrier} - ${mean_amount}" for carrier,mean_amount in zip(lower_price_carriers['Carrier'],lower_price_carriers['Minimum Mean Final Amount'])])
        low_price_carrier_formate=f"""These are the Carriers charging lower price for their services AS of {get_current_date()} \n {
            formatted_low_price_carrier}"""
        st.info(low_price_carrier_formate)

        # Operational or flexibility Efficiency
        st.subheader("Insight into Flexibility in Services Offered by frequent Carriers")
        quote_variability=result_data.groupby('Carrier').agg({'Quote Variability max':'first','Carrier Popularity Index': 'first'}).reset_index()
        max_quote_variability=quote_variability.sort_values('Quote Variability max',ascending=False)
        max_quote_variability=max_quote_variability[max_quote_variability['Carrier Popularity Index']>=carrier_popularity_threshold]
        min_quote_variability=quote_variability.sort_values('Quote Variability max')
        min_quote_variability=min_quote_variability[min_quote_variability['Carrier Popularity Index']>=carrier_popularity_threshold]
        formatted_max_quote_carrier = '\n'.join([f"- {carrier}          - {qoute_val}" for carrier,qoute_val in zip(max_quote_variability['Carrier'][:5],max_quote_variability['Quote Variability max'][:5])])
        formatted_min_quote_carrier = '\n'.join([f"- {carrier}          - {qoute_val}" for carrier,qoute_val in zip(min_quote_variability['Carrier'][1:5],min_quote_variability['Quote Variability max'][1:5])])
        quote_formatted=f"""These are the frequent Carriers offering high pricing flexibility based on Quotes offered AS of {get_current_date()} \n {
            formatted_max_quote_carrier} \n \n These are the frequent Carriers offering low pricing flexibility based on Quotes offered \n {
            formatted_min_quote_carrier} 
"""
        st.info(quote_formatted)

        ## reliability of the carriers
        st.subheader("Insight into Frequent Carrier with On-Time Shipping or most reliable")
        no_delay_carrier=result_data.groupby('Carrier').agg({'No Delay Ratio' : 'mean','Carrier Popularity Index': 'first'}).reset_index()
        no_delay_carrier = no_delay_carrier.sort_values('No Delay Ratio')
        no_delay_carrier=no_delay_carrier[no_delay_carrier['Carrier Popularity Index']>=carrier_popularity_threshold]
        formatted_no_delay_carrier = '\n'.join([f"- {carrier}" for carrier in no_delay_carrier['Carrier'][:4]])
        no_delay_carrier_formate=f"""Top Frequent Carriers that deliver shipments on time or with a positive time difference for a overall route AS of {get_current_date()} \n {
            formatted_no_delay_carrier}"""
        st.info(no_delay_carrier_formate)

        st.subheader("Insights into Carrier Charging Maximum and Minimum Cost per vehicle")
        #sort data by under-served route
        max_cost_per_vehicle=result_data.groupby(['Carrier']).agg({'Cost per Vehicle max' : max}).reset_index()
        #sort to get high cost per vehicle values at top
        min_cost_per_vehicle=max_cost_per_vehicle.sort_values('Cost per Vehicle max')
        max_cost_per_vehicle=max_cost_per_vehicle.sort_values('Cost per Vehicle max',ascending=False)
        formatted_max_cost_per_vehicle = '\n'.join([f"- {carrier} - ${per_cost}" for carrier,per_cost in zip(max_cost_per_vehicle['Carrier'][:5],max_cost_per_vehicle['Cost per Vehicle max'][:5])])
        formatted_min_cost_per_vehicle = '\n'.join([f"- {carrier} - ${per_cost}" for carrier,per_cost in zip(min_cost_per_vehicle['Carrier'][:5],min_cost_per_vehicle['Cost per Vehicle max'][:5])])
        
        cost_per_vehicle_formate=f"""Highest cost charged by Carrier per vehicle AS of {get_current_date()} \n {
            formatted_max_cost_per_vehicle} \n \n Lowest cost Charged by carrier per vehicle AS of {get_current_date()}\n{formatted_min_cost_per_vehicle}"""
        st.info(cost_per_vehicle_formate)

        # under-served route 
        st.subheader("Insights into under-served routes.")
        #sort data by under-served route
        under_served_route=result_data.groupby(['Pickup Location(s)', 'Delivery Location(s)']).agg({'Under-served routes Frequency' : 'max'}).reset_index()
        under_served_route['route']=under_served_route['Pickup Location(s)'] + '-' + under_served_route['Delivery Location(s)']
        under_served_route.sort_values('Under-served routes Frequency',ascending=False,inplace=True)
        formatted_under_served_route = '\n\n'.join([f"- {route} -{frequency}" for route,frequency in zip(under_served_route['route'][:5],under_served_route['Under-served routes Frequency'][:5])])
        under_served_route_formate=f"""These are the most under-served routes AS of {get_current_date()} \n\n {
            formatted_under_served_route}"""
        st.info(under_served_route)


if selected == "Analysis":
    st.title("Analysis")

    try: 
    # Wrap the input fields in st.form
        with st.form(key='analysis_form'):
            from_ = st.text_input("Pickup Location(s)", key="from", placeholder="Enter Pickup Location(e.g. Surrey,BC)")
            to = st.text_input("Delivery Location(s)", key="to", placeholder="Enter Delivery Location(e.g. Surrey,BC)")
            num = st.text_input("Number of Carriers", key="num", placeholder="Enter Number of Carriers")
            submit_button = st.form_submit_button("Run Analysis")
            
        
        if submit_button:
            if from_ and to and num:
                with st.spinner('Getting Carriers...'):
                    carriers,frequent_buyer,frequent_route = get_carriers(from_, to)
                    carrier_df = apply_filters(carriers.head(int(num)))
                    st.success(f"{len(carrier_df)} Carriers Retrived", icon="✅")
                    # carrier_place.dataframe(carrier_df)

                    # with c1:
                    st.subheader("Most Frequently used Routes")
    
                    # frequent_route_sorted=frequent_route.sort_values(by="Route Frequency",ascending=False)
                    chart_c1 = alt.Chart(frequent_route[:5]).mark_bar().encode( 
                    x=alt.X('route:N', sort='-y'), 
                    y='Route Frequency:Q',
                    color=alt.Color('route:N') 
                    )
                    st.altair_chart(chart_c1, use_container_width=True)
                    # with c2:
                    #frequent_buyer_sorted = frequent_buyer.sort_values(by="Buyers Frequency", ascending=False)
                    st.subheader("Buyers Selecting these Route frequently")
                    filtered_by_location=frequent_buyer[(frequent_buyer['Pickup Location(s)']==from_) & (frequent_buyer['Delivery Location(s)']==to) ]
                    chart_c2 = alt.Chart(filtered_by_location[:5]).mark_bar().encode(
                        x=alt.X('Buyer:N', sort='-y'), 
                        y='Buyers Frequency:Q',
                        color=alt.Color('Buyer:N') 
                    )
                    st.altair_chart(chart_c2, use_container_width=True)

                    c3,c4 = st.columns(2)
                    c5,c6 = st.columns(2)
                    c7,c8 = st.columns(2)
                    c9,c10 = st.columns(2)
                    c11,c12 = st.columns(2)
                    c13,c14 = st.columns(2)
                    with c3:
                        #carrier_df_sorted=carrier_df.sort_values("Pickup Time",ascending=True).head(int(num))
                        carrier_df['Pickup Time (seconds)'] = carrier_df['Pickup Time'].dt.total_seconds()
                        st.subheader("Evaluation of carriers based on pickup Time")
                        chart_c3 = alt.Chart(carrier_df).mark_bar().encode(
                        x=alt.X('Carrier:N', sort='-y'),
                        y='Pickup Time (seconds)',
                        color=alt.Color('Carrier:N', legend=None) 
                    )
                        st.altair_chart(chart_c3, use_container_width=True)
                    with c4:
                        carrier_df['Delivery Time (seconds)'] = carrier_df['Delivery Time'].dt.total_seconds()
                        #carrier_df_sorted=carrier_df.sort_values("Delivery Time",ascending=True).head(int(num))
                        st.subheader("Evaluation of carriers based on Delivery times")
                        chart_c4 = alt.Chart(carrier_df).mark_bar().encode(
                        x=alt.X('Carrier:N', sort='-y'),
                        y='Delivery Time (seconds):T',
                        color=alt.Color('Carrier:N', legend=None) 
                    )
                        st.altair_chart(chart_c4, use_container_width=True)

                    with c5:
                        st.subheader("Evaluation of Max quote variability for this routes")
                        # Quote_Variability_sorted=carrier_df.sort_values('Quote Variability max',ascending=False)
                        chart_c5 = alt.Chart(carrier_df).mark_bar().encode(
                        x=alt.X('Carrier:N', sort='-y'),
                        y='Quote Variability max:Q',
                        color=alt.Color('Carrier:N', legend=None) 
                    )
                        st.altair_chart(chart_c5, use_container_width=True)
                    
                    with c6:
                        st.subheader("No Delay Ratio")
                        chart_no_delay_ratio = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='No Delay Ratio:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        
                        st.altair_chart(chart_no_delay_ratio, use_container_width=True)

                    with c7:
                        st.subheader("Time Difference mean")
                        # Time Difference mean
                        chart_time_difference_mean = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='Time Difference mean:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        
                        st.altair_chart(chart_time_difference_mean, use_container_width=True)


                    with c8:
                        st.subheader("Time Difference Standard Deviation")
                        # Time Difference Standard Deviation
                        chart_time_difference_std = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='Time Difference custom_std:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        
                        st.altair_chart(chart_time_difference_std, use_container_width=True)

                    with c9:
                        st.subheader("Final Amount mean")
                        # Final Amount mean
                        chart_final_amount_mean = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='Final Amount mean:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        
                        st.altair_chart(chart_final_amount_mean, use_container_width=True)

                    with c10:
                        st.subheader("Total Trips")
                        # Total Trips
                        chart_total_trips = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='Total Trips max:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        
                        st.altair_chart(chart_total_trips, use_container_width=True)

                    with c11:
                        st.subheader("Unique Routes")
                        # Unique Routes
                        chart_unique_routes = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='Unique Routes max:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        st.altair_chart(chart_unique_routes, use_container_width=True)
                    with c12:
                        st.subheader("Under-served Routes frequency")
                        # Unique Routes
                        chart_unique_routes = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='Under-served routes Frequency:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        st.altair_chart(chart_unique_routes, use_container_width=True)

                    with c13:
                        st.subheader("Cost per vehicle max")
                        # Unique Routes
                        chart_unique_routes = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='Cost per Vehicle max:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        st.altair_chart(chart_unique_routes, use_container_width=True)

                    with c14:
                        st.subheader("Cost per vehicle min")
                        # Unique Routes
                        chart_unique_routes = alt.Chart(carrier_df).mark_bar().encode(
                            x=alt.X('Carrier:N', sort='-y'),
                            y='Cost per Vehicle min:Q',
                            color=alt.Color('Carrier:N', legend=None) 
                        )
                        st.altair_chart(chart_unique_routes, use_container_width=True)
                        
            else:
                st.warning("Please enter all fields", icon="⚠️")

        if 'carrier_data' in st.session_state:
            st.image('TRANSPORT/sql_query.jpg')
            query = st.text_input("Query", key="query", placeholder="Enter SQL Query")
            st.text('Note: Refer to the table as carrier_df')
            st.text('Eg: select Carrier, Pickup Location(s), Delivery Location(s) from not_filter_carrier_data where `Final Amount mean` > 100')
            if st.button("Execute Query", key="clear"):
                if query:
                    with st.spinner('Executing Query...'):
                        result = execute_sql(query)
                        if result is None:
                            st.error("Invalid Query")
                        else:
                            st.success(f"Query Executed Successfully", icon="✅")
                            st.dataframe(result)
                else:
                    st.warning("Please enter a query", icon="⚠️")
        
    except Exception as e:
        logger.error("Error occured while getting carriers")
        logger.error(e)
        st.error("Something went wrong, please try again later")

elif selected == "Quotes":
    st.title("All Quotes")
    left, right = st.columns(2)
    with left:
        if "quotes_data" not in st.session_state:
            df_placeholder = st.warning("Data Loading", icon="ℹ")
            df_quotes = apply_filters(create_quote_db())
            df_placeholder.dataframe(df_quotes)
        else:
            df_quotes = apply_filters(st.session_state.quotes_data)
            df_placeholder = st.dataframe(df_quotes)
        query = st.text_input("Query", key="query", placeholder="Enter SQL Query")
        st.text('Note: Refer to the table as df_quotes')
        st.text('Eg: select * from df_quotes where `Qoute Amount` > 100')
    
        if st.button("Execute Query", key="clear"):
            if query:
                with st.spinner('Executing Query...'):
                    result = execute_sql(query)
                    if result is None:
                        st.error("Invalid Query")
                    else:
                        st.success(f"Query Executed Successfully", icon="✅")
                        st.dataframe(result)
            else:
                st.warning("Please enter a query", icon="⚠️")
    with right:
        st.text_area("Note", "Additional Insights About the data`", key="note", height=210)
