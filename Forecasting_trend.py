import pandas as pd
import streamlit as st
import numpy as np
from nltk.corpus import stopwords
from pytrends.request import TrendReq
import pandas as pd
import time
import re
startTime = time.time()
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
pytrend = TrendReq(hl='en-GB', tz=360)

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
#_max_width_()
st.title("Forecasting the furure trending topics")
st.markdown("Future is yours!")

# # colnames = ["keywords"]
# # df = pd.read_csv("gtkeywords.csv", names=colnames)
# # df2 = df["keywords"].values.tolist()
# # df2.remove("Keywords")

@st.cache
def genData(Trendkey):
    dataset = []
    for x in range(0,1):
        # keywords = [df2[x]]
         pytrend.build_payload(
         Trendkey,
         cat=0,
         timeframe='2010-01-01 2021-06-06',geo='GB')
         data = pytrend.interest_over_time()
         if not data.empty:
              data = data.drop(labels=['isPartial'],axis='columns')
              dataset.append(data)
    return dataset

def showSingleKeyTrend(key):
    result = pd.concat(genData(key), axis=1)
    st.write("We ran into problemw")
    result = pd.concat(genData(key), axis=1)
    result.to_csv('trends.csv')

    executionTime = (time.time() - startTime)
    print('Execution time in sec.: ' + str(executionTime))

    df=pd.read_csv("trends.csv")
    st.write("Displaying data")

    st.write(df.size)
    st.write(df)

    st.line_chart(df)

keys=['deep learning', 'machine learning','python','artificial intelligence','IPL','covaxin']

option = st.selectbox('Which topic do you like?',keys)

'You selected: ', option
key=[]
key.append(option)


@st.cache
def ARIMA_MODEL():
    series = pd.read_csv('trends.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    X = series.values
    size = int(len(X) * 0.76)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5,2,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    #rmse = sqrt(mean_squared_error(test, predictions))
    #print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    tp=pd.DataFrame(test,columns=['test'])
    tp['prediction']=predictions
    return tp

if(st.button("Click to get trend of "+key[0])):
    showSingleKeyTrend(key)
    st.title("Displaying plot of arima mode predictions")
    tp=ARIMA_MODEL()
    #acuracy=accuracy_score(tp['test'], tp['prediction'])
    for x in tp.index:
        if tp.loc[x,"prediction"]<0:
            tp.loc[x,"prediction"]=0
    st.line_chart(tp)
    #st.write(accuracy)
    
#----------------------------------------------------------------------------------------------------------------
#trend of many keywords
st.title("Generating whole list of data")

txt=st.text_area("Enter key words to search:(seperate with ,)")
multiKeys=txt.split(",")
st.write(*multiKeys)

if(st.button("Click to generate result")):
    MultiRes = pd.concat(genData(multiKeys), axis=1)
    MultiRes.to_csv('Multitrends.csv')

    dfMulti=pd.read_csv("Multitrends.csv")
    st.write("Displaying data")

    st.write(dfMulti.size)
    st.write(dfMulti.head(10))

    st.line_chart(dfMulti)

#-----------------------------------------------------------------------------------
#related key words and there data

q_rel=st.text_area("Enter key word to search related keywords:")
q_rel=q_rel.split(",")
countries=['world','india','usa','us','mexico','uk','france','china','germany','japan','france','russia','pakistan','rising', 'query', 'value','price','near']

def get_related_queries(query):
    pytrend.build_payload(query)
    related_queries=pytrend.related_queries()
    a=related_queries.values()
    a=str(a)
    a=(a.split('\n'))
    # a.remove(a[0])
    a=a[1:100]
    st.write("Found related queris:")
    for i in range(1,10):
        st.write(a[i])
    rel_keys=[]
    kwords=[]
    #removing all characters and digits and retainign only alpha's
    for i in a:
        txt = i
        txt.strip("")
        #print(txt)
        #Find all lower case characters alphabetically between "a" and "m":

        x = re.findall("[a-z]+", txt)
        for i in x:
            kwords.append(i)
    #splitting the key word if two words are present(ex: deep learning)
    query=query[0].split(' ')
    #removing stop words running twice remove all things which are remained in first iteration
    for _ in range(2):
        for i in kwords:
            if(i==query[0]):
                kwords.remove(query[0])
            if(len(query)>1 and i==query[1] ):
                kwords.remove(query[1]) 
    st.title("Generated Related  list of keywords")
    st.write(*kwords)

    #removing stop words 
    en_stops = set(stopwords.words('english'))
    for word in kwords: 
        if word not in rel_keys and word not in en_stops and word not in countries:
            rel_keys.append(word)

    st.write("Keys after removing stop words:")
    st.write(*rel_keys)
    return rel_keys


if(st.button("Get related queries of "+q_rel[0],help="Click on the button to get all the queries related to"+q_rel[0])):
    rel_topics=get_related_queries(q_rel)
    rel_topics.insert(0,q_rel[0])
    rel_topics=rel_topics[:5]
    df_rel_topics_res = pd.concat(genData(rel_topics), axis=1)
    df_rel_topics_res.to_csv('rel_topics_trends.csv')

    df_rel_topics=pd.read_csv("rel_topics_trends.csv")
    st.write("Displaying data")

    st.write(df_rel_topics.size)
    st.write(df_rel_topics.head(10))

    st.line_chart(df_rel_topics)

    st.write(df_rel_topics.corr())


