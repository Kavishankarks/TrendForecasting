#importing the required modules
import plotly.express as px # to plot the time series plot
from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf 
from pandas import DataFrame
from pytrends.request import TrendReq
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('ggplot')
import re
import tensorflow as tf
# tf.logging.set_verbocity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
# import altair as alt #linechart labeling in streamlit
import datetime
from PIL import Image
try:
	pytrend = TrendReq(hl='en-GB', tz=360)
except:
	error = '<p style="font-family:sans-serif; color:red; font-size: 42px;">Error  your device is not connected</p>'
	st.markdown(error, unsafe_allow_html=True)
	# st.title("Device is not connected")
import seaborn as sns

stopWords={'i': 1,'me': 1,'my': 1,'myself': 1,'we': 1,'our': 1,'ours': 1,'ourselves': 1,'you': 1,"you're": 1,"you've": 1,"you'll": 1,"you'd": 1,'your': 1,'yours': 1,'yourself': 1,
 'yourselves': 1,'he': 1,'him': 1,'his': 1,'himself': 1,'she': 1,"she's": 1,'her': 1,'hers': 1,'herself': 1,'it': 1,"it's": 1,'its': 1,'itself': 1,'they': 1,
 'them': 1,'their': 1,'theirs': 1,'themselves': 1,'what': 1,'which': 1,'who': 1,'whom': 1,'this': 1,'that': 1,"that'll": 1,'these': 1,'those': 1,'am': 1,'is': 1,'are': 1,
 'was': 1,'were': 1,'be': 1,'been': 1,'being': 1,'china': 1,'germany': 1,'japan': 1,'russia': 1,'pakistan': 1,'rising': 1,'query': 1,'value': 1,'price': 1,'near': 1,
 'has': 1,'had': 1,'having': 1, 'do': 1,'does': 1, 'did': 1,'doing': 1,'a': 1,'an': 1,'the': 1,'and': 1,'but': 1,'if': 1,'or': 1,'because': 1,
 'as': 1,'until': 1,'while': 1,'of': 1,'at': 1,'by': 1,'for': 1,'with': 1,'about': 1,'against': 1,'between': 1,'into': 1,'through': 1,'during': 1,'before': 1,
 'after': 1,'above': 1,'below': 1,'to': 1,'from': 1,'up': 1,'down': 1,'in': 1,'out': 1,'on': 1,'off': 1,'over': 1,'under': 1,'again': 1,'further': 1,'then': 1,
 'once': 1,'here': 1,'there': 1,'when': 1,'where': 1,'why': 1,'how': 1,'all': 1,'any': 1,'both': 1,'each': 1,'few': 1,'more': 1,'most': 1,'other': 1,'some': 1,
 'such': 1,'no': 1,'nor': 1,'not': 1,'only': 1,'own': 1,'same': 1,'so': 1,'than': 1,'too': 1,'very': 1,'s': 1,'t': 1,'can': 1,'will': 1,'just': 1,'don': 1,"don't": 1,
 'should': 1,"should've": 1,'now': 1,'d': 1,'ll': 1,'m': 1,'o': 1,'re': 1,'ve': 1,'y': 1,'ain': 1,'aren': 1,"aren't": 1,'couldn': 1,"couldn't": 1,'didn': 1,"didn't": 1,
 'doesn': 1,"doesn't": 1,'hadn': 1,"hadn't": 1,'hasn': 1,"hasn't": 1,'haven': 1,"haven't": 1,'isn': 1,"isn't": 1,'ma': 1,'mightn': 1,"mightn't": 1,'mustn': 1,
 "mustn't": 1,'needn': 1,"needn't": 1,'shan': 1,"shan't": 1,'shouldn': 1,"shouldn't": 1,'wasn': 1,"wasn't": 1,'weren': 1,"weren't": 1,'won': 1,"won't": 1,'wouldn': 1,
 "wouldn't": 1,'world': 1,'india': 1,'usa': 1,'us': 1,'mexico': 1,'uk': 1,'france': 1}


#generate search queries data from google trends by making API call and return dataset for called function
@st.cache
def generateSearchData(keywords):
  dfk=DataFrame(keywords,columns=['Keywords'])
  dfk.reset_index(drop=True, inplace=True)
  dfk.to_csv('keywords.csv')
  startTime = time.time()
  pytrend = TrendReq(hl='en-GB', tz=360)

  colnames = ["keywords"]
  df = pd.read_csv("keywords.csv", names=colnames)
  df2 = df["keywords"].values.tolist()
  df2.remove("Keywords")
  dataset = []

  for x in range(0,len(df2)):
      keywords = [df2[x]]
      pytrend.build_payload(
      kw_list=keywords,
      cat=0,
      timeframe='2004-01-01 2021-10-10',geo='GB')
      data = pytrend.interest_over_time()
      if not data.empty:
            data = data.drop(labels=['isPartial'],axis='columns')
            dataset.append(data)
  # return dataset
  # result = pd.concat(dataset, axis=1)
  # result.to_csv('trends.csv')

  executionTime = (time.time() - startTime)
  print("Gathering data from Google trends at time"+str(datetime.datetime.now()))
  return ('Gathering is Done and Execution time in sec.: ' + str(executionTime)), dataset

#Arima model:- a statical forecasting model
@st.cache
def ARIMA_MODEL(series):
    # series = pd.read_csv('trends.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    X = series.values
    size = int(len(X) * 0.80)
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
    # rmse = sqrt(mean_squared_error(test, predictions))
    # print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    tp=pd.DataFrame(test,columns=['test'])
    tp['prediction']=predictions
    return tp


#LSTM model for predicting univariate data which is written from scratch
def univariate_LSTM(df,data):
	st.title("LSTM model under execution...\n")
	scaler=StandardScaler()
	data=scaler.fit_transform(data.reshape(-1,1))
	def getData(data,window_size=7):
	    X=[]
	    Y=[]
	    i=0
	    while(i+window_size)<=len(data)-1:
	        X.append(data[i:i+window_size])
	        Y.append(data[i+window_size])
	        i+=1
	    assert len(X)==len(Y)
	    return X,Y
	    
	X,Y=getData(data,window_size=7)

	#Splitting data into test and train 
	st.write("Step 1. Splitting data has been done.")
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=45)

	#Defining the parameters 
	batch_size=7
	window_size=7
	hidden_layer=256
	learning_rate=0.001

	# tf.placeholder() is not compatible with eager execution
	tf.compat.v1.disable_eager_execution()

	#defining the placeholder for input and output
	input_=tf.compat.v1.placeholder(tf.float32, [batch_size, window_size,1])
	target=tf.compat.v1.placeholder(tf.float32,[batch_size,1])

	#Defining all the weights

	#input_gate weinghts
	U_input=tf.Variable(tf.compat.v1.truncated_normal([1,hidden_layer],stddev=0.05))
	W_input=tf.Variable(tf.compat.v1.truncated_normal([hidden_layer,hidden_layer],stddev=0.05))
	b_input=tf.Variable(tf.zeros([hidden_layer]))

	#forget_gate weights
	U_forget=tf.Variable(tf.compat.v1.truncated_normal([1,hidden_layer],stddev=0.05))
	W_forget=tf.Variable(tf.compat.v1.truncated_normal([hidden_layer,hidden_layer],stddev=0.05))
	b_forget=tf.Variable(tf.zeros([hidden_layer]))

	#output_gate weights
	U_output=tf.Variable(tf.compat.v1.truncated_normal([1,hidden_layer],stddev=0.05))
	W_output=tf.Variable(tf.compat.v1.truncated_normal([hidden_layer,hidden_layer],stddev=0.05))
	b_output=tf.Variable(tf.zeros([hidden_layer]))


	#weight for candidate gates
	U_g=tf.Variable(tf.compat.v1.truncated_normal([1,hidden_layer],stddev=0.05))
	W_g=tf.Variable(tf.compat.v1.truncated_normal([hidden_layer,hidden_layer],stddev=0.05))
	b_g=tf.Variable(tf.zeros([hidden_layer]))

	#output_layer weights
	V=tf.Variable(tf.compat.v1.truncated_normal([hidden_layer,1],stddev=0.05))
	b_v=tf.Variable(tf.zeros([1]))
	st.write("Step 2. Defining weights for all data has been done.")

	#Defining LSTM Cell
	def LSTM_cell(input_,prev_hidden_state,prev_cell_state):
	    #it=sigmoid(U(i)*x(i)+W(i)h(t-1)+b(i))
	    input_gate=tf.sigmoid(tf.matmul(input_,U_input)+tf.matmul(prev_hidden_state,W_input)+b_input)
	    
	    #ft=sigmoid(U(f)*x(f)+W(f)h(t-1)+b(f))
	    forget_gate=tf.sigmoid(tf.matmul(input_,U_forget)+tf.matmul(prev_hidden_state,W_forget)+b_forget)
	    
	    #ot=sigmoid(U(o)*x(o)+W(o)h(t-1)+b(o))
	    output_gate=tf.sigmoid(tf.matmul(input_,U_output)+tf.matmul(prev_hidden_state,W_output)+b_output)
	    
	    #gt=tanh(U(g)*x(t)+W(g)h(t-1)+b(g))
	    candidate_gate=tf.tanh(tf.matmul(input_,U_g)+tf.matmul(prev_hidden_state,W_g)+b_g)
	        
	    
	    ct=(prev_cell_state*forget_gate)+(input_gate*candidate_gate)
	    
	    hidden_state=output_gate*tf.tanh(ct)
	    
	    return ct,hidden_state
	st.write("Step 3. LSTM_cell execution...")
	#defining forward propogation
	y_hat=[]
	for i in range(batch_size):
	    hidden_state=np.zeros([1,hidden_layer],dtype=np.float32)
	    cell_state=np.zeros([1,hidden_layer],dtype=np.float32)
	    for t in range(window_size):
	        cell_state, hidden_state = LSTM_cell(tf.reshape(input_[i][t], (-1,1)), hidden_state, cell_state)
	# cell_state,hidden_state=LSTM_cell(tf.reshape(input[i][t],(-1,1)),hidden_state,cell_state)
	    y_hat.append(tf.matmul(hidden_state,V)+b_v)
	st.write("Step 4. LSTM forward propogation...")

	#defining backword propogation
	losses=[]
	for i in range(len(y_hat)):
	    losses.append(tf.losses.mean_squared_error(tf.reshape(target[i],(-1,1)),y_hat[i]))
	    
	    loss=tf.reduce_mean(losses)
	    
	    #performing gradient clipping to aviode exploding gradient problem
	    gradients=tf.gradients(loss,tf.compat.v1.trainable_variables())
	    clipped, _=tf.clip_by_global_norm(gradients,4.0)
	    
	    #Adam optimiser and minimize our loss function:
	    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate).apply_gradients(zip(gradients,tf.compat.v1.trainable_variables()))
	st.write("Step 5. LSTM backword propogation...\n\n")
	st.title("Training the model with 100 epochs")
	# training the LSTM Model
	session=tf.compat.v1.Session()
	session.run(tf.compat.v1.global_variables_initializer())
	epochs=20
	for i in range(epochs):
	    train_prediction=[]
	    index=0
	    epoch_loss=[]
	    while((index+batch_size)<=len(X_train)):
	        X_batch=X_train[index:index+batch_size]
	        Y_batch=y_train[index:index+batch_size]
	        #predict the price and compute the loss
	        predicted,loss_val, _=session.run([y_hat,loss,optimizer],feed_dict={input_:X_batch, target:Y_batch})
	        
	        #store the loss in the epoch list
	        epoch_loss.append(loss_val)
	        
	        #store=ing the prediction train prediction list
	        train_prediction.append(predicted)
	        
	        index+=batch_size
	    
	    #displaying the loss on the every 10th iteration
	    if(i%1==0):
	        st.write("Epoch {}, Loss:{}".format(i,np.mean(epoch_loss)))
	# Making predictions using the LSTM model

	predicted_output=[]
	i=0
	while((i+batch_size)<=len(X_test)):
	    output=session.run([y_hat],feed_dict={input_:X_test[i:i+batch_size]})
	    i+=batch_size
	    predicted_output.append(output)

	# st.write("Predicted output",predicted_output[0])

	predicted_values_test = []
	for i in range(len(predicted_output)):
	 for j in range(len(predicted_output[i][0])):
	   predicted_values_test.append(predicted_output[i][0][j])
	predictions = []

	#appending the predicted value into list
	dataset_length=len(df)
	predicted_length=len(predicted_values_test)
	for i in range(dataset_length):
	  if i <predicted_length:
	    predictions.append(predicted_values_test[i])
	  else:
	    predictions.append(None)

	#Plotting the final result
	st.title("Plotting final predictions results.")
	fig=plt.figure(figsize=(12, 10))
	plt.plot(y_test, label='Actual')
	plt.plot(predictions, label='Predicted',linestyle='dashed')
	plt.legend()
	plt.xlabel('Months')
	plt.ylabel('Trend Component')
	plt.grid()
	plt.show()
	st.pyplot(fig)

def showSingleKeyTrend():
	keywords=['None','deep learning','corona', 'python', 'network', 'image','deepavali','covaxin', 'neural', 'networks', 'google', 'pdf', 'tensorflow', 'reinforcement', 'learning', 'github', 'model', 'classification', 'book']
	option = st.selectbox('Which topic do you like?',keywords)

	'You selected: ', option
	if(option=='None'):
		option=st.text_area("Enter key word:")
	key=[]
	if(st.button("Get trend")):
		key.append(option)
		execution, dataset=generateSearchData(key)
		results=pd.concat(dataset, axis=1)
		results.to_csv('trends.csv')
		df1=pd.read_csv('trends.csv')
		df=pd.read_csv('trends.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
		st.title("Gathering and Displaying data of "+option)
		st.write(execution)
		st.write(df.size)
		st.write(df1)
		st.line_chart(df)
		st.title("ARIMA MODEL PREDICTIONS")
		st.line_chart(ARIMA_MODEL(df))
		univariate_LSTM(df,df.values)
    # univariate_LSTM(df.values)

    # result = pd.concat(genData(key), axis=1)
    # result.to_csv('trends.csv')
    # executionTime = (time.time() - startTime)
    # print('Execution time in sec.: ' + str(executionTime))
    # df=pd.read_csv("trends.csv")
    # st.write("Displaying data")
    # st.write(execution)
    # st.write(df.size)
    # st.write(df)

# st.write()
# showSingleKeyTrend()
# df=pd.read_csv('trends.csv')
# st.write(df)

def get_related_queries(query):
	try:
		pytrend.build_payload(query)
	except:
		st.title("The request failed: Google returned a response with code 400.")
		return 0
	related_queries=pytrend.related_queries()
	a=related_queries.values()
	a=str(a)
	a=(a.split('\n'))
	a=a[1:100]
	st.title("Found related queris:")
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
	#removing stop words running twice remove all things which are remained in first iterations
	for _ in range(2):
		for i in kwords:
			if(i==query[0]):
				kwords.remove(query[0])
			if(len(query)>1 and i==query[1] ):
				kwords.remove(query[1])
	st.title("Generated Related  list of keywords")
	st.write(*kwords)

	#removing stop words 
	# en_stops = set(stopwords.words('english'))
	global stopWords
	for word in kwords: 
		if word not in rel_keys and word not in stopWords:
			rel_keys.append(word)

	st.title("\nKeys after removing stop words:")
	st.write(*rel_keys)
	return rel_keys
    

# @st.cache
def get_related_queries(query):
    pytrend.build_payload(query)
    related_queries=pytrend.related_queries()
    a=related_queries.values()
    a=str(a)
    a=(a.split('\n'))
    # a.remove(a[0])
    a=a[1:100]
    queries=a[:10]
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
    
    rel_keys=[]
    #removing stop words 
    # en_stops = set(stopwords.words('english'))
    for word in kwords: 
        if word not in stopWords and word not in rel_keys:
            rel_keys.append(word)
    return queries,kwords,rel_keys[1:]



def multiKeywordGeneration():
	q_rel=st.text_area("Enter key word to search related keywords:")
	q_rel=q_rel.split(",")
	if(st.button("Get related queries of "+q_rel[0],help="Click on the button to get all the queries related to"+q_rel[0])):
		queries,kwords,rel_topics=get_related_queries(q_rel)
		st.write("Found related queris:")
		for i in range(9):
			st.write(queries[i])
		st.title("Generated Related  list of keywords")
		st.write(*kwords)	
		# rel_topics=list(set(rel_topics))
		st.title("\nKeys after removing stop words:")
		rel_topics.insert(0,q_rel[0])
		st.write(*rel_topics)
		rel_topics=rel_topics[:15]
		execution, dataset=generateSearchData(rel_topics)
		results=pd.concat(dataset, axis=1)
		results.to_csv('rel_topics_trends.csv')
		st.write(execution)
		df_rel_topics=pd.read_csv("rel_topics_trends.csv")
		st.write("Displaying data")
		st.write(df_rel_topics.size)
		st.title("Data of each individual queries")		
		st.write(df_rel_topics)
		st.title("Graph of each search queries")
		st.line_chart(df_rel_topics)
		st.title("Correlation matrix")
		st.write(df_rel_topics.corr())
		fig=plt.figure(figsize=(12,12))
		plt.rcParams["font.weight"] = "bold"
		plt.rcParams['font.size']=10
		plt.title('Correlations between the continuous features of the dataset',fontweight='bold',fontsize = 13)
		f=sns.heatmap(df_rel_topics.corr().round(2),annot=True,cmap='inferno')
		st.title("Correlation matrix visuals")
		st.write(fig)
		st.write(f)

		#take keywords having correlation aboe 0.8 and below -0.6
		st.title("Refining the Correlations")
		a=df_rel_topics.corr().iloc[0]
		keys=[]
		for x in a.index:
		  if(a[x]>0.6 or a[x]<(-0.6)):
		    keys.append(x)
		dfHighCorrelation=df_rel_topics[keys]
		# plot the correlation matrix
		fig1=plt.figure(figsize=(12,12))
		plt.rcParams["font.weight"] = "bold"
		plt.rcParams['font.size']=10
		plt.title('Correlations between the continuous features of the dataset',fontweight='bold',fontsize = 13)
		figHighCorrelation=sns.heatmap(dfHighCorrelation.corr().round(2),annot=True,cmap='inferno')
		st.title("Correlation matrix visuals")
		st.write(fig1)
		st.write(figHighCorrelation)
		#Graph of Highly Correlated keywords
		st.title("Graph of Highly related keywords")
		st.line_chart(dfHighCorrelation)

# --------------------------------------------------------------------------------------------------------------------------------------------

		#multivariate LSTM model for prediction using this data
		data=dfHighCorrelation
		st.title("\n\nMultivariate LSTM model")
		def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
		    X = []
		    y = []
		    start = start + window
		    if end is None:
		        end = len(dataset) - horizon
		    for i in range(start, end):
		        indices = range(i-window, i)
		        X.append(dataset[indices])
		        indicey = range(i+1, i+1+horizon)
		        y.append(target[indicey])
		    return np.array(X), np.array(y)

		a=[]
		for i in data.columns:
		    a.append(i)
		print(a)
		for i in data.select_dtypes('object').columns:
		    le = LabelEncoder().fit(data[i])
		    data[i] = le.transform(data[i])                         
		name=a[1]
		X_scaler = MinMaxScaler()
		Y_scaler = MinMaxScaler()
		X_data = X_scaler.fit_transform(data[a])
		Y_data = Y_scaler.fit_transform(data[[name]])
		hist_window = 48
		horizon = 10
		TRAIN_SPLIT = 150
		x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
		x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon)
		st.write('Multiple window of past history\n')
		st.write(x_train[0])
		st.write('\n Target horizon\n')
		st.write(y_train[0])
		validate = data[a[1]].tail(10)
		# data.drop(data.tail(10).index,inplace=True)

		# Prepare the training data and validation data using 
		# the TensorFlow data function, which faster and efficient way to feed data for training.
		batch_size = 256
		buffer_size = 150
		train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
		train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
		val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
		val_data = val_data.batch(batch_size).repeat() 

		# Build and compile the model
		tf.compat.v1.experimental.output_all_intermediates(True)
		lstm_model = tf.keras.models.Sequential([
		   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True), 
		                                input_shape=x_train.shape[-2:]),
		     tf.keras.layers.Dense(20, activation='tanh'),
		     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
		     tf.keras.layers.Dense(20, activation='tanh'),
		     tf.keras.layers.Dense(20, activation='tanh'),
		     tf.keras.layers.Dropout(0.25),
		     tf.keras.layers.Dense(units=horizon),
		 ])
		lstm_model.compile(optimizer='adam', loss='mse')
		st.write(lstm_model.summary())

		# Configure the model and start training with early stopping and checkpoint.Early stopping stops training when monitored 
		# loss starts increasing above the patience, and checkpoint saves the model weight as it reaches the minimum loss.
		model_path = 'Bidirectional_LSTM_Multivariate.h5'
		early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
		checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
		callbacks=[early_stopings,checkpoint]
		try:
			history = lstm_model.fit(train_data,epochs=5,steps_per_epoch=5,validation_data=val_data,validation_steps=50,verbose=1,callbacks=callbacks)
			multifig=plt.figure(figsize=(16,9))
			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])
			plt.title('Model loss')
			plt.ylabel('loss')
			plt.xlabel('epoch')
			plt.legend(['train loss', 'validation loss'])
			plt.show()
			st.write(multifig)
		except:
			st.title("\n\n\nError in size, model unable to fit and train")
		

# --------------------------------------------------------------------------------------------------------------------------------------------

	# relatedKeyword=st.text_area("Enter key to search related keywords:")
	# try:
	# 	keywords=get_related_queries(relatedKeyword)[:10]
	# 	st.title("Taking top 10 keywords")
	# 	st.write(*keywords)
	# except:
	# 	st.write("Error")
	# return 0

def header():
	st.title("Forecasting the furure trend of a topic")
	st.markdown("Future is yours!")

def homePage():
	# st.title("Forecasting is the next major thing")
	st.title("\t\t-------------Capstone project-------------")
	image = Image.open('imageTrend.jpeg')
	st.image(image, caption='find the trends')


# ---------------------------------------------------------------------------------------------------------------------------------------------
def main():
    page = st.sidebar.selectbox(
        "Select a Page",
        [
        	"Home Page",
            "Forecasting univariate trend",
            "Multivariate  approach"   
        ]
    )
    if page == "Home Page":
        homePage()

    #Second Page
    elif page == "Forecasting univariate trend":
    	st.title("Forecasting trend of given topic")
    	showSingleKeyTrend()

    elif page == "Multiple Keywords":
    	multiKeywordGeneration()

header() 
main()
