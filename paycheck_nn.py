import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd
import numpy as np

tf.config.set_visible_devices([], 'GPU')

start=time.perf_counter()

#additional parameters
p=0.25 #power of relu re-weighting
n=5 #number of goals with state variable
m=2 #number of goals without state variable
monthly_income=4500.
inflation_rate=0.02
rate_of_return=0.04/12 #set to be monthly by dividing by 12
months=120
epochs=20
batch_size=40

#credit card debt [0]
credit_card_balance=825
credit_card_interest_apr=0.1
cc_priority=9.0

#student loans [1]
student_loan_balance=80000
student_loan_interest_apr=0.04
sl_priority=2.5

#savings for mortgage down payment [2]
mortgage_down=157000
#mortgate_down savings invested in treasury
mortgage_priority=3.2

#emergency fund [3]
emergency_priority_1=9.0 #initial emergency fund, for example 3 months of salary
emergency_amount_1=1800
emergency_priority_2=1.0 #rest of emergency fund, for example 12 months; total is amount_1+amount_2
emergency_amount_2=9000

#retirement [4]
retirement_savings_needed=3719982
retirement_priority=3.0

#401K [5]
t401k_max_pct=0.13
t401k_max_priority=4.0
t401k_min_pct=0.06
t401k_min_priority=6.0
t401k_tier1_pct=1.0 #percent income is matched
t401k_tier1_upto=0.03 #up to what percent of income
t401k_tier2_pct=0.5
t401k_tier2_upto=0.03
t401k_match_max=0.045 #maximum percentage employer will match

#IRA [6]
ira_max_cont=500 #monthly ira contribution
ira_priority=3.0

def load_data():
    inf_data=pd.read_csv("CPIAUCNS.csv")
    return_data=pd.read_csv("IXIC.csv")
    treasury_data=pd.read_csv("DGS3MO.csv")

    #only start data where return data is well defined
    ind=inf_data.index[inf_data["DATE"]==return_data.iloc[0,0]].tolist()[0]
    inf_data=inf_data.iloc[ind:,:]
    
    ind=treasury_data.index[treasury_data["DATE"]==return_data.iloc[0,0]].tolist()[0]
    treasury_data=treasury_data.iloc[ind:,:]

    #only end data where inflation data is well defined
    ind=return_data.index[return_data["Date"]==inf_data.iloc[-1,0]].tolist()[0]
    return_data=return_data.iloc[:(ind+1),:]
    
    ind=treasury_data.index[treasury_data["DATE"]==inf_data.iloc[-1,0]].tolist()[0]-\
        treasury_data.index[treasury_data["DATE"]==treasury_data.iloc[0,0]].tolist()[0]
    treasury_data=treasury_data.iloc[:(ind+1),:]
    
    #calculate month by month percentage changes for data
    inflation_pct=tf.constant(inf_data.iloc[:,1].pct_change(periods=1)[1:])
    returns_pct=tf.constant(return_data.iloc[:,1].pct_change(periods=1)[1:])
    treasury_pct=tf.constant(treasury_data.iloc[1:,1]/100.)

    inflation_rates=np.zeros([len(inflation_pct)-months+1,months])
    rate_of_returns=np.zeros([len(returns_pct)-months+1,months])
    treasury_rates=np.zeros([len(treasury_pct)-months+1,months])

    for i in range(len(inflation_rates)):
        inflation_rates[i,:]=inflation_pct[i:(i+months)]
        rate_of_returns[i,:]=returns_pct[i:(i+months)]
        treasury_rates[i,:]=treasury_pct[i:(i+months)]
    return tf.constant(inflation_rates,dtype="float32"), tf.constant(rate_of_returns,dtype="float32"), tf.constant(treasury_rates,dtype="float32")
  
@tf.function(jit_compile=True)
def u_1(priority,x):
    return -1*tf.nn.relu(priority*x) 
    
@tf.function(jit_compile=True)
def u_2(priority_1,priority_2,hinge,x):
    return -1*tf.nn.relu(priority_2*x)-tf.nn.relu((priority_1-priority_2)*(x-hinge))

@tf.function(jit_compile=True)
def match_pct(pct):
    temp=tf.math.minimum(pct, t401k_tier1_upto)*t401k_tier1_pct
    temp=temp+tf.cast((pct>t401k_tier1_upto),dtype='float32')*\
        tf.math.minimum(t401k_tier2_upto, pct-t401k_tier1_upto)*t401k_tier2_pct
    return tf.math.minimum(t401k_match_max,temp)

@tf.function(jit_compile=True)
def train(model,x0,optimizer,inflation_rates,rate_of_returns,treasury_rates):
    m = len(inflation_rates)
    x = tf.repeat(x0,repeats=m,axis=0)
    loss=tf.zeros([m],dtype='float32')
    temp_income=monthly_income*tf.ones([m],dtype='float32')
    for i in tf.range(months):
        pct=model(tf.concat([x,tf.transpose([inflation_rates[:,i]]),\
            tf.transpose([rate_of_returns[:,i]]),tf.transpose([treasury_rates[:,i]])],axis=1))
        pct=pct*tf.concat([tf.math.pow(tf.nn.relu(x),p),tf.ones([m,2],dtype='float32')],axis=1)
        pct=pct/(tf.transpose([tf.math.reduce_sum(pct,axis=1)])+0.00001)
        temp_income=temp_income*tf.cast( (1+inflation_rates[:,i]) , dtype='float32')
        payment=pct*temp_income[:,tf.newaxis]
        x=x*tf.transpose([tf.ones([m],dtype='float32')+credit_card_interest_apr,\
            tf.ones([m],dtype='float32')+student_loan_interest_apr,\
                tf.ones([m],dtype='float32')+treasury_rates[:,i],\
                    tf.ones([m],dtype='float32'),\
                        tf.ones([m],dtype='float32')+rate_of_returns[:,i]]) #update interest rate
        #for retirement savings, update rate of return with r_new=1-(1-r)*(1+rate)=-rate+r*(1+rate)
        #x=x-[0,0,0,0,rate_of_returns[i]]
        x=x-tf.transpose([tf.zeros([m],dtype='float32'),\
            tf.zeros([m],dtype='float32'),\
                treasury_rates[:,i],\
                    tf.zeros([m],dtype='float32'),\
                        rate_of_returns[:,i]]) #update interest rate
        match=match_pct(pct[:,5]) #match percent
        match=match*temp_income #match amount
        
        x=x-tf.transpose(\
            [payment[:,0]/credit_card_balance,\
            payment[:,1]/student_loan_balance,\
                payment[:,2]/mortgage_down,\
                    payment[:,3]/(emergency_amount_1+emergency_amount_2),\
                        (payment[:,4]+payment[:,5]+payment[:,6]+match)/retirement_savings_needed]\
                            )
        x=tf.nn.relu(x)
        loss=loss+u_1(cc_priority,x[:,0])\
            +u_1(sl_priority,x[:,1])\
                +u_1(mortgage_priority,x[:,2])\
                    +u_2(emergency_priority_1,emergency_priority_2,emergency_amount_2/(emergency_amount_1+emergency_amount_2),x[:,3])\
                        +u_1(retirement_priority,x[:,4])\
                            +u_2(t401k_min_priority,t401k_max_priority,(t401k_max_pct-t401k_min_pct)/t401k_max_pct,1.-pct[:,5]/t401k_max_pct)\
                                +u_1(ira_priority,1.-(payment[:,6]/ira_max_cont) )
    grads=tf.gradients(-1*tf.math.reduce_sum(loss),model.trainable_weights)
    optimizer.apply_gradients(zip(grads,model.trainable_weights))
    return tf.math.reduce_sum(loss)

@tf.function(jit_compile=False)
def train_loop(model,optimizer,x0,inflation_rates,rate_of_returns,treasury_rates):
    for i in tf.range(epochs):
        inf_dataset = tf.data.Dataset.from_tensor_slices(inflation_rates)
        returns_dataset = tf.data.Dataset.from_tensor_slices(rate_of_returns)
        treasury_dataset= tf.data.Dataset.from_tensor_slices(treasury_rates)
        
        inf_dataset = inf_dataset.shuffle(len(inflation_rates))
        inf_dataset = inf_dataset.batch(batch_size)
        
        returns_dataset = returns_dataset.shuffle(len(rate_of_returns))
        returns_dataset = returns_dataset.batch(batch_size)
        
        treasury_dataset = treasury_dataset.shuffle(len(treasury_rates))
        treasury_dataset = treasury_dataset.batch(batch_size)        
        for inf,ror,tre in zip(inf_dataset,returns_dataset,treasury_dataset):
            loss=train(model,x0,optimizer,inf,ror,tre)
            tf.print('Loss:', loss/tf.cast(len(inf),dtype='float32'), 'Iterations:', i) #metrics, comment out to jit for faster execution

def eval(model,x0,inflation_rates,rate_of_returns,treasury_rates):
    x = x0
    loss=0.0
    results=-1*tf.ones((months,n),dtype=tf.dtypes.float32)
    payments=-1*tf.ones((months,n+m),dtype=tf.dtypes.float32)
    extra_funds=tf.zeros((months,1),dtype=tf.dtypes.float32)
    incomes=-1*tf.ones((months,1),dtype=tf.dtypes.float32)
    temp_income=monthly_income
    temp=0.0
    
    for i in tf.range(months):
        pct=model(tf.concat([x,[[inflation_rates[i],rate_of_returns[i],treasury_rates[i]]]  ],axis=1))
        pct=pct*tf.concat([tf.math.pow(tf.nn.relu(x),p),tf.ones([1,2],dtype='float32')],axis=1)
        pct=pct/(tf.transpose([tf.math.reduce_sum(pct,axis=1)])+0.00001)
        temp_income=temp_income*tf.cast( (1+inflation_rates[i]) , dtype='float32')
        payment=(temp_income+temp)*pct
        payment=payment[0]
        x=x*(tf.ones((1,n))+[credit_card_interest_apr,student_loan_interest_apr,treasury_rates[i],0,rate_of_returns[i]]) #update interest rate
        #for retirement savings, update rate of return with r_new=1-(1-r)*(1+rate)=-rate+r*(1+rate)
        x=x-[0,0,treasury_rates[i],0,rate_of_returns[i]]
        match=match_pct(pct[0][5]) #match percent
        match=match*temp_income #match amount
        
        x=x-[payment[0]/credit_card_balance,\
            payment[1]/student_loan_balance,\
                payment[2]/mortgage_down,\
                    payment[3]/(emergency_amount_1+emergency_amount_2),\
                        (payment[4]+payment[5]+payment[6]+match)/retirement_savings_needed]
        temp=[credit_card_balance,student_loan_balance,mortgage_down,(emergency_amount_1+emergency_amount_2),retirement_savings_needed]*x
        temp=-1*tf.math.reduce_sum(temp[temp<0])    
        x=tf.nn.relu(x)
        results=tf.concat([results[:i,:],x,results[(i+1):,:]],0)
        payments=tf.concat([payments[:i,:],[payment],payments[(i+1):,:]],0)
        extra_funds=tf.concat([extra_funds[:i,:],[[temp]],extra_funds[(i+1):,:]],0)
        incomes=tf.concat([incomes[:i,:],[[temp_income]],incomes[(i+1):,:]],0)
    
    fig, axs = plt.subplots(4, 3)
    axs[0,0].plot(tf.range(months),credit_card_balance*results[:,0] )
    axs[0,0].set_title('Credit Card')
    axs[0,1].plot(tf.range(months),student_loan_balance*results[:,1])
    axs[0,1].set_title('Student Loan')
    axs[1,0].plot(tf.range(months),mortgage_down*results[:,2])
    axs[1,0].set_title('Mortgage Down')
    axs[1,1].plot(tf.range(months),(emergency_amount_1+emergency_amount_2)*results[:,3])
    axs[1,1].set_title('Emergency Fund')
    axs[0,2].plot(tf.range(months),retirement_savings_needed*results[:,4])
    axs[0,2].set_title('Retirement')
    axs[2,0].plot(tf.range(months),tf.math.minimum(payments[:,5],tf.transpose(incomes*t401k_max_pct)[0] ) )
    axs[2,0].set_title('401K')
    axs[2,1].plot(tf.range(months),tf.math.minimum( payments[:,6],ira_max_cont ) )
    axs[2,1].set_title('IRA')
    axs[1,2].plot(tf.range(months),extra_funds)
    axs[1,2].set_title('Extra Funds')
    axs[2,2].plot(tf.range(months),incomes)
    axs[2,2].set_title('Monthly Income')
    
    axs[3,0].plot(tf.range(months),inflation_rates )
    axs[3,0].set_title('Inflation')
    axs[3,1].plot(tf.range(months),rate_of_returns )
    axs[3,1].set_title('NASDAQ Returns')
    axs[3,2].plot(tf.range(months),treasury_rates)
    axs[3,2].set_title('3 Month Treasury Returns')
    plt.show()    


inflation_rates,rate_of_returns,treasury_rates=load_data()

Full_one = keras.Sequential(
    [
        layers.Dense(10, activation="elu", name="layer1"),
        layers.Dense(10, activation="elu", name="layer2"),
        layers.Dense(n+m, activation="softmax", name="layer3"),
    ]
)
optimizer1 = keras.optimizers.Adam(learning_rate=1e-3)
x0=tf.ones((1,n),dtype=tf.dtypes.float32)
Full_one(tf.concat([x0,[[1,1,1]]],axis=1))
train_loop(Full_one,optimizer1,x0,inflation_rates,rate_of_returns,treasury_rates)

elapsed = time.perf_counter() - start
tf.print('Elapsed %.3f seconds.' % elapsed)

eval_index=200
eval(Full_one,x0,inflation_rates[eval_index,:],rate_of_returns[eval_index,:],treasury_rates[eval_index,:])

#Full_one.save_weights("mymodel")
