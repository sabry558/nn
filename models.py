import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class models:
   
   def __init__(self,learning_rate,mse_threshold,epochs,bias):
     self.data_frame=None
     self.weights=None
     self.L_r=learning_rate
     self.mse=mse_threshold
     self.epochs=epochs
     self.bias=bias
      
   def signum(self, x):
       return 1 if x >= 0 else -1

   def read_csv(self,str):
      self.data_frame=pd.read_csv(str)

   def preprocess_data(self,f1,f2,c1,c2):
      
      gender_distribution = self.data_frame.groupby('bird category')['gender'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
      self.data_frame['gender'] = self.data_frame.apply(lambda row: gender_distribution[row['bird category']] if pd.isnull(row['gender']) else row['gender'], axis=1)




      label_encoder=preprocessing.LabelEncoder()
      self.data_frame.iloc[:,0]=label_encoder.fit_transform(self.data_frame.iloc[:,0])
      normalizer=preprocessing.StandardScaler()
      
      self.data_frame.iloc[:, 1:-1] = normalizer.fit_transform(self.data_frame.iloc[:, 1:-1])

      data =self.data_frame[(self.data_frame['bird category'] == c1) | (self.data_frame['bird category'] == c2)]

      data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])

      
      data=data.iloc[:,[f1,f2,-1]]

      

      X=data.iloc[:,:-1].values
      Y=data.iloc[:,-1].values
      Y = np.where(Y == 0, -1, Y)

   

      X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,stratify=Y,shuffle=True)
      return X_train,X_test,Y_train,Y_test

   def preceptron_model(self,f1,f2,c1,c2):
       X_train,X_test,Y_train,Y_test=self.preprocess_data(f1,f2,c1,c2)
       if(self.bias==True):
        self.weights=np.random.rand(3)
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
           
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
       else:
          self.weights=np.random.rand(2)
          Y_train = np.array(Y_train)  
       X_train = np.array(X_train, dtype=float)
       for epoch in range(self.epochs):
            for i in range(len(X_train)):
                input = np.dot(self.weights, X_train[i])         
                output = self.signum(input)
                error = Y_train[i] - output
                self.weights += self.L_r * error * X_train[i]

       

       predictions = [self.predict_perceptron(x) for x in X_test]
       print(self.accuracy(predictions,Y_test))

       return predictions,X_test,Y_test ,self.accuracy(predictions,Y_test)
   


   def adaline_model(self,f1,f2,c1,c2):
      X_train,X_test,Y_train,Y_test=self.preprocess_data(f1,f2,c1,c2)
      if(self.bias==True):
        self.weights=np.random.rand(3)
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
           
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
      else:
        self.weights=np.random.rand(2)
        Y_train = np.array(Y_train) 

      X_train = np.array(X_train, dtype=float)
  
      for epoch in range(self.epochs):
            for i in range(len(X_train)):
                input = np.dot(self.weights, X_train[i])         
                error = Y_train[i] - input
                self.weights += self.L_r * error * X_train[i]

            mse_temp=np.mean((Y_train-np.dot(X_train,self.weights))**2)
            if(mse_temp<=self.mse):
                break            

      predictions = [self.predict_adaline(x) for x in X_test]

      print(self.accuracy(predictions,Y_test))
      return predictions,X_test,Y_test , self.accuracy(predictions,Y_test)
       





   def predict_perceptron(self, X):
       
        return self.signum(np.dot(X,self.weights))

   
   def predict_adaline(self,X):   

    output=np.dot(X,self.weights)
    return 1 if output >= 0 else -1
   

   def accuracy(self,pred,real):
      counter=0
      for i in range (len(pred)):
         if(pred[i]==real[i]):
            counter=counter+1
      return counter/len(pred)  


   def confusion_mat(self,pred,real):
      tp=tn=fp=fn=0
      for i in range(len(pred)):
         if(pred[i]==real[i] and pred[i]==1):
            tp+=1
         elif(pred[i]==real[i] and pred[i]==-1):
            tn+=1
         elif(pred[i]==-1 and real[i]==1):
            fn+=1
         else:
            fp+=1             

      return [[tn,fp],[fn,tp]]       

               
             
                


 
          
        

     