import numpy as np
#read data from csv
f = open('C:/Users/ppp45/Desktop/ML/output.csv','w')
test=np.genfromtxt("C:/Users/ppp45/Desktop/ML/test.csv",delimiter=',')
test = np.delete(test, [0,1], axis=1)
test = np.nan_to_num(test)
test_data = np.array(test[0:18,0:24])
predict = np.zeros(240)
for i in range(1,240):
    test_data = np.append(test_data, test[0+18*i:18+18*i, 0:9],1)
temp = test_data[9:10,:]
test_data =np.delete(test_data, 9, axis=0)
test_data = np.append(test_data, temp,0)
test_data = test_data.T
train=np.genfromtxt("C:/Users/ppp45/Desktop/ML/train.csv",delimiter=',')
train = np.delete(train, [0,1,2], axis=1)
train = np.delete(train, (0), axis=0)
train = np.nan_to_num(train)
train_data = np.array(train[0:18,0:24])
for i in range(240):
    train_data = np.append(train_data, train[0+18*i:18+18*i, 0:24],1)
temp = train_data[9:10,:]
train_data =np.delete(train_data, (9), axis=0)
train_data = np.append(train_data, temp,0)

#linear regression
#y=b[0~17]+w[0~17]*x[0~17]
#initial parameter
weight = np.ones(18)[np.newaxis]
bias = np.zeros(9)[np.newaxis]
weight = weight.T
bias = bias.T
lr = 0.01
iteration = 10000

#using Adagrad
pre_wgrad = np.zeros(18)[np.newaxis]
pre_wgrad = pre_wgrad.T
pre_bgrad = np.zeros(9)[np.newaxis]
pre_bgrad = pre_bgrad.T

#implement gradient descent
for i in range(iteration):
    for n in range(5784-9-1):   
        train_x = train_data[0:18, 0+n:9+n]
        train_x = train_x.T
        train_y = train_data[17:18, 1+n:10+n]
        yp = np.matmul(train_x,weight)+bias
        L = yp - train_y.T 
        #record previous gradient for Adagrad 
        pre_wgrad = pre_wgrad +(2*np.dot(train_x.T,L))**2
        pre_bgrad = pre_bgrad +(2*L)**2
        adad_w = np.sqrt(pre_wgrad)
        adad_b = np.sqrt(pre_bgrad)
        a =lr*2*np.dot(train_x.T,L)
        #one example one update => more efficetive
        weight -= np.divide(a, adad_w ,out=np.zeros_like(a), where=adad_w!=0)  
        bias -= lr*2*L/adad_b
   
     

#output
print(test_data.shape)
f.write("id,value\n")
for i in range(240):
    x= test_data[8+9*i:9+9*i, :]
    predict[i] = np.dot(x,weight)+bias[8]
    f.write("id_"+str(i)+","+str(predict[i])+"\n")
f.close()
print(predict)