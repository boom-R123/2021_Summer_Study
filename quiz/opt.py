import numpy as np
import matplotlib.pyplot as plt
import math
x = np.array([0, 0.5, 0.8, 1.1, 1.5, 1.9, 2.2, 2.4, 2.6, 3.0])
y = np.array([0.9, 2.1, 2.7, 3.1, 4.1, 4.8, 5.1, 5.9, 6.0, 7.0])
def SGD(x, y ,epoch=10, lr=0.01):
    print('---SGD---')
    w = 0
    b = 0
    for i in range(epoch):
        cost = 0
        for j in range(x.shape[0]):
            y_p = w * x[j] + b
            dw = x[j] * (y_p - y[j])
            db = y_p -y[j]
            w = w - lr * dw
            b = b - lr * db
            cost += (y_p - y[j]) ** 2
        cost /= (2 * 10)
        print('epoch:' + str(i+1) + '\tcost:' + str(cost))
    y_p = x * w + b
    plt.figure()
    plt.title('SGD')
    plt.plot(x, y ,color = 'r', label='True')
    plt.plot(x, y_p, color = 'b', label='predict')
    plt.legend()
    plt.show()
def Momentum(x, y ,epoch=10, lr=0.01, bt=0.9):
    print('---Momentum---')
    w = 0
    b = 0
    v_dw = 0
    v_db = 0
    for i in range(epoch):
        cost = 0
        for j in range(x.shape[0]):
            y_p = w * x[j] + b
            dw = x[j] * (y_p - y[j])
            db = y_p -y[j]
            v_dw = bt * v_dw + (1- bt) * dw
            v_db = bt * v_db + (1 -bt) * db
            w = w - lr * v_dw
            b = b - lr * v_db
            cost += (y_p - y[j]) ** 2
        cost /= (2 * 10)
        print('epoch:' + str(i+1) + '\tcost:' + str(cost))
    y_p = x * w + b
    plt.figure()
    plt.title('Momentum')
    plt.plot(x, y ,color = 'r', label='True')
    plt.plot(x, y_p, color = 'b', label='predict')
    plt.legend()
    plt.show()
def RMSprop(x, y ,epoch=10, lr=0.01, bt=0.9, ep=1e-8):
    print('---RMSprop---')
    w = 0
    b = 0
    S_dw = 0
    S_db = 0
    for i in range(epoch):
        cost = 0
        for j in range(x.shape[0]):
            y_p = w * x[j] + b
            dw = x[j] * (y_p - y[j])
            db = y_p -y[j]
            S_dw = bt * S_dw + (1 - bt) * dw ** 2
            S_db = bt * S_db + (1 - bt) * db ** 2
            w = w - lr * dw / math.sqrt(S_dw + ep)
            b = b - lr * db / math.sqrt(S_db + ep)
            cost += (y_p - y[j]) ** 2
        cost /= (2 * 10)
        print('epoch:' + str(i+1) + '\tcost:' + str(cost))
    y_p = x * w + b
    plt.figure()
    plt.title('RMSprop')
    plt.plot(x, y ,color = 'r', label='True')
    plt.plot(x, y_p, color = 'b', label='predict')
    plt.legend()
    plt.show()
def Adam(x, y ,epoch=10, lr=0.01, b1=0.9, b2=0.999, ep=1e-8):
    print('---Adam---')
    w = 0
    b = 0
    v_dw = 0
    v_db = 0
    S_dw = 0
    S_db = 0
    for i in range(epoch):
        cost = 0
        for j in range(x.shape[0]):
            y_p = w * x[j] + b
            dw = x[j] * (y_p - y[j])
            db = y_p -y[j]
            v_dw = b1 * v_dw + (1 - b1) * dw
            v_db = b1 * v_db + (1 - b1) * db
            S_dw = b2 * S_dw + (1 - b2) * dw ** 2
            S_db = b2 * S_db + (1 - b2) * db ** 2
            v_dw_c = v_dw / (1 - b1 ** (i * 10 + j + 1))
            v_db_c = v_db / (1 - b1 ** (i * 10 + j + 1))
            S_dw_c = S_dw / (1 - b2 ** (i * 10 + j + 1))
            S_db_c = S_db / (1 - b2 ** (i * 10 + j + 1))
            w = w - lr * v_dw_c / math.sqrt(S_dw_c + ep)
            b = b - lr * v_db_c / math.sqrt(S_db_c + ep)
            cost += (y_p - y[j]) ** 2
        cost /= (2 * 10)
        print('epoch:' + str(i+1) + '\tcost:' + str(cost))
    y_p = x * w + b
    plt.figure()
    plt.title('Adam')
    plt.plot(x, y ,color = 'r', label='True')
    plt.plot(x, y_p, color = 'b', label='predict')
    plt.legend()
    plt.show()
SGD(x,y,epoch=10)
Momentum(x,y,epoch=7)
RMSprop(x,y,epoch=50)
Adam(x,y,lr=0.01,epoch=140)

