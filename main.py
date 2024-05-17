import random
import matplotlib.pyplot as plt
import numpy as np



INPUT_LAYER = 4
OUT_LAYER = 3
HIDDEN_LAYER = 10
LEARNING_RATE = 0.00037
EPOCH = 707
BATCH_SIZE = 50

# attributes_input_data = np.random.randn(1, INPUT_LAYER)
# real_answer = random.randint(0,OUT_LAYER-1)
#Нормльное распеределение
weight_input_hidden = np.random.rand(INPUT_LAYER, HIDDEN_LAYER)
weight_hidden_out = np.random.rand(HIDDEN_LAYER, OUT_LAYER)
bias_hidden = np.random.rand(1, HIDDEN_LAYER)
bias_out = np.random.rand(1, OUT_LAYER)

#Равномерное распределение
weight_input_hidden = (weight_input_hidden - 0.5) * 2 * np.sqrt(1/INPUT_LAYER)
weight_hidden_out = (weight_hidden_out - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)
bias_hidden = (bias_hidden - 0.5) * 2 * np.sqrt(1/INPUT_LAYER)
bias_out = (bias_out - 0.5) * 2 * np.sqrt(1/HIDDEN_LAYER)


def relu(t):
    return np.maximum(t,0)

def softmax(t):
    t_max = np.max(t)#Чтобы не переполнялась экспонента
    out = np.exp(t-t_max)
    return out/np.sum(out)
def softmax_batch(t):
    t_max = np.max(t)
    out = np.exp(t-t_max)
    return out/np.sum(out, axis=1, keepdims=True)

def sparse_cross_entropy(z,y):
    return -np.log(z[0,y]+ 1e-9)#Для nan значений

def sparse_cross_entropy_batch(z,y):
    return -np.log(np.array([z[j,y[j]]+1e-9 for j in range(len(y))]))

def to_full(y, num_classes):
    y_full = np.zeros((1,num_classes))
    y_full[0,y] = 1
    return y_full

def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full
def relu_derivative(t):
    return (t>=0).astype(float)

from sklearn import datasets
iris = datasets.load_iris()
m = np.max(iris.data)
print(m)
dataset = [(iris.data[i][None,...]/m, iris.target[i]) for i in range(len(iris.target))]
print(dataset)

loss = []

for era in range(EPOCH):
    random.shuffle(dataset) #Чтобы на каждой эпохе перемешивался датасет
    for i in range(len(dataset)//BATCH_SIZE ): #Остаток датасета от деления на целое отбрасывается

        batch_attributes, batch_answers = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
        attributes_input_data = np.concatenate(batch_attributes, axis=0)
        real_answer = np.array(batch_answers)

        #Прямое распространение
        t1 = attributes_input_data @ weight_input_hidden + bias_hidden
        h1 = relu(t1)
        t2 = h1 @ weight_hidden_out + bias_out
        z = softmax_batch(t2) #Наш вектор из вероятностей, предсказанные моделью
        exception = np.sum(sparse_cross_entropy_batch(z,real_answer))

        #ОБратное распространение
        y_full = to_full_batch(real_answer, OUT_LAYER) #Полный вектор правильного ответа(Теперь матрица или список из векторов с правильными ответами)
            #Градиенты
        dE_dt2 = z - y_full
        dE_dWeight_hidden_out = h1.T @ dE_dt2
        dE_dBias_out = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ weight_hidden_out.T
        dE_dt1 = dE_dh1 * relu_derivative(t1)
        dE_dWeight_input_hidden = attributes_input_data.T @ dE_dt1
        dE_dBias_hidden = np.sum(dE_dt1, axis=0, keepdims=True)

        #Обновление данных/обучение
        weight_input_hidden = weight_input_hidden - LEARNING_RATE * dE_dWeight_input_hidden
        weight_hidden_out = weight_hidden_out - LEARNING_RATE * dE_dWeight_hidden_out
        bias_hidden = bias_hidden - LEARNING_RATE * dE_dBias_hidden
        bias_out = bias_out - LEARNING_RATE * dE_dBias_out

        loss.append(exception)#Записываем ошибку для будущего графика
    print(era+1, "Эпоха с ошибкой ",loss[era])

def predict(attributes_input_data):
    t1 = attributes_input_data @ weight_input_hidden + bias_hidden
    h1 = relu(t1)
    t2 = h1 @ weight_hidden_out + bias_out
    z = softmax(t2)  # Наш вектор из вероятностей, предсказанные моделью
    return z

def find_accurancy():
    correct_answer = 0
    for x,y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct_answer += 1
    acc = correct_answer/len(dataset)
    return acc

accurancy = find_accurancy()
print("Accurancy: ", accurancy)
plt.plot(loss)
plt.show()