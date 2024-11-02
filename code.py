!pip install pykan

test_label

from kan import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

dataset = {}


dtype = torch.get_default_dtype()

train_input = pd.read_csv('train_input - Sheet1.csv')

test_input = pd.read_csv('test_input - Sheet1.csv')

train_label = pd.read_csv('train_label - Sheet1.csv')

test_label = pd.read_csv('test_label - Sheet1.csv')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_input_scaled = scaler.fit_transform(train_input)
test_input_scaled = scaler.transform(test_input)


train_input = pd.DataFrame(train_input_scaled, columns=train_input.columns)
test_input = pd.DataFrame(test_input_scaled, columns=test_input.columns)


def create_bins(data, column_name):
    min_val = data[column_name].min()
    max_val = data[column_name].max()
    if min_val == max_val:
        # All values are the same, create a single bin
        data[column_name + '_bin'] = 0
        print("All the same")
    else:
        bins = np.linspace(min_val, max_val, 3)
        data[column_name + '_bin'] = pd.cut(data[column_name], bins=bins, labels=False, include_lowest=True)
    return data

train_label = create_bins(train_label, 'UHI_winter_day')
test_label = create_bins(test_label, 'UHI_winter_day')
train_label = train_label.drop('UHI_winter_day', axis=1)
test_label = test_label.drop('UHI_winter_day', axis=1)

from kan import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

dataset = {}
dtype = torch.get_default_dtype()

numpy_array1 = train_input.values
traininput = torch.tensor(numpy_array1, dtype=torch.float32)
dataset['train_input'] = torch.from_numpy(numpy_array1).type(dtype)

numpy_array2 = test_input.values
testinput = torch.tensor(numpy_array2, dtype=torch.float32)
dataset['test_input'] = torch.from_numpy(numpy_array2).type(dtype)

numpy_array3 = train_label.values
trainlabel = torch.tensor(numpy_array3, dtype=torch.float32)
dataset['train_label'] = torch.from_numpy(numpy_array3[:,None]).type(dtype)

numpy_array4 = test_label.values
testlabel = torch.tensor(numpy_array4, dtype=torch.float32)
dataset['test_label'] = torch.from_numpy(numpy_array4[:,None]).type(dtype)

X = dataset['train_input']
y = dataset['train_label']

plt.scatter(X[:,0], X[:,1], c=y[:,0])

model = KAN(width=[4,1], grid=3, k=3)

def train_acc():
    return torch.mean((torch.round(model(dataset['train_input'])[:,0]) == dataset['train_label'][:,0]).type(dtype))

def test_acc():
    return torch.mean((torch.round(model(dataset['test_input'])[:,0]) == dataset['test_label'][:,0]).type(dtype))

results = model.fit(dataset, opt="LBFGS", steps=5, metrics=(train_acc, test_acc));
results['train_acc'][-1], results['test_acc'][-1]#normalize the data
lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','tan','abs']
model.auto_symbolic(lib=lib)
formula = model.symbolic_formula()[0][0]
formula
# how accurate is this formula?
def acc(formula, X, y):
    batch = X.shape[0]
    correct = 0
    for i in range(batch):
        correct += np.round(np.array(formula.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)) == y[i,0]
    return correct/batch

print('train acc of the formula:', acc(formula, dataset['train_input'], dataset['train_label']))
print('test acc of the formula:', acc(formula, dataset['test_input'], dataset['test_label']))
model.plot(beta=5)
