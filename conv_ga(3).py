import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import pygad

device = torch.device("cpu")

# Step 0: get data
file_path = "/Users/katerina/Documents/GAoptimizedCNN/KS11-3.csv"  
df = pd.read_csv(file_path)
df['volume'] = df['volume']#.astype(np.float32)


# Step 1: Prepare the feats
N = 7

# momentum
df['momentum'] = df['close'] - df['close'].shift(N)

# stochastic K
df['stochastic_K'] = 100 * (df['close'] - df['low'].rolling(N).min()) / (df['high'].rolling(N).max() - df['low'].rolling(N).min())

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(N).mean()
avg_loss = loss.rolling(N).mean()
RS = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + RS))

#macd
ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26

# LW %R
df['LW'] = (df['high'] - df['close']) / (df['high'] - df['low']) * 100

# A/D Oscillator
A_D = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
df['A/D Oscillator'] = (df['high'] - df['close'].shift(1)) / (df['high'] - df['low']) * 100

# CCI
TP = (df['high'] + df['low'] + df['close']) / 3
SMA = TP.rolling(N).mean()
MAD = (TP - SMA).abs().rolling(N).mean()
df['CCI'] = (TP - SMA) / (0.015 * MAD)

# label
next_close = df['close'].shift(-1)
df['label'] = (next_close > df['close']).astype(int)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Step 2: Normalize the Data
for col in ['momentum', 'stochastic_K', 'RSI', 'MACD', 'LW', 'A/D Oscillator', 'CCI']:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Step 3: Split the Data
train_size = int(0.8 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

train_label = train_data['label']
test_label = test_data['label']
train_data = train_data[['momentum', 'stochastic_K', 'RSI', 'MACD', 'LW', 'A/D Oscillator', 'CCI']]
test_data = test_data[['momentum', 'stochastic_K', 'RSI', 'MACD', 'LW', 'A/D Oscillator', 'CCI']]

# Step 4: Make Trainable Dataset
# hyper params of training
TIME_WINDOW_SIZE = 64
num_epochs = 256
batch_size = 8
lr = 0.005


train_sequences = []
train_labels = []
for i in range(len(train_data) - TIME_WINDOW_SIZE+1):
    train_sequences.append(train_data.iloc[i:i+TIME_WINDOW_SIZE].values)
    train_labels.append(train_label.iloc[i+TIME_WINDOW_SIZE-1])

test_sequences = []
test_labels = []
for i in range(len(test_data) - TIME_WINDOW_SIZE+1):
    test_sequences.append(test_data.iloc[i:i+TIME_WINDOW_SIZE].values)
    test_labels.append(test_label.iloc[i+TIME_WINDOW_SIZE-1])


train_sequences_tensor = torch.Tensor(train_sequences).to(device)
train_labels_tensor = torch.Tensor(train_labels).to(device)
test_sequences_tensor = torch.Tensor(test_sequences).to(device)
test_labels_tensor = torch.Tensor(test_labels).to(device)


# Step 5: Define the TimeSeriesCNN
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_length: int, num_kernel_conv1: int, num_kernel_conv2: int, kernel_size_conv1: list, kernel_size_conv2 : list, kernel_size_pooling: int, fc_size = 64):
        super(TimeSeriesCNN, self).__init__()
        self.input_length = input_length
        self.num_kernel_conv1 = num_kernel_conv1
        self.num_kernel_conv2 = num_kernel_conv2
        self.kernel_size_conv1 = kernel_size_conv1
        self.kernel_size_conv2 = kernel_size_conv2
        self.kernel_size_pooling = kernel_size_pooling
        
        self.conv1_layers = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_kernel_conv1, kernel_size=kernel_size_conv1[0]),#4),
            nn.Conv1d(in_channels=1, out_channels=num_kernel_conv1, kernel_size=kernel_size_conv1[1]),#22),
            nn.Conv1d(in_channels=1, out_channels=num_kernel_conv1, kernel_size=kernel_size_conv1[2]),#27),
            nn.Conv1d(in_channels=1, out_channels=num_kernel_conv1, kernel_size=kernel_size_conv1[3]),#6),
            nn.Conv1d(in_channels=1, out_channels=num_kernel_conv1, kernel_size=kernel_size_conv1[4]),#17),
            nn.Conv1d(in_channels=1, out_channels=num_kernel_conv1, kernel_size=kernel_size_conv1[5]),#10),
            nn.Conv1d(in_channels=1, out_channels=num_kernel_conv1, kernel_size=kernel_size_conv1[6]),#23)
        ])
        
        self.conv2_layers = nn.ModuleList([
            nn.Conv1d(in_channels=num_kernel_conv1, out_channels=num_kernel_conv2, kernel_size=kernel_size_conv2[0]),#3),
            nn.Conv1d(in_channels=num_kernel_conv1, out_channels=num_kernel_conv2, kernel_size=kernel_size_conv2[1]),#2),
            nn.Conv1d(in_channels=num_kernel_conv1, out_channels=num_kernel_conv2, kernel_size=kernel_size_conv2[2]),#12),
            nn.Conv1d(in_channels=num_kernel_conv1, out_channels=num_kernel_conv2, kernel_size=kernel_size_conv2[3]),#3),
            nn.Conv1d(in_channels=num_kernel_conv1, out_channels=num_kernel_conv2, kernel_size=kernel_size_conv2[4]),#6),
            nn.Conv1d(in_channels=num_kernel_conv1, out_channels=num_kernel_conv2, kernel_size=kernel_size_conv2[5]),#1),
            nn.Conv1d(in_channels=num_kernel_conv1, out_channels=num_kernel_conv2, kernel_size=kernel_size_conv2[6]),#4)
        ])
        
        self.pool = nn.MaxPool1d(kernel_size=kernel_size_pooling)

        # Adjust output_length based on convolution and pooling operations
        num_channel = len(kernel_size_conv1)
        output_length = 0
        for i in range(num_channel):
            output_length += int(((input_length - kernel_size_conv1[i] + 1 - kernel_size_conv2[i] + 1)-kernel_size_pooling)/kernel_size_pooling) + 1
        
        self.fc1 = nn.Linear(in_features=output_length*num_kernel_conv2, out_features=fc_size)
        self.fc2 = nn.Linear(in_features=fc_size, out_features=1)

    def forward(self, x):
        # conv layer 1
        x_convs1 = [nn.functional.elu(conv(x[:,:,i].unsqueeze(1))) for i, conv in enumerate(self.conv1_layers)]

        # conv layer 2
        x_convs2 = [nn.functional.elu(self.conv2_layers[i](x_convs1[i])) for i in range(len(self.conv2_layers))]

        # pooling
        x_pools = [self.pool(x_convs2[i]) for i in range(len(x_convs2))]

        # flatten
        x_flattened = [x_pools[i].view(x_pools[i].size(0), -1) for i in range(len(x_pools))]
        x_concat = torch.cat(x_flattened, dim=1)

        # full connect 1
        x_fc1 = nn.functional.elu(self.fc1(x_concat))

        # full connect 2 and output
        x_final = torch.sigmoid(self.fc2(x_fc1)).squeeze()
        return x_final
    

# Step 6: Training and Evaluation

def train_and_eval(num_kernel_conv1, num_kernel_conv2, kernel_size_conv1, kernel_size_conv2, kernel_size_pooling, print_accuracy = False, return_history = False):

    # Instantiate the TimeSeriesCNN model
    model = TimeSeriesCNN(input_length = TIME_WINDOW_SIZE, num_kernel_conv1=num_kernel_conv1, num_kernel_conv2=num_kernel_conv2, kernel_size_conv1=kernel_size_conv1, kernel_size_conv2=kernel_size_conv2, kernel_size_pooling=kernel_size_pooling).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_value = []
    accuracy_in_test = []
    accuracy_in_train = []

    
    try: # if params is valid, train and eval
        # Training loop
        for epoch in range(num_epochs):
            avg_loss_per_epoch = 0.0
            num_batch = 0
            correct_train = 0
            total_train = 0
            for i in range(0, train_sequences_tensor.shape[0],batch_size):
                batch = train_sequences_tensor[i:i+batch_size,:,:]
                labels = train_labels_tensor[i:i+batch_size]

                # Forward pass
                #indicators = indicators.transpose(0,1)
                outputs = model(batch)
                predicted = (outputs.data > 0.5).to(int)
                total_train += batch_size
                correct_train += (predicted == labels).sum()
                # Compute loss and perform backpropagation
                loss = criterion(outputs, labels)
                avg_loss_per_+= loss.data
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batch += batch_size

            accuracy_train = 100 * correct_train / total_train
            accuracy_in_train.append(accuracy_train)
            loss_value.append(avg_loss_per_epoch/num_batch)

            # Evaluate the model on the test dataset
            with torch.no_grad():
                correct_test = 0
                total_test = 0
                for i in range(0, test_sequences_tensor.shape[0],batch_size):
                    batch = test_sequences_tensor[i:i+batch_size,:,:]
                    labels = test_labels_tensor[i:i+batch_size]

                    outputs = model(batch)
                    predicted = (outputs.data > 0.5).to(int)
                    total_test += batch_size
                    correct_test += (predicted == labels).sum()
                accuracy_test = 100 * correct_test / total_test
                accuracy_in_test.append(accuracy_test)
                if(print_accuracy):
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {accuracy_train:.2f}%, Test Accuracy: {accuracy_test:.2f}%')
        
        fitness = float(accuracy_test)
        if return_history:
            return model, fitness, loss_value, accuracy_in_train, accuracy_in_test

        return fitness
    
    except: # input is not valid, return 0 as fitness
        fitness = 0.0
        return fitness

# step 7 define fitness func
def fitness_func(ga, solution, solution_idx):
    num_kernel_conv1 = solution[0]
    num_kernel_conv2 = solution[1]
    kernel_size_conv1 = solution[2:9]
    kernel_size_conv2 = solution[10:17]
    kernel_size_pooling = solution[18]
    
    fitness = train_and_eval(num_kernel_conv1= num_kernel_conv1, 
                             num_kernel_conv2= num_kernel_conv2,
                             kernel_size_conv1= kernel_size_conv1,
                             kernel_size_conv2= kernel_size_conv2,
                             kernel_size_pooling= kernel_size_pooling,
                             print_accuracy= False,
                             return_history= False
                            )
    return fitness

num_generations=10
sol_per_pop = 5
num_parents_mating=2
num_genes = 19
init_range_low = 1
init_range_high = 32

ga_instance = pygad.GA( num_generations=num_generations,
                        num_parents_mating=num_parents_mating, 
                        fitness_func=fitness_func,
                        sol_per_pop=sol_per_pop, 
                        num_genes=num_genes,
                        gene_type=int,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        mutation_probability=0.25,
                        crossover_probability=0.7)

ga_instance.run()
ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
num_kernel_conv1 = solution[0]
num_kernel_conv2 = solution[1]
kernel_size_conv1 = solution[2:9]
kernel_size_conv2 = solution[10:17]
kernel_size_pooling = solution[18]

print("Parameters of the best solution:")
print("num_kernel_conv1:",num_kernel_conv1)
print("num_kernel_conv2:",num_kernel_conv2)
print("kernel_size_conv1:",kernel_size_conv1)
print("kernel_size_conv2:",kernel_size_conv2)
print("kernel_size_pooling:",kernel_size_pooling)

print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# step 9 train and output best pop



# step 8 find best pop with GA

model, fitness, loss_value, accuracy_in_train, accuracy_in_test = train_and_eval(num_kernel_conv1= num_kernel_conv1, 
                                                                                 num_kernel_conv2= num_kernel_conv2,
                                                                                 kernel_size_conv1= kernel_size_conv1,
                                                                                 kernel_size_conv2= kernel_size_conv2,
                                                                                 kernel_size_pooling= kernel_size_pooling,
                                                                                 print_accuracy=True,
                                                                                 return_history= True
                                                                                 )


plt.figure()
plt.subplot(1,2,1)
plt.plot(loss_value)
plt.title("BCE Loss")
plt.xlabel("epoch")
plt.subplot(1,2,2)
plt.plot(accuracy_in_train)
plt.plot(accuracy_in_test)
plt.title(f"accuracy train:{accuracy_in_train[-1]:.2f}%, test:{accuracy_in_test[-1]:.2f}%")
plt.xlabel("epoch")
plt.legend(['accuracy_in_train','accuracy_in_test'])
#plt.legend(['train','test'])
plt.show()
torch.save(model, "model.pth")
pass