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
df['volume'] = df['volume']

# 删除包含NaN值的行
df.dropna(inplace=True)

# Step 1: Prepare the features
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

# macd
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
TIME_WINDOW_SIZE = 64
num_epochs = 35
batch_size = 8
lr = 0.0008  # 下降到0.001后，会显著过拟合
weight_decay = 1e-1  # L2正则化系数，1e-3 1e-4 3.16e-4

train_sequences = []
train_labels = []
for i in range(len(train_data) - TIME_WINDOW_SIZE + 1):
    train_sequences.append(train_data.iloc[i:i + TIME_WINDOW_SIZE].values)
    train_labels.append(train_label.iloc[i + TIME_WINDOW_SIZE - 1])

test_sequences = []
test_labels = []
for i in range(len(test_data) - TIME_WINDOW_SIZE + 1):
    test_sequences.append(test_data.iloc[i:i + TIME_WINDOW_SIZE].values)
    test_labels.append(test_label.iloc[i + TIME_WINDOW_SIZE - 1])

train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)

train_sequences_tensor = torch.Tensor(train_sequences).to(device)
train_labels_tensor = torch.Tensor(train_labels).to(device)
test_sequences_tensor = torch.Tensor(test_sequences).to(device)
test_labels_tensor = torch.Tensor(test_labels).to(device)

# 数据增强函数
def augment_data(sequences, labels, noise_level=0.01, augment_ratio=0.5):
    augmented_sequences = []
    augmented_labels = []
    num_augmented = int(len(sequences) * augment_ratio)
    for _ in range(num_augmented):
        idx = np.random.randint(0, len(sequences))
        seq = sequences[idx]
        label = labels[idx]
        noise = np.random.normal(0, noise_level, seq.shape)
        augmented_seq = seq + noise
        augmented_sequences.append(augmented_seq)
        augmented_labels.append(label)
    return np.array(augmented_sequences), np.array(augmented_labels)

# Augmenting the data
augmented_train_sequences, augmented_train_labels = augment_data(train_sequences, train_labels)
train_sequences = np.concatenate((train_sequences, augmented_train_sequences), axis=0)
train_labels = np.concatenate((train_labels, augmented_train_labels), axis=0)
train_sequences_tensor = torch.Tensor(train_sequences).to(device)
train_labels_tensor = torch.Tensor(train_labels).to(device)

# Step 5: Define the TimeSeriesCNN
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_length: int, num_kernel_conv1: int, num_kernel_conv2: int, kernel_size_conv1: list, kernel_size_conv2: list, kernel_size_pooling: int, fc_size=16, dropout_rate=0.8):
        super(TimeSeriesCNN, self).__init__()
        self.input_length = input_length
        self.num_kernel_conv1 = num_kernel_conv1
        self.num_kernel_conv2 = num_kernel_conv2
        self.kernel_size_conv1 = kernel_size_conv1
        self.kernel_size_conv2 = kernel_size_conv2
        self.kernel_size_pooling = kernel_size_pooling
        self.dropout_rate = dropout_rate

        self.conv1_layers = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_kernel_conv1, kernel_size=kernel_size_conv1[i]) for i in range(7)
        ])

        self.conv2_layers = nn.ModuleList([
            nn.Conv1d(in_channels=num_kernel_conv1, out_channels=num_kernel_conv2, kernel_size=kernel_size_conv2[i]) for i in range(7)
        ])

        self.pool = nn.MaxPool1d(kernel_size=kernel_size_pooling)

        num_channel = len(kernel_size_conv1)
        output_length = 0
        for i in range(num_channel):
            output_length += int(((input_length - kernel_size_conv1[i] + 1 - kernel_size_conv2[i] + 1) - kernel_size_pooling) / kernel_size_pooling) + 1

        self.fc1 = nn.Linear(in_features=output_length * num_kernel_conv2, out_features=fc_size + 200)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=fc_size + 200, out_features=40)
        self.fc3 = nn.Linear(in_features=40, out_features=1)

    def forward(self, x):
        # conv layer 1
        x_convs1 = [nn.functional.elu(conv(x[:, :, i].unsqueeze(1))) for i, conv in enumerate(self.conv1_layers)]

        # conv layer 2
        x_convs2 = [nn.functional.elu(self.conv2_layers[i](x_convs1[i])) for i in range(len(self.conv2_layers))]

        # pooling
        x_pools = [self.pool(x_convs2[i]) for i in range(len(x_convs2))]

        # flatten
        x_flattened = [x_pools[i].view(x_pools[i].size(0), -1) for i in range(len(x_pools))]
        x_concat = torch.cat(x_flattened, dim=1)

        # full connect 1
        x_fc1 = nn.functional.elu(self.fc1(x_concat))
        x_fc1 = self.dropout(x_fc1)

        # full connect 2
        x_fc2 = nn.functional.elu(self.fc2(x_fc1))

        # full connect 3 and output
        x_final = torch.sigmoid(self.fc3(x_fc2)).squeeze()
        return x_final

# Step 6: Training and Evaluation with Early Stopping
class EarlyStopping:
    def __init__(self, patience=100, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')

def train_and_eval(num_kernel_conv1, num_kernel_conv2, kernel_size_conv1, kernel_size_conv2, kernel_size_pooling, print_accuracy=False, return_history=False):
    print("开始训练模型...")
    model = TimeSeriesCNN(input_length=TIME_WINDOW_SIZE, 
                          num_kernel_conv1=num_kernel_conv1, 
                          num_kernel_conv2=num_kernel_conv2, 
                          kernel_size_conv1=kernel_size_conv1, 
                          kernel_size_conv2=kernel_size_conv2, 
                          kernel_size_pooling=kernel_size_pooling).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=40, delta=0.001)

    loss_value = []
    accuracy_in_test = []
    accuracy_in_train = []

    try:
        print("Starting training...")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            model.train()
            avg_loss_per_epoch = 0.0
            num_batch = 0
            correct_train = 0
            total_train = 0
            for i in range(0, train_sequences_tensor.shape[0], batch_size):
                batch = train_sequences_tensor[i:i+batch_size, :, :]
                labels = train_labels_tensor[i:i+batch_size]

                if batch.size(0) == 0:  # 检查批次大小是否为零
                    continue

                # Forward pass
                outputs = model(batch)
                outputs = outputs.view(-1)  # 将输出调整为1D张量
                predicted = (outputs.data > 0.5).to(int)
                total_train += batch_size
                correct_train += (predicted == labels).sum().item()

                # Compute loss and perform backpropagation
                loss = criterion(outputs, labels.float())
                avg_loss_per_epoch += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batch += batch_size

            accuracy_train = 100 * correct_train / total_train
            accuracy_in_train.append(accuracy_train)
            loss_value.append(avg_loss_per_epoch / num_batch)

            # Evaluate the model on the test dataset
            model.eval()
            with torch.no_grad():
                correct_test = 0
                total_test = 0
                avg_val_loss = 0.0
                for i in range(0, test_sequences_tensor.shape[0], batch_size):
                    batch = test_sequences_tensor[i:i+batch_size, :, :]
                    labels = test_labels_tensor[i:i+batch_size]

                    if batch.size(0) == 0:  # 检查批次大小是否为零
                        continue

                    outputs = model(batch)
                    outputs = outputs.view(-1)  # 将输出调整为1D张量
                    predicted = (outputs.data > 0.5).to(int)
                    total_test += batch_size
                    correct_test += (predicted == labels).sum().item()
                    val_loss = criterion(outputs, labels.float())
                    avg_val_loss += val_loss.item()
                accuracy_test = 100 * correct_test / total_test
                accuracy_in_test.append(accuracy_test)
                avg_val_loss = avg_val_loss / total_test

                early_stopping(avg_val_loss, model)

                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                if print_accuracy:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {accuracy_train:.2f}%, Test Accuracy: {accuracy_test:.2f}%')

        # 绘制横轴为 epoch，纵轴为 accuracy 的图
        plt.figure()
        plt.plot(range(1, len(accuracy_in_train) + 1), accuracy_in_train, label='Train Accuracy')
        plt.plot(range(1, len(accuracy_in_test) + 1), accuracy_in_test, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.show()

        fitness = float(accuracy_test)
        if return_history:
            return model, fitness, loss_value, accuracy_in_train, accuracy_in_test

        return fitness

    except KeyboardInterrupt:
        print("Training interrupted. Saving the current model state.")
        torch.save(model, "interrupted_model.pth")
        if return_history:
            return model, float(accuracy_test), loss_value, accuracy_in_train, accuracy_in_test
        return float(accuracy_test)

    except Exception as e:
        print(f"An error occurred: {e}")
        return 0.0

# Step 7: Define Fitness Function
def fitness_func(ga_instance, solution, solution_idx):
    num_kernel_conv1 = int("".join(map(str, solution[:6])), 2) + 32
    num_kernel_conv2 = int("".join(map(str, solution[6:12])), 2) + 32
    kernel_size_conv1 = [int("".join(map(str, solution[12 + i*5:17 + i*5])), 2) + 16 for i in range(7)]
    kernel_size_conv2 = [int("".join(map(str, solution[47 + i*5:52 + i*5])), 2) + 16 for i in range(7)]
    kernel_size_pooling = int("".join(map(str, solution[82:87])), 2) + 16

    fitness = train_and_eval(num_kernel_conv1=num_kernel_conv1,
                             num_kernel_conv2=num_kernel_conv2,
                             kernel_size_conv1=kernel_size_conv1,
                             kernel_size_conv2=kernel_size_conv2,
                             kernel_size_pooling=kernel_size_pooling,
                             print_accuracy=False,
                             return_history=False)
    return fitness

# GA parameters
num_generations = 10
sol_per_pop = 100
num_parents_mating = 50
num_genes = 87  # Adjusted for binary representation
init_range_low = 0
init_range_high = 1

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_type=int,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    mutation_probability=0.25, 
    crossover_probability=0.7
)

ga_instance.run()
ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
num_kernel_conv1 = int("".join(map(str, solution[:6])), 2) + 32
num_kernel_conv2 = int("".join(map(str, solution[6:12])), 2) + 32
kernel_size_conv1 = [int("".join(map(str, solution[12 + i*5:17 + i*5])), 2) + 16 for i in range(7)]
kernel_size_conv2 = [int("".join(map(str, solution[47 + i*5:52 + i*5])), 2) + 16 for i in range(7)]
kernel_size_pooling = int("".join(map(str, solution[82:87])), 2) + 16

print("Parameters of the best solution:")
print("num_kernel_conv1:", num_kernel_conv1)
print("num_kernel_conv2:", num_kernel_conv2)
print("kernel_size_conv1:", kernel_size_conv1)
print("kernel_size_conv2:", kernel_size_conv2)
print("kernel_size_pooling:", kernel_size_pooling)
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Step 9: Train and output best pop
model, fitness, loss_value, accuracy_in_train, accuracy_in_test = train_and_eval(num_kernel_conv1=num_kernel_conv1,
                                                                                 num_kernel_conv2=num_kernel_conv2,
                                                                                 kernel_size_conv1=kernel_size_conv1,
                                                                                 kernel_size_conv2=kernel_size_conv2,
                                                                                 kernel_size_pooling=kernel_size_pooling,
                                                                                 print_accuracy=True,
                                                                                 return_history=True)

plt.figure()

# 绘制损失值（BCE Loss）
plt.subplot(1, 2, 1)
plt.plot(loss_value, label='Loss')
plt.title("BCE Loss")
plt.xlabel("epoch")
plt.legend()

# 绘制训练和测试集的准确率
plt.subplot(1, 2, 2)
plt.plot(accuracy_in_train, label='Train Accuracy')
plt.plot(accuracy_in_test, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f"accuracy train:{accuracy_in_train[-1]:.2f}%, test:{accuracy_in_test[-1]:.2f}%")
plt.legend()

plt.show()

# 保存模型
torch.save(model, "model.pth")
