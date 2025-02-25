import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pygad
import os
import random
import psutil

device = torch.device("cpu")

# 定义超参数的初始值
initial_num_kernel_conv1 = 15
initial_num_kernel_conv2 = 60
initial_kernel_size_conv1 = [3, 3, 3, 3, 3, 3, 3]
initial_kernel_size_conv2 = [3, 3, 3, 3, 3, 3, 3]
initial_kernel_size_pooling = 2

# Step 0: get data
file_path = "/Users/katerina/Documents/GAoptimizedCNN/KS11-3.csv"
df = pd.read_csv(file_path)
df['volume'] = df['volume']

# 打印前5行数据
print(df.head())

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
num_epochs = 150
batch_size = 8
lr = 0.0015

weight_decay = 1e-3

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
    def __init__(self, input_length: int, num_kernel_conv1: int, num_kernel_conv2: int, kernel_size_conv1: list, kernel_size_conv2: list, kernel_size_pooling: int, fc_size=128, dropout_rate=0.4):
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

# Step 6: Training and Evaluation
def train_and_eval(num_kernel_conv1, num_kernel_conv2, kernel_size_conv1, kernel_size_conv2, kernel_size_pooling, generation, individual, print_accuracy=False, return_history=False):
    print(f"开始训练模型: Generation {generation}, Individual {individual}")
    model = TimeSeriesCNN(input_length=TIME_WINDOW_SIZE, 
                          num_kernel_conv1=num_kernel_conv1, 
                          num_kernel_conv2=num_kernel_conv2, 
                          kernel_size_conv1=kernel_size_conv1, 
                          kernel_size_conv2=kernel_size_conv2, 
                          kernel_size_pooling=kernel_size_pooling).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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

                if batch.size(0) == 0:
                    continue

                # Forward pass
                outputs = model(batch)
                outputs = outputs.view(-1)
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

                    if batch.size(0) == 0:
                        continue

                    outputs = model(batch)
                    outputs = outputs.view(-1)
                    predicted = (outputs.data > 0.5).to(int)
                    total_test += batch_size
                    correct_test += (predicted == labels).sum().item()
                    val_loss = criterion(outputs, labels.float())
                    avg_val_loss += val_loss.item()
                accuracy_test = 100 * correct_test / total_test
                accuracy_in_test.append(accuracy_test)
                avg_val_loss = avg_val_loss / total_test

                if print_accuracy:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {accuracy_train:.2f}%, Test Accuracy: {accuracy_test:.2f}%')

        # 打印超参数
        print(f"Generation: {generation}, Individual: {individual}, num_kernel_conv1: {num_kernel_conv1}, num_kernel_conv2: {num_kernel_conv2}, kernel_size_conv1: {kernel_size_conv1}, kernel_size_conv2: {kernel_size_conv2}, kernel_size_pooling: {kernel_size_pooling}")
        
        # 保存每个个体的训练图像
        plt.figure()
        plt.plot(range(1, len(accuracy_in_train) + 1), accuracy_in_train, label='Train Accuracy')
        plt.plot(range(1, len(accuracy_in_test) + 1), accuracy_in_test, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Generation {generation}, Individual {individual} Accuracy')
        plt.legend()
        plt.savefig(f'/Users/katerina/Documents/GAoptimizedCNN/train_2nd/gen_{generation}_ind_{individual}_accuracy.png')
        plt.close()

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
    num_kernel_conv1, num_kernel_conv2, kernel_size_conv1, kernel_size_conv2, kernel_size_pooling = decode_gene(solution)

    generation = ga_instance.generations_completed
    individual = solution_idx
    fitness = train_and_eval(num_kernel_conv1=num_kernel_conv1,
                             num_kernel_conv2=num_kernel_conv2,
                             kernel_size_conv1=kernel_size_conv1,
                             kernel_size_conv2=kernel_size_conv2,
                             kernel_size_pooling=kernel_size_pooling,
                             generation=generation,
                             individual=individual,
                             print_accuracy=False,
                             return_history=False)
    print(f"Generation: {generation}, Individual: {individual}, Fitness: {fitness}")
    return fitness

# 定义保存图形和参数的函数
def save_plot_and_params(ga_instance):
    generation = ga_instance.generations_completed
    print(f"Generation completed: {generation}")
    
# Save parameters to CSV
    solutions = ga_instance.population
    params = []
    for idx, solution in enumerate(solutions):
        num_kernel_conv1, num_kernel_conv2, kernel_size_conv1, kernel_size_conv2, kernel_size_pooling = decode_gene(solution)
        params.append({
            'Generation': generation,
            'Individual': idx,
            'num_kernel_conv1': num_kernel_conv1,
            'num_kernel_conv2': num_kernel_conv2,
            'kernel_size_conv1': kernel_size_conv1,
            'kernel_size_conv2': kernel_size_conv2,
            'kernel_size_pooling': kernel_size_pooling,
        })

    params_df = pd.DataFrame(params)
    params_df.to_csv(f'/Users/katerina/Documents/GAoptimizedCNN/train_2nd/params_gen_{generation}.csv', index=False)

    # Save to Excel
    params_df.to_excel(f'/Users/katerina/Documents/GAoptimizedCNN/train_2nd/params_gen_{generation}.xlsx', index=False)

    # 保存图形
    plt.figure()
    plt.plot(range(1, len(ga_instance.best_solutions_fitness) + 1), ga_instance.best_solutions_fitness, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Generations')
    plt.legend()
    plt.savefig(f'/Users/katerina/Documents/GAoptimizedCNN/train_2nd/fitness_plot_gen_{ga_instance.generations_completed}.png')
    plt.close()

    # 保存参数
    solutions = ga_instance.population
    params = []
    for solution in solutions:
        num_kernel_conv1 = initial_num_kernel_conv1
        num_kernel_conv2 = initial_num_kernel_conv2
        kernel_size_conv1 = initial_kernel_size_conv1
        kernel_size_conv2 = initial_kernel_size_conv2
        kernel_size_pooling = initial_kernel_size_pooling

        params.append({
            'generation': ga_instance.generations_completed,
            'num_kernel_conv1': num_kernel_conv1,
            'num_kernel_conv2': num_kernel_conv2,
            'kernel_size_conv1': kernel_size_conv1,
            'kernel_size_conv2': kernel_size_conv2,
            'kernel_size_pooling': kernel_size_pooling,
        })

    params_df = pd.DataFrame(params)
    params_df.to_csv(f'/Users/katerina/Documents/GAoptimizedCNN/train_2nd/params_gen_{ga_instance.generations_completed}.csv', index=False)

# 设置随机种子
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 编码基因的函数
def encode_gene(num_kernel_conv1, num_kernel_conv2, kernel_size_conv1, kernel_size_conv2, kernel_size_pooling):
    gene = []
    gene += list(map(int, bin(num_kernel_conv1 - 1)[2:].zfill(6)))
    gene += list(map(int, bin(num_kernel_conv2 - 1)[2:].zfill(6)))
    for size in kernel_size_conv1:
        gene += list(map(int, bin(size - 2)[2:].zfill(5)))
    for size in kernel_size_conv2:
        gene += list(map(int, bin(size - 2)[2:].zfill(5)))
    gene += list(map(int, bin(kernel_size_pooling - 1)[2:].zfill(5)))
    return gene

# 解码基因的函数
def decode_gene(gene):
    kernel_size_conv1 = [(int("".join(map(str, gene[12 + i*5:17 + i*5])), 2) % 31 + 2) for i in range(7)]
    kernel_size_conv2 = [(int("".join(map(str, gene[47 + i*5:52 + i*5])), 2) % 31 + 2) for i in range(7)]
    num_kernel_conv1 = int("".join(map(str, gene[:6])), 2) % 63 + 1
    num_kernel_conv2 = int("".join(map(str, gene[6:12])), 2) % 63 + 1
    kernel_size_pooling = int("".join(map(str, gene[82:87])), 2) % 32 + 1

    # 确保在基因全为 0 时产生一个非零值
    if all(g == 0 for g in gene):
        num_kernel_conv1 = random.randint(1, 63)
        num_kernel_conv2 = random.randint(1, 63)
        kernel_size_conv1 = [random.randint(1, 32) for _ in range(7)]
        kernel_size_conv2 = [random.randint(1, 32) for _ in range(7)]
        kernel_size_pooling = random.randint(1, 32)

    return num_kernel_conv1, num_kernel_conv2, kernel_size_conv1, kernel_size_conv2, kernel_size_pooling

# 初始化种群的函数
def initial_population(sol_per_pop, num_genes):
    population = []
    for _ in range(sol_per_pop):
        gene = [random.randint(0, 1) for _ in range(num_genes)]
        population.append(gene)
    return population

# 生成下一代的函数
def next_generation(population, fitness_scores, num_parents_mating):
    parents = select_parents(population, fitness_scores, num_parents_mating)
    offspring = crossover(parents)
    offspring = mutate_gene(offspring)
    return parents + offspring

# 选择父代的函数
def select_parents(population, fitness_scores, num_parents_mating):
    parents = []
    for _ in range(num_parents_mating):
        max_fitness_idx = np.where(fitness_scores == np.max(fitness_scores))
        max_fitness_idx = max_fitness_idx[0][0]
        parents.append(population[max_fitness_idx])
        fitness_scores[max_fitness_idx] = -99999999999
    return parents

# 交叉操作的函数
def crossover(parents):
    offspring = []
    for k in range(len(parents) // 2):
        parent1 = parents[2 * k]
        parent2 = parents[2 * k + 1]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.append(child1)
        offspring.append(child2)
    return offspring

# 变异基因的函数
def mutate_gene(offspring):
    mutation_percent_genes = 0.1

    for gene_idx in range(len(offspring)):
        if random.random() < mutation_percent_genes:
            mutation_type = random.choice(["random", "swap", "inversion", "scramble"])
            if mutation_type == "random":
                # 随机选择一个基因进行变异
                mutated_gene = random.choice([0, 1])
                offspring[gene_idx][random.randint(0, len(offspring[gene_idx]) - 1)] = mutated_gene
            elif mutation_type == "swap":
                # 交换两个基因的位置
                index1 = random.randint(0, len(offspring[gene_idx]) - 1)
                index2 = random.randint(0, len(offspring[gene_idx]) - 1)
                offspring[gene_idx][index1], offspring[gene_idx][index2] = offspring[gene_idx][index2], offspring[gene_idx][index1]
            elif mutation_type == "inversion":
                # 反转一段基因
                start_index = random.randint(0, len(offspring[gene_idx]) - 1)
                end_index = random.randint(start_index, len(offspring[gene_idx]) - 1)
                offspring[gene_idx][start_index:end_index] = reversed(offspring[gene_idx][start_index:end_index])
            elif mutation_type == "scramble":
                # 打乱一段基因
                start_index = random.randint(0, len(offspring[gene_idx]) - 1)
                end_index = random.randint(start_index, len(offspring[gene_idx]) - 1)
                scrambled_segment = offspring[gene_idx][start_index:end_index]
                random.shuffle(scrambled_segment)
                offspring[gene_idx][start_index:end_index] = scrambled_segment

            # 确保至少有一个基因被设置为 1
            if all(gene == 0 for gene in offspring[gene_idx]):
                offspring[gene_idx][random.randint(0, len(offspring[gene_idx]) - 1)] = 1

    return offspring

# Step 6: Training and Evaluation
def train_and_eval(num_kernel_conv1, num_kernel_conv2, kernel_size_conv1, kernel_size_conv2, kernel_size_pooling, generation, individual, print_accuracy=False, return_history=False):
    print(f"开始训练模型: Generation {generation}, Individual {individual}")
    model = TimeSeriesCNN(input_length=TIME_WINDOW_SIZE, 
                          num_kernel_conv1=num_kernel_conv1, 
                          num_kernel_conv2=num_kernel_conv2, 
                          kernel_size_conv1=kernel_size_conv1, 
                          kernel_size_conv2=kernel_size_conv2, 
                          kernel_size_pooling=kernel_size_pooling).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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

                if batch.size(0) == 0:
                    continue

                # Forward pass
                outputs = model(batch)
                outputs = outputs.view(-1)
                predicted = (outputs.data > 0.5).to(int)
                total_train += batch_size
                correct_train += (predicted == labels).sum().item()

                # Compute loss and perform backpropagation
                loss = criterion(outputs, labels.float())
                avg_loss_per_epoch += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batch += 1

            avg_loss_per_epoch /= num_batch
            accuracy_train = correct_train / total_train
            print(f"训练集 - 损失: {avg_loss_per_epoch}, 准确率: {accuracy_train}")
            loss_value.append(avg_loss_per_epoch)
            accuracy_in_train.append(accuracy_train)

            model.eval()
            total_test = test_sequences_tensor.shape[0]
            correct_test = 0

            with torch.no_grad():
                outputs = model(test_sequences_tensor)
                outputs = outputs.view(-1)
                predicted = (outputs.data > 0.5).to(int)
                correct_test += (predicted == test_labels_tensor).sum().item()
                accuracy_test = correct_test / total_test
                print(f"测试集 - 准确率: {accuracy_test}")
                accuracy_in_test.append(accuracy_test)
        
        print(f"Generation {generation}, Individual {individual}, Test Accuracy: {accuracy_in_test[-1]}")

        if return_history:
            return accuracy_in_test[-1], (loss_value, accuracy_in_train, accuracy_in_test)
        else:
            return accuracy_in_test[-1]

    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

# Step 7: Fitness Function
def fitness_func(solution, solution_idx):
    decoded_solution = decode_gene(solution)
    fitness = train_and_eval(*decoded_solution, generation=0, individual=solution_idx)
    return fitness

# Step 8: Save plot and parameters function
def save_plot_and_params(generation, best_fitness, best_solution):
    print(f"Generation {generation}: Best Fitness = {best_fitness}")
    print(f"Best solution: {decode_gene(best_solution)}")

# Step 9: Initialize Population and Run GA
sol_per_pop = 100
num_genes = 87
num_parents_mating = 5
num_generations = 10
pop_size = (sol_per_pop, num_genes)
crossover_probability = 0.8
mutation_probability = 0.2

population = initial_population(sol_per_pop, num_genes)
fitness_scores = np.zeros(sol_per_pop)

best_fitness = 0
best_solution = None

for generation in range(num_generations):
    print(f"Generation {generation}")
    for i in range(sol_per_pop):
        fitness_scores[i] = fitness_func(population[i], i)

    parents = select_parents(population, fitness_scores, num_parents_mating)
    offspring = crossover(parents)
    offspring = mutate_gene(offspring)

    population = parents + offspring

    best_fitness_idx = np.argmax(fitness_scores)
    best_fitness = fitness_scores[best_fitness_idx]
    best_solution = population[best_fitness_idx]

    save_plot_and_params(generation, best_fitness, best_solution)

# Step 10: Get Best Solution
best_solution_params = decode_gene(best_solution)
print(f"Best solution parameters: {best_solution_params}")

# Step 11: Train and Evaluate Best Model
final_accuracy, history = train_and_eval(*best_solution_params, generation=num_generations, individual=0, return_history=True)

# Step 12: Check System Resource Usage
import os, psutil

process = psutil.Process(os.getpid())
print(f"Memory used: {process.memory_info().rss / 1024 / 1024:.2f} MB")
print(f"CPU used: {psutil.cpu_percent()}%")

# Step 13: Plot Training and Testing Accuracy
import matplotlib.pyplot as plt

loss_value, accuracy_in_train, accuracy_in_test = history

plt.plot(loss_value, label='Loss')
plt.plot(accuracy_in_train, label='Train Accuracy')
plt.plot(accuracy_in_test, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# Step 14: Save Model
#torch.save(model.state_dict(), "best_model.pth")

# Step 14: Save Model
#torch.save(model.state_dict(), "/Users/katerina/Documents/GAoptimizedCNN/train_2nd/model.pth")

