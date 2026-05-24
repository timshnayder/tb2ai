import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Initial batch size configuration for dynamic scaling
INITIAL_BATCH_SIZE = 32
MAX_BATCH_SIZE = 256
BATCH_SCALE_EPOCHS = 3 # Double the batch size every 3 epochs
current_batch_size = INITIAL_BATCH_SIZE

# Layout configuration:
# 10 = mirror
# 11 = spray
LAYOUT_ID = 10

suffix = ""
if LAYOUT_ID == 11:
    suffix = "_spray"
elif LAYOUT_ID == 10:
    suffix = "_mirror"
else:
    suffix = f"_layout_{LAYOUT_ID}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training will run on: {device}")
print(f"Initial batch size: {current_batch_size} (scales to {MAX_BATCH_SIZE})")
print(f"Training {suffix[1:]} model.")

with open(f'vocab{suffix}.json', 'r') as f: vocab = json.load(f)
with open(f'tokenized_climbs{suffix}.json', 'r') as f: tokenized_climbs = json.load(f)
with open(f'climb_grades{suffix}.json', 'r') as f: climb_grades = json.load(f)
with open(f'vocab_coords{suffix}.json', 'r') as f: vocab_coords_dict = json.load(f)
with open(f'no_matching{suffix}.json', 'r') as f: no_matching = json.load(f)
with open(f'climb_angles{suffix}.json', 'r') as f: climb_angles = json.load(f)

#input - target pairs
X_data, y_data, X_grades, X_angles, X_no_match = [], [], [], [], []
for climb, grade, angle, no_match in zip(tokenized_climbs, climb_grades, climb_angles, no_matching):
    for i in range(1, len(climb)):
        X_data.append(climb[:i])
        y_data.append(climb[i])
        X_grades.append(grade)
        X_angles.append(angle)
        X_no_match.append(no_match)

#pad x so we can turn into tensor, (y and grades are already all 1 element long)
max_len = max(len(seq) for seq in X_data)
X_padded = [[0] * (max_len - len(seq)) + seq for seq in X_data]

#turn all into tensors
X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y_data, dtype=torch.long)
grades_tensor = torch.tensor(X_grades, dtype=torch.long)
angle_tensor = torch.tensor(X_angles, dtype=torch.long)
no_match_tensor = torch.tensor(X_no_match, dtype=torch.float32)

coord_matrix = [vocab_coords_dict[str(i)] for i in range(len(vocab))]
coord_tensor = torch.tensor(coord_matrix, dtype=torch.float32).to(device)

class ConditionalClimbDataset(Dataset):
    def __len__(self): return len(X_tensor)
    def __getitem__(self, idx): return X_tensor[idx], grades_tensor[idx], angle_tensor[idx], no_match_tensor[idx], y_tensor[idx]

dataset = ConditionalClimbDataset()
dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)

# 2. Define Conditional LSTM Model
class ConditionalClimbGenerator(nn.Module):
    def __init__(self, vocab_size, num_grades, max_angle, embedding_dim, hidden_dim, coord_tensor):
        super(ConditionalClimbGenerator, self).__init__()
        self.hold_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Grade embedding gives the AI a "mood context" (e.g., V3 vs V10)
        self.grade_embedding = nn.Embedding(num_grades, embedding_dim)
        #half size so doesn't overpower holds
        self.angle_embedding = nn.Embedding(max_angle, embedding_dim//2)
        
        self.register_buffer('coords', coord_tensor)
        # hold (64) + grade (64) + angle(32) + coord x(1) + coord y(1) + no_match(1)
        input_size = (embedding_dim*2) + (embedding_dim//2)+3

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, holds, grades, angles, no_match):
        hold_embeds = self.hold_embedding(holds) # Shape: [batch, seq_len, embed_dim]
        
        # Embed the grade and expand it so it tags along with every hold in the sequence
        grade_embeds = self.grade_embedding(grades).unsqueeze(1).repeat(1, holds.size(1), 1)
        #embed the angle and expand it
        angle_embeds = self.angle_embedding(angles).unsqueeze(1).repeat(1, holds.size(1), 1)
        
        no_match_expanded = no_match.unsqueeze(1).unsqueeze(2).repeat(1, holds.size(1), 1)

        hold_coords = self.coords[holds]

        # Combine the inputs together
        combined_input = torch.cat((hold_embeds, grade_embeds, angle_embeds, hold_coords, no_match_expanded), dim=2)
        
        lstm_out, _ = self.lstm(combined_input)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

# Initialize and Train
VOCAB_SIZE = len(vocab)
#set safe maximums
NUM_GRADES = 45 
MAX_ANGLE = 15
model = ConditionalClimbGenerator(VOCAB_SIZE, NUM_GRADES, MAX_ANGLE, embedding_dim=64, hidden_dim=256, coord_tensor=coord_tensor)
model = model.to(device)

#try loading old model, if not start fresh
MODEL_FILENAME = f'conditional_climb_ai{suffix}.pth'
if os.path.exists(MODEL_FILENAME):
    try:
        model.load_state_dict(torch.load(MODEL_FILENAME))
        print(f"-> Found existing brain! Successfully loaded '{MODEL_FILENAME}'. Resuming training...")
    except Exception as e:
        print(f"-> Found '{MODEL_FILENAME}', but it failed to load (architecture likely changed).")
else:
    print(f"-> No existing brain found. Starting fresh.")
#

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.002)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

epochs = int(input("Enter number of epochs to train for: "))
for epoch in range(epochs):
    # Dynamic Batch Size Scaling
    if epoch > 0 and epoch % BATCH_SCALE_EPOCHS == 0 and current_batch_size < MAX_BATCH_SIZE:
        current_batch_size = min(MAX_BATCH_SIZE, current_batch_size * 2)
        print(f"\n---> [Scale Up] Dynamic Batch Size increased to: {current_batch_size}")
        dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)

    model.train()
    total_loss = 0
    for holds, grades, angles, no_match, targets in dataloader:
        holds = holds.to(device)
        grades = grades.to(device)
        angles = angles.to(device)
        no_match = no_match.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(holds, grades, angles, no_match)

        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avgLoss = total_loss/len(dataloader)
    scheduler.step(avgLoss)
    currentLr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avgLoss:.4f} | LR: {currentLr}")

torch.save(model.state_dict(), MODEL_FILENAME)
print(f"Model trained and saved to {MODEL_FILENAME}!")