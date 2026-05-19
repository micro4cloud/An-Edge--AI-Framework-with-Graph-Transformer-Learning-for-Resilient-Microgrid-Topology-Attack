import torch  
import torch.nn as nn  
from torch.utils.data import TensorDataset, DataLoader  

from sklearn.metrics import accuracy_score  
from models.egat import EGAT  
from utils.preprocessin g import *  
 
SEQ_LEN = 20  
BATCH_SIZE = 32  
EPOCHS = 30  
LR = 0.001  
 
 
df = load_dataset("data/raw/dnp3.csv")  
 
X, y = preprocess(df)  
 
X_seq, y_seq = create_sequences(X, y, SEQ_LEN)  
 
from sklearn.model_selection import train_test_split  
 
X_train, X_test, y_train, y_test = train_test_split(  
    X_seq, 
    y_seq, 
    test_size=0.2,  
    random_state=42  
) 
 
X_train = torch.tensor(X_train, dtype=torch.float32)  
X_test = torch.tensor(X_test, dtype=torch.float32)  
 
y_train = torch.tensor(y_train, dt ype=torch.long)  
y_test = torch.tensor(y_test, dtype=torch.long)  
 
train_loader = DataLoader(  
    TensorDataset(X_train, y_train),  
    batch_size=BATCH_SIZE,  
    shuffle=True  
) 
 
model = EGAT(  
    input_dim=X_train.shape[2],  
    hidden_dim=64,  
    num_heads=4 , 
    num_classes=2  
) 
 
criterion = nn.CrossEntropyLoss()  
 
optimizer = torch.optim.Adam(  
    model.parameters(),  
    lr=LR 
) 
 
for epoch in range(EPOCHS):  
 
    model.train()  
 
    total_loss = 0  
 
    for xb, yb in train_loader:  
 
        optimizer.zero_grad()  
 

        pred = model(xb)  
 
        loss = criterion(pred, yb)  
 
        loss.backward()  
 
        optimizer.step()  
 
        total_loss += loss.item()  
 
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")  
 
 
torch.save(  
    model.state_di ct(), 
    "results/checkpoints/egat_model.pth"  
) 
