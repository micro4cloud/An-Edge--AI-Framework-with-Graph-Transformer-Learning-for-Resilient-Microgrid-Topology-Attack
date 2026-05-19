import torch  
from sklearn.metrics import (  
    accuracy_score,  
    precision_score,  
    recall_score,  
    f1_score  
) 
 
from models.egat import EGAT  
 
model = EGAT(  
    input_dim=20,  
    hidden_dim=64,  
    num_heads=4,  
    num_classes=2  
) 
 
model.load_state_dict(  
    torch.load(  
        "results/checkpoints/egat_model.pth"  
    ) 
) 
 
model.eval()  
 
with torch.no_grad():  
 
    pred = model(X_test)  
 
    pred = torch.argmax(pred, dim=1)  
 
acc = accuracy_score(y_test, pred)  
prec = precision_score(y_test, pred)  
rec = recall_score(y_test, pred)  
f1 = f1_score(y_test, pred)  

 
print("Accuracy:", acc)  
print("Precision:", prec)  
print("Recall:", rec)  
print("F1 -score:", f1)
