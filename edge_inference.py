import torch  
import numpy as np  
from models.egat import EGAT  
 
THRESHOLD = 0.7  
 
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
 
buffer = []  
 
WINDOW = 20  
 
 
def infer(sample):  
 
    global buffer  
 
    buffer.append(sample)  
 
    if len(buffer) > WINDOW:  
        buffer.pop(0)  
 
    if len(buffer) == WINDOW:  
 
        x = np.array(buffer)  
 
        x = torch.tensor(  
            x, 
            dtype=torch.float32  
        ).unsqueeze(0)  
 
        with torch.no_grad():  
 
            y_hat = model(x)  
 
            score = y_hat[0][1].item()  
 
        if score >= THRESHOLD:  
 
            print("Attack detected:", score)  
 
        else: 
 
            print("Normal traffic")
