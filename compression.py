import torch  
import torch.nn.utils.prune as prune  
 
 
def prune_model(model, amount=0.3):  
 
    for module in model.modules():  
 
        if isinstance(module, torch.nn.Linear):  
 
            prune.l1_unstructured(  
                module, 
                name='weight',  
                amount=amount  
            ) 
 
    return model  
 
 
 
def quantize_model(model):  
 
    model.qconfig = torch.quanti zation.get_default_qconfig('fbgemm')  
 
    torch.quantization.prepare(model, inplace=True)  
 
    torch.quantization.convert(model, inplace=True)  
 
    return model
