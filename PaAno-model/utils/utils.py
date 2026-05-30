from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_model(model_path, device):
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


@torch.no_grad()
def create_memory_bank(model, data_loader, device, num_cores=None):  
    model.eval()
    embeddings = []
    indices = []
    
    for data, batch_indices in tqdm(data_loader, total=len(data_loader), desc="Extracting embeddings"):
        data = data.to(device)
        # 推理获取所有的特征向量
        h = model.embedding(data)
        embeddings.append(h.detach().cpu().float())  
        indices.append(batch_indices.detach().cpu())

    embeddings_tensor = torch.cat(embeddings, dim=0)
    indices_tensor    = torch.cat(indices, dim=0)
    num_samples       = embeddings_tensor.size(0)

 
    if num_cores is None:
        return embeddings_tensor, indices_tensor

  
    # if isinstance(num_cores, float):
    #     k = int(round(num_cores * num_samples))    
    # else:
    #     k = int(num_cores)
    # # 记忆库至少500个
    # min_cores_eff = min(500, max(1, num_samples - 1)) 
    # num_cores = max(min_cores_eff, min(k, num_samples - 1))

    if num_cores >= num_samples:
        return embeddings_tensor, indices_tensor

  
    flattened = embeddings_tensor.view(num_samples, -1)
    flattened = F.normalize(flattened, p=2, dim=1)

    # 聚类
    mbk = MiniBatchKMeans(
        n_clusters=num_cores,
        init='k-means++',
        random_state=42,
        batch_size=max(8192, num_cores),   
        max_iter=50,                       
        n_init=1,                          
        reassignment_ratio=0.01
    )
    mbk.fit(flattened.numpy())


    centers = torch.tensor(mbk.cluster_centers_, dtype=flattened.dtype)  
    distances = torch.cdist(flattened, centers, p=2)   
    core_indices = torch.argmin(distances, dim=0)      
    # 筛选
    embeddings_tensor = embeddings_tensor[core_indices]
    indices_tensor    = indices_tensor[core_indices]

    return embeddings_tensor, indices_tensor



# Source code : https://github.com/decisionintelligence/CATCH/blob/master/ts_benchmark/baselines/catch/layers/RevIN.py
import torch
import torch.nn as nn



def triplet_grad(x, r=0.01): # for gradual update
    return x.detach() + (x - x.detach()) * r
