import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sched
from tqdm import tqdm
import random
import annoy

import gc

 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

'''
Data Loader
'''

def train_val_split(dataset, val_size = 0.2, train_size=0.8, n_splits = None): # Сплит по artistid
    artist_ids = dataset['artistid'].unique()
    if n_splits != None:
        spl = ShuffleSplit(n_splits=n_splits, test_size = 0.1, train_size = 0.1)
        splits = list(spl.split(artist_ids))
        splits_data = []
        for split in splits:
            trainset = dataset[dataset['artistid'].isin(split[0])].copy()
            valset = dataset[dataset['artistid'].isin(split[1])].copy()
            splits_data.append([trainset, valset])
        return splits_data
    else:    
        train_artist_ids, val_artist_ids = train_test_split(artist_ids, test_size = val_size, train_size = train_size) ## !!! test_size = val_size, 
        trainset = dataset[dataset['artistid'].isin(train_artist_ids)].copy()
        valset = dataset[dataset['artistid'].isin(val_artist_ids)].copy()
        return trainset, valset

class FeaturesLoader: 
    def __init__(self, features_dir_path, meta_info, device='cpu', crop_size = 60):
        self.features_dir_path = features_dir_path
        self.meta_info = meta_info ## информация о путях до данных
        ### Делает словарь {trackid : путь до данных}
        self.trackid2path = meta_info.set_index('trackid')['archive_features_path'].to_dict()
        self.crop_size = crop_size
        self.device = device
        
    def _load_item(self, track_id):
            
        ### берём один файл данных 
        track_features_file_path = self.trackid2path[track_id] 
        ### данные имеют размер (512, 81) (81 чанк, embedding 512)
        track_features = np.load(os.path.join(self.features_dir_path, track_features_file_path))
        ## обрезка данных, чтобы взять 60 последовательных чанка посередине
        padding = (track_features.shape[1] - self.crop_size) // 2
        return track_features[:, padding:padding+self.crop_size] ## (512, 60)
    
    def load_batch(self, tracks_ids):
               
        ## Сборка в батчи
        batch = [self._load_item(track_id) for track_id in tracks_ids]

        return torch.tensor(np.array(batch)).to(self.device) ### (1024, 512, 60)

class TrainLoader:
    def __init__(self, features_loader, batch_size = 256, features_size = (512, 60)):
        self.features_loader = features_loader
        self.batch_size = batch_size
        self.features_size = features_size
        ### Сборка в треки всех артистов.
        ### Data Frame index - artistid, list[id трэков], list[пути]
        self.artist_track_ids = self.features_loader.meta_info.groupby('artistid').agg(list)
        
    def _generate_pairs(self, track_ids):
        ### Разбивка треков по парам, чтобы потом считать между ними расстояние
        np.random.shuffle(track_ids)
        pairs = [track_ids[i-2:i] for i in range(2, len(track_ids)+1, 2)]
        return pairs
        
    def _get_pair_ids(self):
        ## in each pair songs are of the same artist 
        artist_track_ids = self.artist_track_ids.copy()
        artist_track_pairs = artist_track_ids['trackid'].map(self._generate_pairs) ## на пары разбиваются композиции одного артиста
        artist_track_pairs_exploded = artist_track_pairs.explode().dropna().sample(frac=1)
        for pair_ids in artist_track_pairs_exploded:
            yield pair_ids
            
    def _get_batch(self, batch_ids): ## batch_ids (512, 2)
                
        batch_ids = np.array(batch_ids).reshape(-1) ## (1024)
        batch_features = self.features_loader.load_batch(batch_ids) ## (1024, 512, 60) = (2*batch_size, 512, 60)
        batch_features = batch_features.reshape(self.batch_size, 2, *self.features_size)
        return batch_features ## (512, 2, 512, 60) =  = (batch_size, 2, 512, 60)
        
    def __iter__(self):
                
        batch_ids = []
        
        for pair_ids in self._get_pair_ids(): ##(74503?, 2)
            batch_ids.append(pair_ids)
            ### pair_ids - это генератор, поэтому при новом запуске, он считывается с прошлого места
            ### прочитанные элементы удаляются 
            if len(batch_ids) == self.batch_size:
                batch = self._get_batch(batch_ids)
                yield batch
                batch_ids = []

class TestLoader:
    def __init__(self, features_loader, batch_size = 256, features_size = (512, 60)):
        self.features_loader = features_loader
        self.batch_size = batch_size
        self.features_size = features_size
        
    def __iter__(self):
        batch_ids = []
        for track_id in tqdm(self.features_loader.meta_info['trackid'].values):
            batch_ids.append(track_id)
            if len(batch_ids) == self.batch_size:
                yield batch_ids, self.features_loader.load_batch(batch_ids) 
                batch_ids = []
        if len(batch_ids) > 0:
            yield batch_ids, self.features_loader.load_batch(batch_ids) 
            
#%%

'''
Loss & Metrics
'''

class NT_Xent(nn.Module):
    
    '''
    Считает loss и avg_rank
    '''
    
    
    def __init__(self, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean") # поменял sum на mean и убрал деление на N внизу
        self.similarity_f = nn.CosineSimilarity(dim=2) # 

    def mask_correlated_samples(self, batch_size):
        ### матрица из 4х блоков (batch_size x batch_size).
        ### Каждый блок заполнен True, а по диагонали False
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


    def forward(self, z_i, z_j): #z_i = (batch, 128)
        
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
 
        ## матрица схожести размером (2*batch, 2*batch)
        ## т.е. матрица свхожести объединённого набора векторов (z_i + z_j)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size) # расстояния внутри каждой пары
        sim_j_i = torch.diag(sim, -batch_size) # вроде равно предыдущему

        mask = self.mask_correlated_samples(batch_size)
        ## Расстояние между векторами в одной паре - они одинаковые
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) ## (2*batch, 1)
        ## Развёрнутая в вектор матрица расстояний между всеми элементами кроме тех, которые в парах. Они разные.
        negative_samples = sim[mask].reshape(N, -1) ## (2*batch, 2*batch-2)

        labels = torch.zeros(N).to(positive_samples.device).long() ## 2*batch
        # labels все нули, т.к. positive samples стоят первыми в логитах
        logits = torch.cat((positive_samples, negative_samples), dim=1) ## (2*batch, 2*batch-1)
        # Есть (batch-1) классов, batch объектов.
        loss = self.criterion(logits, labels)


        return loss


def get_ranked_list(embeds, top_size, annoy_num_trees = 32):
    # embeds - словарь {track_id:embeddings}
    # annoy - библиотека для нахождения ближайшей точки к данной
    annoy_index = None
    annoy2id = []
    id2annoy = dict()
    for track_id, track_embed in embeds.items(): # иттерируется по словарю {track_id:embeddings}
        id2annoy[track_id] = len(annoy2id) ## просто словарь {track_id:номер в списке}
        annoy2id.append(track_id)
        if annoy_index is None:
            annoy_index = annoy.AnnoyIndex(len(track_embed), 'angular')
        annoy_index.add_item(id2annoy[track_id], track_embed)
    annoy_index.build(annoy_num_trees, n_jobs=-1)
    ranked_list = dict()
    for track_id in embeds.keys():
        candidates = annoy_index.get_nns_by_item(id2annoy[track_id], top_size+1)[1:] # exclude trackid itself
        candidates = list(filter(lambda x: x != id2annoy[track_id], candidates))
        ranked_list[track_id] = [annoy2id[candidate] for candidate in candidates]
    return ranked_list

def position_discounter(position):
    return 1.0 / np.log2(position+1)   

def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg

def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg

def eval_submission(submission, gt_meta_info, top_size = 100):
    track2artist_map = gt_meta_info.set_index('trackid')['artistid'].to_dict()
    artist2tracks_map = gt_meta_info.groupby('artistid').agg(list)['trackid'].to_dict()
    ndcg_list = []
    for query_trackid in tqdm(submission.keys()):
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count-1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map, top_size=top_size)
        try:
            ndcg_list.append(dcg/ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)

#%%

'''
Train & Inference functions
'''

class BasicNet(nn.Module):
    def __init__(self, output_features_size): ## по умолчанию 256
        super().__init__()
        self.output_features_size = output_features_size       
        self.trans_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=0.1, activation='gelu')
        self.tranformer = nn.TransformerEncoder(self.trans_layer, num_layers=3)
        self.lin = nn.Linear(512, 256, bias=False)

    def forward(self, x): ## (batch, 512, 60) = (batch, embeddings, chunk) == (batch, channe, L)
        
        x = torch.transpose(x, 1,2)  
        x = self.tranformer(x)
        x = x.mean(axis = 1)
                
        x = self.lin(x)
                
        return x #(batch, 256)
    
class BasicNet_bias(nn.Module):
    def __init__(self, output_features_size): ## по умолчанию 256
        super().__init__()
        self.output_features_size = output_features_size      
        
        self.trans_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=0.5, activation='gelu')
        self.tranformer = nn.TransformerEncoder(self.trans_layer, num_layers=3)
        self.lin = nn.Linear(512, 256, bias=True)


    def forward(self, x): ## (batch, 512, 60) = (batch, embeddings, chunk) == (batch, channe, L)
        
        x = torch.transpose(x, 1,2)  
        x = self.tranformer(x)
        x = x.mean(axis = 1)
        x = self.lin(x)
                
        return x #(batch, 256)

class SimCLR(nn.Module):
    
    ### модель состоящая из энкодера (последовательные свёртки, выход - вектор 256) и проектора- 256 --> 128
    
    def __init__(self, encoder, projection_dim):
        super().__init__()
        self.encoder = encoder
        self.n_features = encoder.output_features_size ## 256
        self.projection_dim = projection_dim ## 128
        
    def forward(self, x_i, x_j): ### На вход идут пары песен, которые были выше сгенерированы
        h_i = self.encoder(x_i) ### сворачиваем песню (60 x 512) в вектор 256
        h_j = self.encoder(x_j)
        
        return h_i, h_j
    

def inference(model, loader):
    ## создаёт словарь {track_id:embeddings}
    global tracks_features
    
    embeds = dict()
    for tracks_ids, tracks_features in loader:
        with torch.no_grad():
            tracks_embeds = model(tracks_features) ### сворачиваем песню (60 x 512) в вектор 256
            for track_id, track_embed in zip(tracks_ids, tracks_embeds):
                embeds[track_id] = track_embed.cpu().numpy()
    return embeds

def train(module, train_loader, val_loader, valset_meta, optimizer, criterion, num_epochs, checkpoint_path,
          temperature_list, scheduler, top_size = 100):
    max_ndcg = None
    
    global train_progres
    
    train_progres = []  

    for epoch in range(num_epochs):
        
        torch.cuda.empty_cache()
        gc.collect()
        
        if epoch <= len(temperature_list) - 1:
            temp = temperature_list[epoch]
        else:
            temp = 0.04
        #temp = 0.05
        print(f'Temperature = {temp}')
        
        criterion = NT_Xent(temperature = temp)        
        
        module.train()
        print(f"LR = {optimizer.param_groups[0]['lr']}")
        
        for batch in tqdm(train_loader):

            optimizer.zero_grad()            

            x_i, x_j = batch[:, 0, :, :], batch[:, 1, :, :]
            h_i, h_j = module(x_i, x_j)
            loss = criterion(h_i, h_j)
            loss.backward()

            optimizer.step()

        scheduler.step()
        
        
        with torch.no_grad():
            module.eval()
            model_encoder = module.encoder
            embeds_encoder = inference(model_encoder, val_loader) ## создаёт словарь {track_id:embeddings}
            ranked_list_encoder = get_ranked_list(embeds_encoder, top_size)
            val_ndcg_encoder = eval_submission(ranked_list_encoder, valset_meta)
            
            print("Validation nDCG on epoch {}".format(epoch))
            print("Encoder - {}".format(val_ndcg_encoder))
            if (max_ndcg is None) or (val_ndcg_encoder > max_ndcg):
                max_ndcg = val_ndcg_encoder
                torch.save(module.state_dict(), checkpoint_path)
            
            train_progres.append([loss.detach().cpu().numpy(), val_ndcg_encoder])
            print(f"Last loss = {loss.detach().cpu().numpy()}")

def val_score(module, val_loader, valset_meta, top_size = 100):
    
    with torch.no_grad():
        module.eval()  ##
        model_encoder = module.encoder
        embeds_encoder = inference(model_encoder, val_loader) ## создаёт словарь {track_id:embeddings}
        ranked_list_encoder = get_ranked_list(embeds_encoder, top_size)
        val_ndcg_encoder = eval_submission(ranked_list_encoder, valset_meta)
        
        print('')        
        print("Encoder - {}".format(val_ndcg_encoder))

def save_submission(submission, submission_path):
    with open(submission_path, 'w') as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))

#%%

'''
Seed
'''

seed = 142

def seed_everyting(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everyting(seed)

#%%

TRAINSET_DIRNAME = 'train_features'
TESTSET_DIRNAME = 'test_features'
TRAINSET_META_FILENAME = 'train_meta.tsv'
TESTSET_META_FILENAME = 'test_meta.tsv'
SUBMISSION_FILENAME = 'submission.txt'
MODEL_FILENAME = 'model.pt'
CHECKPOINT_FILENAME = 'best.pt'

DATA_DIR = 'C:\Timur\Data\Yandex audio artist prediction'
MODEL_DIR = 'D:\Timur\Data\Yandex audio artist prediction\model_transformer base'

#%%

TRAINSET_PATH = os.path.join(DATA_DIR, TRAINSET_DIRNAME)
TESTSET_PATH = os.path.join(DATA_DIR, TESTSET_DIRNAME)
TRAINSET_META_PATH = os.path.join(DATA_DIR, TRAINSET_META_FILENAME)
TESTSET_META_PATH = os.path.join(DATA_DIR, TESTSET_META_FILENAME)
SUBMISSION_PATH = SUBMISSION_FILENAME
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_FILENAME)

#%%

BATCH_SIZE = 256
N_CHANNELS = 256
PROJECTION_DIM = 128
NUM_EPOCHS = 8
LR = 1e-4
TEMPERATURE = 0.1

#%%

train_meta_info0 = pd.read_csv(TRAINSET_META_PATH, sep='\t')
test_meta_info = pd.read_csv(TESTSET_META_PATH, sep='\t')

#%%

sim_clr = SimCLR(
    encoder = BasicNet(N_CHANNELS),
    projection_dim = PROJECTION_DIM
).to(device)

#%%

for seedi in [142]:#42, 142, 242, 342, 442, 162, 262, 362, 462, 562, 662, 42, 142, 242, 342, 
           
    seed_everyting(seedi)

    torch.cuda.empty_cache()
    gc.collect()
    
    train_meta_info, validation_meta_info = train_val_split(train_meta_info0, val_size=0.1, train_size = 0.9)
    
    CHECKPOINT_FILENAME = f'best_{seedi}.pt'
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_FILENAME)
    
    
    if seedi in [162, 262, 362, 462, 562, 662]:
        sim_clr = SimCLR(
            encoder = BasicNet_bias(N_CHANNELS),
            projection_dim = PROJECTION_DIM
        ).to(device)
    else:
        sim_clr = SimCLR(
            encoder = BasicNet(N_CHANNELS),
            projection_dim = PROJECTION_DIM
        ).to(device) 
    
    sim_clr.load_state_dict(torch.load(CHECKPOINT_PATH))
    
    # CHECKPOINT_FILENAME = 'best.pt'

    optim = torch.optim.AdamW(sim_clr.parameters(), 4e-5)
    
    # warmup_steps = 300
    # sched_func = lambda step: 300/(1e-4)*512**(-0.5)*min((step+1)**(-2), step*warmup_steps**(-2.5))
    # scheduler = sched.LambdaLR(optim, lr_lambda=sched_func)
    scheduler = sched.MultiStepLR(optim, milestones=[3], gamma=0.7)
    
    print(f'----Train seed = {seedi} ----')
    train(
        module = sim_clr,
        train_loader = TrainLoader(FeaturesLoader(TRAINSET_PATH, train_meta_info, device), batch_size = BATCH_SIZE),
        val_loader = TestLoader(FeaturesLoader(TRAINSET_PATH, validation_meta_info, device), batch_size = BATCH_SIZE),
        valset_meta = validation_meta_info,
        optimizer = optim,
        criterion = NT_Xent(temperature = TEMPERATURE),
        num_epochs = NUM_EPOCHS,
        checkpoint_path = CHECKPOINT_PATH,
        #temperature_list = [0.005, 0.005, 0.005,
        #                  0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        temperature_list = [0.04, 0.04, 0.04, 0.04],
        scheduler = scheduler
        #temperature_list = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        #temperature_list = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    )

#%%

print("Validation")
val_score(
    module = sim_clr,
    val_loader = TestLoader(FeaturesLoader(TRAINSET_PATH, validation_meta_info, device), batch_size = BATCH_SIZE),
    valset_meta = validation_meta_info)

#%%
seed_everyting(42)
sim_clr = SimCLR(
    encoder = BasicNet(N_CHANNELS),
    projection_dim = PROJECTION_DIM
).to(device)

torch.cuda.empty_cache()
gc.collect()

sim_clr.load_state_dict(torch.load(CHECKPOINT_PATH))

#%%


print("Submission")
test_loader = TestLoader(FeaturesLoader(TESTSET_PATH, test_meta_info, device), batch_size = BATCH_SIZE)
model = sim_clr.encoder
model.eval()
embeds = inference(model, test_loader)
submission = get_ranked_list(embeds, 100)
save_submission(submission, SUBMISSION_PATH)
torch.save(sim_clr.state_dict(), MODEL_PATH)

#%%

save_submission(submission, "D:\Timur\Google Drive\Science\ML\Contests\Yandex audio artist prediction\submission.txt")

#%%