# FT-Transformer: https://github.com/lucidrains/tab-transformer-pytorch
import pandas as pd
import numpy as np
import torch
from torch import nn, einsum
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from einops import rearrange, repeat
from tqdm import tqdm
import random

# dataset
class TrainDataset(Dataset):
    def __init__(self, data, scaler=None):
        self.data = data
        self.X = self.data.drop(columns=['deposit']).values
        self.y = self.data['deposit'].values
        self.scaler = scaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
    
    
class TestDataset(Dataset):
    def __init__(self, data, scaler):
        self.data = data
        self.X = self.data.values
        self.scaler = scaler
        self.X = self.scaler.transform(self.X)
    
    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)    
    
    
# activation function
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# feed forward layer
def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# attention layer
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder
class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# ft-transformer model
class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical
        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend
        x, attns = self.transformer(x, return_attn = True)

        # get cls token
        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))
        logits = self.to_logits(x)

        if not return_attn:
            return logits
        return logits, attns    

# train
def train_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs, patience):
    best_val_loss = float('inf')
    counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    for epoch in range(n_epochs):
        train_losses = []
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            input_data, target = data
            input_data = input_data.to(device)
            target = target.to(device)
            input_data.to(device)
            output = model(input_data[:, :2].type(torch.int64), input_data[:, 2:])
            loss = criterion(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        val_loss, _ = predict(model, val_loader, criterion, device)
        print(f'Epoch {epoch}, Train Loss: {sum(train_losses) / len(train_losses)}, Val Loss: {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f'./checkpoint/FT_transformer.pth')
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping')
                break
        scheduler.step(val_loss)
    
    return model

# predict
def predict(model, data_loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input_data, target = data
            input_data = input_data.to(device)
            target = target.to(device)
            output = model(input_data[:, :2].type(torch.int64), input_data[:, 2:])
            # MAE Loss
            loss = criterion(output, target.view(-1, 1))
            losses.append(loss.item())
            preds.append(output.cpu().numpy())

    loss = sum(losses) / len(losses)
    return loss, np.concatenate(preds).flatten()    


def set_seed(val):
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.cuda.manual_seed_all(val)
    np.random.seed(val)
    random.seed(val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    merged_data = pd.read_csv('../data/real_final_df.csv')
    sample_submission = pd.read_csv('../data/sample_submission.csv')

    merged_data['contract_date'] = pd.to_datetime(merged_data['contract_year_month'].astype(str) + merged_data['contract_day'].astype(str), format='%Y%m%d')
    merged_data['contract_date'] = (merged_data['contract_date'] - pd.Timestamp('2019-01-01')).dt.days

    train_data = merged_data[merged_data['_type'] == 'train'].drop(columns = ['_type'])
    test_data = merged_data[merged_data['_type'] == 'test'].drop(columns = ['_type'])
    
    columns = [
        # category
        'cluster_kmeans', 'contract_type',
        # base
        'latitude', 'longitude', 'contract_date', 'deposit_mean', 'area_m2', 'floor', 'age', 'distance_from_gangnam',
        # subway
        'nearest_subway_distance_km', 'subways_within_500m', 'subways_within_1km', 'subways_within_2km',
        # school
        'school_count_within_1km', 'closest_elementary_distance', 'closest_middle_distance', 'closest_high_distance',
        # others
        'apt_deposit_mean', 'last_deposit_by_area', 'interest_rate', 
        # park
        'nearest_park_distance', 'park_count_500m','total_park_area_500m', 'park_count_1000m', 'total_park_area_1000m',
        'park_count_2000m', 'total_park_area_2000m', 'weighted_park_score','avg_distance_5_parks',    
    ]

    train_data = train_data[columns + ['deposit']]
    test_data = test_data[columns]
    train_data = train_data.fillna(-999)
    test_data = test_data.fillna(-999)
    
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_dataset = TrainDataset(train_data)
    val_dataset = TrainDataset(val_data, scaler=train_dataset.scaler)
    test_dataset = TestDataset(test_data, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    model = FTTransformer(categories= (5, 3), num_continuous=14, dim=32, depth=6, heads=12, ff_dropout=0.2, attn_dropout=0.2).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    model.train()
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, 150, 6)  
    test_preds = predict(model, test_loader, device)  
    
    sample_submission['deposit'] = test_preds
    sample_submission.to_csv('ft_transformer_submission.csv', index=False)