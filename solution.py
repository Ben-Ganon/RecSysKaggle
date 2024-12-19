#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np
from torch import autograd
import gc
from sentence_transformers import SentenceTransformer
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')


# In[2]:


embedding_dimensions = 512


# In[3]:


# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
# tokenizer.pad_token = tokenizer.eos_token
# llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", torch_dtype=torch.float32)
# llm_model.eval()
# def embed_batch(input_texts):
#      with torch.no_grad():
#             tokenized = tokenizer(input_texts, return_tensors='pt', padding=True)
#             outputs = llm_model(**tokenized, output_hidden_states=True)
#             outputs = outputs.hidden_states[-1][:, -1, :]
#             embed_list = torch.nn.functional.normalize(outputs, p=2.0, dim = -1).detach()
#             del outputs, tokenized
#      return embed_list


# In[4]:


def embed_batch(input_texts):
    embeddings = embedding_model.encode(input_texts)
    embeddings = torch.tensor(embeddings)
    return embeddings


# In[49]:


def embed_llm(input_texts):
    mini_batch_size = 50
    len_input = len(input_texts)
    if len_input > mini_batch_size:
        embed_list = []
        for i in range(0, len_input, mini_batch_size):
            j = min(i + mini_batch_size, len_input) -1 
            current_texts = input_texts[i:j]
            if not current_texts:
                continue
            embedded_current_texts = embed_batch(current_texts)
            embed_list.extend(embedded_current_texts)
        embeding_tensor = torch.stack(embed_list)
    else:
       embeding_tensor = embed_batch(input_texts)
    embeding_tensor = F.normalize(embeding_tensor, dim=1)
    return embeding_tensor


# In[6]:


embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=embedding_dimensions)


# In[47]:


# F.normalize(embed_llm(['hey, im ben', 'hey, im jeff']), dim=1).norm()


# In[51]:


# embed_llm(['hey, im ben', 'hey, im jeff'])


# In[8]:


def load_dataset(part='train', num=0):
    if part=='train':
        users = pd.read_csv("Data/train_users.csv")
        reviews = pd.read_csv("Data/train_reviews.csv")
        matches = pd.read_csv("Data/train_matches.csv")
    elif part=='val':
        users = pd.read_csv("Data/val_users.csv")
        reviews = pd.read_csv("Data/val_reviews.csv")
        matches = pd.read_csv("Data/val_matches.csv")
    if num > 0:
        users = users.sample(n=num)
    return users, reviews, matches


# In[9]:
print("Loading Data")

users_train, reviews_train, matches_train = load_dataset('train')
users_val, reviews_val, matches_val = load_dataset('val')
reviews_train = reviews_train.fillna("None Given")
reviews_val = reviews_val.fillna("None Given")


# In[10]:


def check_diff(feature):
    unique_train = set(users_train[feature])
    unique_val = set(users_val[feature])
    print(len(set(users_train[feature]) | set(users_val[feature])))
    print(len(users_train[~users_train[feature].isin(unique_val)]))
    print(users_train[~users_train[feature].isin(unique_val)][feature].value_counts())


# In[12]:


combined_guest = pd.concat([users_train['guest_country'], users_val['guest_country']]).unique()
combined_country = pd.concat([users_train['accommodation_country'], users_val['accommodation_country']]).unique()
combined_type = pd.concat([users_train['accommodation_type'], users_val['accommodation_type']]).unique()


# In[13]:


target_cols = ['guest_country', 'accommodation_country', 'accommodation_type']
target_arrs = [combined_guest, combined_country, combined_type]
combined_data = {}
for col in target_cols:
    # Get unique values from both A and B
    unique_values = pd.concat([users_train[col], users_val[col]]).drop_duplicates().reset_index(drop=True)
    combined_data[col] = unique_values

# Create a padded DataFrame for fitting
max_length = max(len(combined_data[col]) for col in target_cols)
aligned_data = pd.DataFrame({
    col: combined_data[col].reindex(range(max_length)) for col in target_cols
})

# Step 3: Initialize and fit BinaryEncoder on aligned combined data
binary_encoder = ce.BinaryEncoder(cols=target_cols, return_df=True)
binary_encoder.fit(aligned_data)


# In[14]:


def preprocess_users(users):
    onehot_encoder = OneHotEncoder(sparse_output=False)
    binary_encoded = binary_encoder.transform(users[target_cols])
    encoded = pd.concat([users, binary_encoded], axis=1)
    encoded = encoded.drop(target_cols, axis=1)
    onehot_encoded = onehot_encoder.fit_transform(encoded[['guest_type']])
    encoded = pd.concat([encoded, onehot_encoded], axis=1)
    encoded = encoded.drop(['guest_type'], axis=1)
    normalized_encoded = MinMaxScaler().fit_transform(encoded[encoded.columns.difference(['user_id'])])
    normalized_encoded = pd.concat([encoded['user_id'], normalized_encoded])
    return encoded


# In[15]:

print("Prepping Data")
# print(users_train[['guest_type']])
prep_users_train = preprocess_users(users_train)
prep_users_val = preprocess_users(users_val)



# In[64]:


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, ff_dim, input_dim=33, embedding_dim=embedding_dimensions, dropout=0.1):
        """
        Args:
            input_dim (int): Size of the flattened input.
            embedding_dim (int): Dimension of the output embedding space.
            d_model (int): Dimension of the Transformer model (hidden size).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            ff_dim (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(TransformerEncoder, self).__init__()
        
        # Input projection: Project input into d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        # Positional encoding (to give position information to the Transformer)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            # activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,)
        self.encoder_bn = nn.BatchNorm1d(d_model)
        # Final projection to embedding space
        self.output_proj = nn.Linear(d_model, embedding_dim)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
        Returns:
            torch.Tensor: Embedded output of shape [batch_size, embedding_dim].
        """
        # Project input to d_model
        x = self.input_proj(x)  # Shape: [batch_size, d_model]
        x = self.input_bn(x)
        # Add positional encoding (sequence length = 1 since input is flattened)
        x = self.positional_encoding(x.unsqueeze(1))  # Add sequence dimension: [batch_size, seq_len=1, d_model]
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: [batch_size, seq_len=1, d_model]
        
        # Remove sequence dimension and project to embedding space
        x = x.squeeze(1)  # Shape: [batch_size, d_model]
        x = self.encoder_bn(x)
        x = self.output_proj(x)  # Shape: [batch_size, embedding_dim]
        x = x.squeeze(0)
        return x

class PositionalEncoding(nn.Module):
    """Positional Encoding Module to add positional information to the Transformer inputs."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # Add positional encoding
        return self.dropout(x)


# In[17]:


preprocess_prompt = """
    The following data is a review by a user of an anonymous hotel/accomodation.
    The review title is: {title}.
    The user wrote this positive review portion: {positive}.
    The user wrote this negative review portion: {negative}.
    The overall Score the user gave is: {score}.
    When published, this review garnered {review} 'helpful' notes.
    
    """


# In[18]:


def format_review(user_review):
    return preprocess_prompt.format(title=user_review['review_title'], positive=user_review['review_positive'],
                                                    negative=user_review['review_negative'], score=user_review['review_score'],
                                                    review=user_review['review_helpful_votes'])


# In[19]:


def merge_data(users, reviews, matches):
    merged_m_r = pd.merge(matches, reviews, on=['review_id', 'accommodation_id'], how='inner')
    merged_u_m_r = pd.merge(merged_m_r, users, on='user_id', how='inner')
    # merged_u_m_r = merged_u_m_r.rename(columns={"accommodation_id_x": "accommodation_id"})
    return merged_u_m_r


# In[20]:


def find_users_review(user, reviews, matches):
    user_id = user['user_id']
    user_accommodation = user['accommodation_id']
    user = user.drop(['user_id', 'accommodation_id'])
    user_matches = matches[(matches['user_id'] == user_id) & (matches['accommodation_id'] == user_accommodation)]
    user_match_review = user_matches['review_id'].values[0]
    user_review = reviews[reviews['review_id'] == user_match_review]
    return user_review


# In[21]:


class UserReviewDataset(Dataset):
    def __init__(self, users, reviews, matches):
          # User data
        self.matches = matches
        self.reviews = reviews
        self.merged = merge_data(users, reviews, matches)
        self.users = users.drop(columns=['user_id', 'accommodation_id'])
    def __len__(self):
        return len(self.merged)
        
    def sample(self, frac):
        self.merged = self.merged.sample(frac=frac)
    
    def __getitem__(self, idx):
        sample = self.merged.iloc[idx]
        user_cols = ['accommodation_id_x'] + self.users.columns.tolist()
        user = sample[user_cols]
        user_review = sample[['review_title', 'review_positive', 'review_negative', 'review_score', 'review_helpful_votes']]
        formatted_review = format_review(user_review=user_review)
        user = torch.tensor(user.values.astype(np.float32)).detach()
        return user, formatted_review


# In[22]:


class EvalDataset(Dataset):
    def __init__(self, users, reviews, matches):
          # User data
        self.matches = matches
        self.reviews = reviews
        self.merged = merge_data(users, reviews, matches)
        self.users = users.drop(columns=['user_id', 'accommodation_id'])
    def __len__(self):
        return len(self.merged)
        
    def sample(self, frac):
        self.merged = self.merged.sample(frac=frac)
    
    def __getitem__(self, idx):
        sample = self.merged.iloc[idx]
        user_cols = ['accommodation_id_x'] + self.users.columns.tolist()
        user = sample[user_cols]
        user = torch.tensor(user.values.astype(np.float32)).detach()
        return user, sample['accommodation_id_x'], sample['review_id']


# In[62]:


EPOCHS = 5
D_MODEL=1024
NUM_HEADS=8
NUM_LAYERS=2
FF_DIM=1024
LR = 0.001
BATCH_SIZE = 32
GRAD_CLIP = 1.
SAVE_EVERY = 10000
PRINT_EVERY = 10
MODEL_PATH = "Models/model_B"

# In[24]:


def del_all(model, dataset, dataloader):
    if model:
        del(model)
    if dataset:
        del(dataset)
    if dataloader:
        del(dataloader)


# In[35]:


def eval_model(model, user_val, review_val, match_val):
    model.eval()
    dataset = EvalDataset(user_val, review_val, match_val)
    dataset.sample(frac=1/1000)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, generator=torch.Generator(device='cuda'))
    total_mrr = 0
    num_samples = 0
    print("Starting model eval")
    with torch.no_grad():
        similarity = torch.nn.CosineSimilarity()
        for batch in dataloader:
            user, accommodation_id, actual_review_id = batch
            possible_user_reviews = review_val[review_val['accommodation_id'] == accommodation_id.item()]
            possible_user_reviews_reset = possible_user_reviews.reset_index(drop=True)
            actual_review_id = actual_review_id[0]
            actual_review_index = possible_user_reviews_reset[possible_user_reviews_reset['review_id'] == actual_review_id].index[0]
            possible_user_reviews = possible_user_reviews.apply(format_review, axis=1).to_list()
            embedded_possible_user_reviews = embed_llm(possible_user_reviews)
            embedded_user = model(user)
            similarity_vector = similarity(embedded_user, embedded_possible_user_reviews)
            topk_values, topk_indices = torch.topk(similarity_vector, k=10)
           
            if actual_review_index not in topk_indices:
                num_samples += 1
            else:
                review_rank = (topk_indices == actual_review_index).nonzero(as_tuple=True)[0].item() + 1
                mrr = 1 / (review_rank)
                num_samples += 1
                total_mrr += mrr

    print(f"Eval Average MRR@10: {total_mrr/num_samples}")


def save_model(model, optimizer, loss, i):
    model_path = MODEL_PATH + f"_epoch_{i}"
    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'd_model': D_MODEL,
        'num_heads': NUM_HEADS,
        'num_layers':NUM_LAYERS,
        'lr': LR,
        'ff_dim': FF_DIM,
        'batch_size': BATCH_SIZE
    }, model_path)

# In[63]:


def train_embedder(model, dataloader, user_val, review_val, match_val, epochs, lr):
    model = model

    criterion = nn.CosineEmbeddingLoss()
    # criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    print("Starting model training")
    for i in range(epochs):
        total_loss = 0
        num_samples = 0
        loss = 0
        for iter, batch in enumerate(dataloader):
              # Set model to training mode
            users, reviews = batch
            if iter % SAVE_EVERY == SAVE_EVERY -1:
                save_model(model, optimizer, loss, i)
                eval_model(model, user_val, review_val, match_val)
                model.train()
            # with autograd.detect_anomaly():
            optimizer.zero_grad()  # Reset gradients
            users, reviews = batch
            embedded_reviews = embed_llm(reviews)
            
            embedded_users = model(users)
            targets = torch.ones((embedded_reviews.shape[0]))
            # print(f'shapes: users: {embedded_users.shape} \n reviews: {embedded_reviews.shape} \n targets: {targets.shape}')
            loss = criterion(embedded_users, embedded_reviews, target=targets)
            if torch.isnan(loss):
                print("nan loss")
                exit()
            # Backpropagation
            # loss.requires_grad = True
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_iter = loss.item()
            
            del users, embedded_users, embedded_reviews, reviews, targets
            
            gc.collect()

            
            total_loss += loss_iter
            num_samples += BATCH_SIZE
            if i % PRINT_EVERY == 0:
                print(f'Loss at iter {iter}: {loss_iter}')
                
        print(f'{i} Epoch Loss Avg: {total_loss / num_samples}')
        model_path = f"Models/model_B_epoch-{i}"
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)

    # Save the model
    model_path = f"Models/model_A_final"
    torch.save(model.state_dict(), model_path)


# In[58]:


model = TransformerEncoder(d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, ff_dim=FF_DIM)


# In[32]:


dataset = UserReviewDataset(prep_users_train, reviews_train, matches_train)


# In[33]:


dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device='cuda'))


# In[ ]:


# del_all(model, dataset, dataloader)
train_embedder(model, dataloader, prep_users_val, reviews_val, matches_val, EPOCHS, LR) 


# In[ ]:




