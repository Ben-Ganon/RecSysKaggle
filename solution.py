

# In[1]:
# print_file("\n ---IMPORTING--- \n")

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
import torch.nn.init as init
import argparse
import os
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--frac", type=float, default=1/1000)
parser.add_argument("--status_file", type=str, default="status_debug.txt")
args = parser.parse_args()

with open(args.status_file, 'w') as f:
    f.write("\n")

def print_file(text):
    with open(args.status_file, 'a') as f:
        f.write(text + '\n')

# In[2]:
print_file("\n ---STARTING--- \n")

EMBED_DIM = 512
DATA_FRAC = 1 / 1000

preprocess_prompt = """
    The following data is a review by a user of an anonymous hotel/accomodation.
    The review title is: {title}.
    The user wrote this positive review portion: {positive}.
    The user wrote this negative review portion: {negative}.
    The overall Score the user gave is: {score}.
    When published, this review garnered {review} 'helpful' notes.
    
    """
preprocess_prompt_no_score_votes = """
    Review title is: {title}.
    Positive review portion: {positive}.
    Negative review portion: {negative}.
"""

# In[18]:


def format_review(user_review):
    return preprocess_prompt.format(title=user_review['review_title'], positive=user_review['review_positive'],
                                                    negative=user_review['review_negative'], score=user_review['review_score'],
                                                    review=user_review['review_helpful_votes'])

def format_review_no_score_votes(user_review):
    return preprocess_prompt_no_score_votes.format(title=user_review['review_title'], positive=user_review['review_positive'],
                                                    negative=user_review['review_negative'])

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
    mini_batch_size = 100
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


embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# In[47]:


# F.normalize(embed_llm(['hey, im ben', 'hey, im jeff']), dim=1).norm()


# In[51]:


# embed_llm(['hey, im ben', 'hey, im jeff'])


# In[8]:


def load_dataset(part='train', frac=None):
    if part=='train':
        users = pd.read_csv("Data/train_users.csv")
        reviews = pd.read_csv("Data/train_reviews.csv")
        matches = pd.read_csv("Data/train_matches.csv")
    elif part=='val':
        users = pd.read_csv("Data/val_users.csv")
        reviews = pd.read_csv("Data/val_reviews.csv")
        matches = pd.read_csv("Data/val_matches.csv")
    if frac:
        # sample the data, and make sure to take users that match the sampled reviews
        reviews = reviews.sample(frac=frac)
        matches = matches[matches['review_id'].isin(reviews['review_id'])]
        users = users[users['user_id'].isin(matches['user_id'])]
    return users, reviews, matches


# In[9]:
print_file("Loading Data")

users_train, reviews_train, matches_train = load_dataset('train', frac=args.frac)
users_val, reviews_val, matches_val = load_dataset('val', frac=args.frac)
reviews_train = reviews_train.fillna("None Given")
reviews_val = reviews_val.fillna("None Given")


# In[10]:


def check_diff(feature):
    unique_train = set(users_train[feature])
    unique_val = set(users_val[feature])
    print_file(len(set(users_train[feature]) | set(users_val[feature])))
    print_file(len(users_train[~users_train[feature].isin(unique_val)]))
    print_file(users_train[~users_train[feature].isin(unique_val)][feature].value_counts())


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
    columns_to_avg_std = ['accommodation_star_rating', 'accommodation_score', 'room_nights']
    columns_to_count = ['guest_country', 'guest_type', 'accommodation_country']
    for col in columns_to_avg_std:
        average = users.groupby('accommodation_id')[col].mean()
        users[f'average_{col}'] = users['accommodation_id'].map(average)
    for col in columns_to_count:
        count = users[col].value_counts(normalize=True)
        users[f'count_{col}'] = users[col].map(count)
    average_month = users['month'].mean()
    users['average_month'] = average_month
    onehot_encoder = OneHotEncoder(sparse_output=False)
    binary_encoded = binary_encoder.transform(users[target_cols])
    # replace the target columns with the binary encoded columns
    encoded = pd.concat([users, binary_encoded], axis=1)
    encoded = encoded.drop(target_cols, axis=1)
    onehot_encoded = onehot_encoder.fit_transform(encoded[['guest_type']])
    encoded = pd.concat([encoded, onehot_encoded], axis=1)
    encoded = encoded.drop(['guest_type'], axis=1)
    features_to_normalize = ['accommodation_score', 'month', 'room_nights', 'accommodation_star_rating', 'average_month'
                             ] + [f'average_{col}' for col in columns_to_avg_std]
    normalized_values = MinMaxScaler().fit_transform(users[features_to_normalize])
    encoded = encoded.drop(features_to_normalize, axis=1)
    encoded = pd.concat([encoded, normalized_values], axis=1)
    return encoded


def preprocess_reviews(reviews):
    reviews['review_format'] = reviews.apply(format_review_no_score_votes, axis=1)
    reviews['review_embed'] = reviews['review_format'].apply(lambda x: F.normalize(embedding_model.encode(x,
                                                                                                          convert_to_tensor=True), dim=-1))
    features_to_normalize = ['review_score', 'review_helpful_votes']
    normalized_values = MinMaxScaler().fit_transform(reviews[features_to_normalize])
    reviews = reviews.drop(features_to_normalize, axis=1)
    reviews = pd.concat([reviews, normalized_values], axis=1)
    # concat review_embed and average and std columns
    reviews['review_embed'] = reviews.apply(lambda x: torch.cat([x['review_embed'].cpu(),
                                                                     torch.tensor(x['review_score']).unsqueeze(0).cpu(),
                                                                    torch.tensor(x[
                                                                                     'review_helpful_votes']).unsqueeze(0).cpu(),
                                                                    ], dim=-1), axis=1)
    return reviews
# In[15]:

print_file("Prepping Data")
# print_file(users_train[['guest_type']])
prep_users_train = preprocess_users(users_train)
prep_users_val = preprocess_users(users_val)

if os.path.exists("Data/prep_users_train.pkl"):
    prep_reviews_train = pd.read_pickle("Data/prep_reviews_train.pkl")
    prep_reviews_val = pd.read_pickle("Data/prep_reviews_val.pkl")

else:
    prep_reviews_train = preprocess_reviews(reviews_train)
    prep_reviews_val = preprocess_reviews(reviews_val)

    if not os.path.exists("Data/prep_users_train.pkl"):
        prep_reviews_train.to_pickle("Data/prep_reviews_train.pkl")
        prep_reviews_val.to_pickle("Data/prep_reviews_val.pkl")

OUTPUT_DIM = prep_reviews_train['review_embed'].iloc[0].shape[0]
EPOCHS = 5
INPUT_DIM = prep_users_train.shape[1] - 2
D_MODEL = 512
MLP_1 = 128
MLP_2 = 256
LR = 0.0001
BATCH_SIZE = 16
GRAD_CLIP = 0.1
PRINT_EVERY = 100
SAVE_EVERY = 10000
MODEL_PATH = "Models/Model_Cosine"

# In[64]:


class LatentSpaceEmbedder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=D_MODEL, mlp_1=MLP_1, mlp_2=MLP_2,
                 output_dim=OUTPUT_DIM, dropout=0.1):
        super(LatentSpaceEmbedder, self).__init__()
        
        # Normalization layer
        self.normalization = nn.BatchNorm1d(input_dim, dtype=torch.get_default_dtype())
        self.dropout = nn.Dropout(dropout)
        # Fully connected layers with dropout and batch normalization
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_1),
            nn.BatchNorm1d(mlp_1, dtype=torch.get_default_dtype()),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_1, mlp_2),
            nn.BatchNorm1d(mlp_2, dtype=torch.get_default_dtype()),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_2, output_dim),
            nn.BatchNorm1d(output_dim, dtype=torch.get_default_dtype()),
        )

        # Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Apply Xavier initialization to all Linear layers in the model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # Uniform Xavier initialization
                if m.bias is not None:
                    m.bias.data.fill_(0)  # Initialize biases to zero


    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        """
        # Normalize the input
        x = self.normalization(x)
        
        # Pass through MLP to generate embeddings
        output = self.mlp(x)
        return output


# In[17]:




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
        user_cols = self.users.columns.tolist()
        user = sample[user_cols]
        # user_review = sample[['review_title', 'review_positive', 'review_negative', 'review_score', 'review_helpful_votes']]
        # formatted_review = format_review(user_review=user_review)
        # get review tensor from series
        user_review = sample['review_embed']
        user = torch.tensor(user.values.astype(np.float32), dtype=torch.get_default_dtype()).detach()
        target = 1
        return user, user_review, target


class NegativeSampleDataset(Dataset):
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
        user_cols = self.users.columns.tolist()
        user = sample[user_cols]
        random_review = self.reviews.sample(n=1)
        # formatted_review = format_review(random_review)
        # get value from series
        review_embed = random_review['review_embed'].values[0]
        user = torch.tensor(user.values.astype(np.float32), dtype=torch.get_default_dtype()).detach()
        target = -1
        return user, review_embed, target


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
        user_cols =  self.users.columns.tolist()
        user = sample[user_cols]
        user = torch.tensor(user.values.astype(np.float32), dtype=torch.get_default_dtype()).detach()
        return user, sample['accommodation_id_x'], sample['review_id']


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (target) * euclidean_distance.pow(2)
        neg = (1-target) * torch.clamp(self.margin - euclidean_distance, min=0.0).pow(2)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive


# In[62]:




# In[24]:


def del_all(model, dataset, dataloader):
    if model:
        del(model)
    if dataset:
        del(dataset)
    if dataloader:
        del(dataloader)


# In[35]:


def eval_model(model, user_val, review_val, match_val, frac=1):
    model.eval()
    dataset = EvalDataset(user_val, review_val, match_val)
    dataset.sample(frac=frac)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, generator=torch.Generator(device='cuda'))
    total_rr = 0
    num_samples = 0
    print_file("Starting model eval")
    with torch.no_grad():
        similarity = torch.nn.CosineSimilarity()
        # similarity = F.pairwise_distance
        for batch in dataloader:
            user, accommodation_id, actual_review_id = batch
            possible_user_reviews = review_val[review_val['accommodation_id'] == accommodation_id.item()]
            possible_user_reviews_reset = possible_user_reviews.reset_index(drop=True)
            actual_review_id = actual_review_id[0]
            actual_review_index = possible_user_reviews_reset[possible_user_reviews_reset['review_id'] == actual_review_id].index[0]
            possible_user_reviews_embed = torch.stack(possible_user_reviews['review_embed'].to_list()).cpu().type(
                torch.float32)
            # possible_user_reviews = possible_user_reviews.apply(format_review, axis=1).to_list()
            # embedded_possible_user_reviews = embed_llm(possible_user_reviews)
            embedded_user = model(user).cpu().type(torch.float32)
            similarity_vector = similarity(embedded_user, possible_user_reviews_embed)
            k = min(10, len(possible_user_reviews))
            topk_values, topk_indices = torch.topk(similarity_vector, k=k, largest=True)
           
            if actual_review_index in topk_indices:
                review_rank = (topk_indices == actual_review_index).nonzero(as_tuple=True)[0].item() + 1
                rr = 1 / (review_rank)
                total_rr += rr
            num_samples += 1
                

    print_file(f"Eval Average MRR@10: {total_rr/num_samples}")


def save_model(model, optimizer, loss, i):
    model_path = MODEL_PATH + f"_epoch_{i}"
    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'd_model': D_MODEL,
        'lr': LR,
        'batch_size': BATCH_SIZE
    }, model_path)

# In[63]:


def train_embedder(model, dataloader, user_val, review_val, match_val, epochs, lr):
    # criterion = ContrastiveLoss(margin=3)
    # criterion = nn.MSELoss()
    criterion = nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    eval_model(model, user_val, review_val, match_val, frac=1)
    print_file("Starting model training")
    for i in range(epochs):
        total_loss = 0
        num_samples = 0
        loss = 0
        model.train()
        for iter, batch in enumerate(dataloader):
            if iter % SAVE_EVERY == SAVE_EVERY -1:
                save_model(model, optimizer, loss, i)
              # Set model to training mode
            # with autograd.detect_anomaly():
            optimizer.zero_grad()  # Reset gradients
            users, reviews, targets = batch
            # embedded_reviews = embed_llm(reviews)
            reviews = reviews.to('cuda')
            embedded_users = model(users)
            # print_file(f'shapes: users: {embedded_users.shape} \n reviews: {embedded_reviews.shape} \n targets: {targets.shape}')
            loss = criterion(embedded_users, reviews, target=targets)
            if torch.isnan(loss):
                print_file("nan loss")
                exit()
            # Backpropagation
            # loss.requires_grad = True
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_iter = loss.item()
            
            total_loss += loss_iter
            num_samples += 1
            if iter % PRINT_EVERY == 0:
                print_file(f'Average Loss at Iter {iter}: {total_loss/ num_samples}')
        
        print_file(f'{i} Epoch Loss Avg: {total_loss / num_samples}')
        save_model(model, optimizer, loss, i)
        eval_model(model, user_val, review_val, match_val)
    # Save the model
    model_path = f"Models/model_A_final"
    torch.save(model.state_dict(), model_path)


# In[58]:


model = LatentSpaceEmbedder()


# In[32]:

dataset = UserReviewDataset(prep_users_train, prep_reviews_train, matches_train)

negative_sampledataset = NegativeSampleDataset(prep_users_train, prep_reviews_train, matches_train)

concatenated_dataset = torch.utils.data.ConcatDataset([dataset, negative_sampledataset])



dataloader = DataLoader(dataset=concatenated_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device='cuda'))

# In[ ]:
# del_all(model, dataset, dataloader)
train_embedder(model, dataloader, prep_users_val, prep_reviews_val, matches_val, EPOCHS, LR)


# In[ ]:




