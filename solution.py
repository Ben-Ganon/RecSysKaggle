

# In[1]:
# print_file("\n ---IMPORTING--- \n")
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np
from torch import autograd
from sentence_transformers import SentenceTransformer
import torch.nn.init as init
import argparse
import os
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--frac", type=float, default=None)
parser.add_argument("--status_file", type=str, default="status_debug.txt")
parser.add_argument("--num_groups", type=int, default=None)
args = parser.parse_args()

with open(args.status_file, 'w') as f:
    f.write("\n")

def print_file(text):
    with open(args.status_file, 'a') as f:
        f.write(text + '\n')
    print(text)

# In[2]:
print_file("\n ---STARTING--- \n")


preprocess_prompt = """
    The following data is a review by a user of an anonymous hotel/accomodation.
    The review title is: {title}.
    The user wrote this positive review portion: {positive}.
    The user wrote this negative review portion: {negative}.
    The overall Score the user gave is: {score}.
    When published, this review garnered {review} 'helpful' notes.
    
    """
preprocess_prompt_no_score_votes = """clustering: The following is a review on a website for some accommodation.
    The review title is: {title}.
    The positive review portion: {positive}.
    The negative review portion: {negative}.
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


EMBED_DIM = 368



def embed_batch(embedder, texts, batchsize=128):
    embeddings = []
    truncate_dim = EMBED_DIM
    for i in tqdm(range(0, len(texts), batchsize)):
        batch = texts[i:i+batchsize]
        batch_embeddings = embedder.encode(batch, convert_to_tensor=True)
        batch_embeddings = F.layer_norm(batch_embeddings, normalized_shape=(batch_embeddings.shape[1],))
        batch_embeddings = batch_embeddings[:, :truncate_dim]
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        embeddings.extend(batch_embeddings.to('cpu'))
    return embeddings


def load_dataset(part='train', frac=None):
    print_file("Loading Data")
    if part=='train':
        users = pd.read_csv("Data/train_users.csv")
        reviews = pd.read_csv("Data/train_reviews.csv")
        matches = pd.read_csv("Data/train_matches.csv")
    elif part=='val':
        users = pd.read_csv("Data/val_users.csv")
        reviews = pd.read_csv("Data/val_reviews.csv")
        matches = pd.read_csv("Data/val_matches.csv")
    elif part=='test':
        reviews = pd.read_csv("Data/test_reviews.csv")
        users = pd.read_csv("Data/test_users.csv")
        matches = None
    if frac:
        # sample the data, and make sure to take users that match the sampled reviews
        reviews = reviews.sample(frac=frac)
        if matches is not None:
            matches = matches[matches['review_id'].isin(reviews['review_id'])]
        users = users[users['user_id'].isin(matches['user_id'])]
    return users, reviews, matches



def preprocess_users(users_train, users_val, users_test, which='train'):
    target_cols = ['guest_country', 'accommodation_country', 'accommodation_type']
    # target_arrs = [combined_guest, combined_country, combined_type]
    combined_data = {}
    for col in target_cols:
        # Get unique values from both A and B
        unique_values = pd.concat([users_train[col], users_val[col], users_test[col]]).drop_duplicates().reset_index(drop=True)
        combined_data[col] = unique_values

    # Create a padded DataFrame for fitting
    max_length = max(len(combined_data[col]) for col in target_cols)
    aligned_data = pd.DataFrame({
        col: combined_data[col].reindex(range(max_length)) for col in target_cols
    })

    # Step 3: Initialize and fit BinaryEncoder on aligned combined data
    binary_encoder = ce.BinaryEncoder(cols=target_cols, return_df=True)
    binary_encoder.fit(aligned_data)

    if which == 'train':
        users = users_train
    elif which == 'val':
        users = users_val
    elif which == 'test':
        users = users_test

    columns_to_avg_std = ['accommodation_star_rating', 'accommodation_score', 'room_nights']
    columns_to_count = ['guest_country', 'guest_type', 'accommodation_country', ]

    for col in columns_to_avg_std:
        average = users.groupby('accommodation_id')[col].mean()
        users[f'average_{col}'] = users['accommodation_id'].map(average)
    for col in columns_to_count:
        count = users[col].value_counts(normalize=True)
        users[f'count_{col}'] = users[col].map(count)
    users['count_guest_country'].fillna(users['count_guest_country'].mode()[0], inplace=True)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    binary_encoded = binary_encoder.transform(users[target_cols])
    # replace the target columns with the binary encoded columns
    encoded = pd.concat([users, binary_encoded], axis=1)
    encoded = encoded.drop(target_cols, axis=1)
    onehot_encoded = onehot_encoder.fit_transform(encoded[['guest_type']])
    encoded = pd.concat([encoded, onehot_encoded], axis=1)
    encoded = encoded.drop(['guest_type'], axis=1)
    features_to_normalize = ['accommodation_score', 'month', 'room_nights', 'accommodation_star_rating'
                             ] + [f'average_{col}' for col in columns_to_avg_std]
    normalized_values = MinMaxScaler().fit_transform(users[features_to_normalize])
    encoded = encoded.drop(features_to_normalize, axis=1)
    encoded = pd.concat([encoded, normalized_values], axis=1)
    average_month = encoded['month'].mean()
    encoded['average_month'] = average_month
    return encoded


def preprocess_reviews(embedder, reviews):
    reviews['review_format'] = reviews.apply(format_review_no_score_votes, axis=1)
    reviews['review_embed'] = embed_batch(embedder, reviews['review_format'].tolist())
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


# In[64]:


class TwoTowerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, tower_1_dim, bridge_dim_1, bridge_dim_2, bridge_dim_3, dropout=0.1):
        super(TwoTowerModel, self).__init__()
        activation = nn.LeakyReLU
        # Normalization layer

        self.tower1 = nn.Sequential(
            nn.Linear(input_dim, tower_1_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(tower_1_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout)
        )

        self.tower2 = nn.Sequential(
            nn.Linear(input_dim, tower_1_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(tower_1_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
        )

        self.bridge_fc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim + hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + hidden_dim, bridge_dim_1),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim_1, bridge_dim_2),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim_2, bridge_dim_3),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim_3, 1),
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

    def forward(self, users, reviews):
        users = self.tower1(users)
        reviews = self.tower2(reviews)
        concat = torch.cat([users, reviews], dim=-1)
        pred = self.bridge_fc(concat)
        return pred


class UserEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_1, mlp_2, mlp_3, mlp_4, mlp_5,
                 output_dim, dropout):
        super(UserEmbedder, self).__init__()
        activation = nn.LeakyReLU
        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_1),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_1, mlp_2),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_2, mlp_3),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_3, mlp_4),
            activation(),
            nn.Dropout(dropout),
             nn.Linear(mlp_4, mlp_5),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_5, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
             nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
             nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
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
        # x = self.input_batch_norm(x)
        x = self.mlp(x)
        return x



class ReviewEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_1, mlp_2,
                 output_dim, dropout):
        super(ReviewEmbedder, self).__init__()
        activation = nn.LeakyReLU
        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_1),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_1, mlp_2),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_2, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
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
        # x = self.input_batch_norm(x)
        x = self.mlp(x)
        return x


def merge_data(users, reviews, matches):
    merged_m_r = pd.merge(matches, reviews, on=['review_id'], how='inner')
    merged_u_m_r = pd.merge(merged_m_r, users, on='user_id', how='inner')
    # merged_u_m_r = merged_u_m_r.rename(columns={"accommodation_id_x": "accommodation_id"})
    return merged_u_m_r



class EvalDataset(Dataset):
    def __init__(self, users, reviews, matches):
          # User data
        self.matches = matches
        self.reviews = reviews
        self.users = users.drop(columns=['user_id', 'accommodation_id'])
        self.merged = reviews
    def __len__(self):
        return len(self.merged)
        
    def sample(self, frac):
        self.merged = self.merged.sample(frac=frac)
    
    def __getitem__(self, idx):
        sample = self.merged.iloc[idx]
        user_cols = self.users.columns.tolist()
        user = sample[user_cols]
        user = torch.tensor(user.values.astype(np.float32), dtype=torch.get_default_dtype())
        possible_user_reviews = self.reviews[self.reviews['accommodation_id'] == sample['accommodation_id_x'].item()]
        possible_user_reviews_reset = possible_user_reviews.reset_index(drop=True)
        actual_review_id = sample['review_id']
        actual_review_index = possible_user_reviews_reset[possible_user_reviews_reset['review_id'] == actual_review_id].index[0]
        possible_user_reviews_embed = torch.stack(possible_user_reviews_reset['review_embed'].to_list()).cpu().type(
                torch.float32).squeeze()
        accomodation_id = sample['accommodation_id_x']
        return user, actual_review_index, possible_user_reviews_embed, accomodation_id, 0, 0


class TestDataset(Dataset):
    def __init__(self, users, reviews):
        self.users = users
        self.reviews = reviews
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        sample = self.users.iloc[idx]
        accom_id = sample['accommodation_id']
        user_id = sample['user_id']
        user_possible_reviews = self.reviews[self.reviews['accommodation_id'] == accom_id]
        
        sample = sample.drop(labels=['user_id', 'accommodation_id'])
        sample = torch.tensor(sample.values.astype(np.float32), dtype=torch.get_default_dtype()).detach()

        possible_user_reviews_embed = torch.stack(user_possible_reviews['review_embed'].to_list()).cpu().type(
                torch.float32)

        return sample, 0, possible_user_reviews_embed, accom_id, user_possible_reviews['review_id'].to_list(), user_id


class PreproccessReviewsDataset(Dataset):
    def __init__(self, reviews):
        self.reviews = reviews
    
    def __getitem__(self, idx): 
        return format_review_no_score_votes(self.reviews.iloc[idx])


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


def eval_model(model_users, model_reviews, dataloader, reps=-1, test=False, batchsize=1, tower=False):
    model_users.eval()
    model_reviews.eval()
    total_rr = 0
    num_samples = 0
    print_file("Starting model eval")
    largest = True
    if test:
        output_csv = pd.DataFrame(columns=['accommodation_id', 'user_id', 'review_1', 'review_2', 'review_3', 'review_4', 'review_5', 'review_6', 'review_7'
                                           , 'review_8', 'review_9', 'review_10'])
    output_list = []
    with torch.no_grad():
        i = 0
        for batch in dataloader:
            i += 1
            if i > reps != -1:
                break
            user, actual_review_index, possible_user_reviews_embed, accomodation_id, possible_reviews_ids, user_id = batch
            
            actual_review_index = actual_review_index.squeeze()
            
            if len(possible_user_reviews_embed.shape) > 2:
                possible_user_reviews_embed = possible_user_reviews_embed.squeeze()
            actual_review_index = actual_review_index.to('cuda')

            possible_user_reviews_embed = possible_user_reviews_embed.to('cuda')
            # user_dupe = user.repeat(possible_user_reviews_embed.shape[0], 1)
            user = user.to('cuda')
            embedded_user = model_users(user)
            embedded_reviews = model_reviews(possible_user_reviews_embed)
            similarity_vector = (embedded_user @ embedded_reviews.T).squeeze()
            
            k = min(10, possible_user_reviews_embed.shape[0])
            topk_values, topk_indices = torch.topk(similarity_vector, k=k, largest=largest)

            if not test:
                if actual_review_index in topk_indices:
                    review_rank = (topk_indices == actual_review_index).nonzero(as_tuple=True)[0].item() + 1
                    rr = 1 / (review_rank)
                    total_rr += rr
                num_samples += 1
            else:
                index_list = topk_indices.tolist()
                if len(index_list) < 10:
                    zero_index = [0] * (10 - len(index_list))
                    index_list = index_list + zero_index
                review_ids = [possible_reviews_ids[idx][0] for idx in index_list]
                output_line = [accomodation_id.item(), user_id[0]] + review_ids
                output_csv.loc[len(output_csv)] = output_line
        if test:
            if 'ID' not in output_csv.columns:
                output_csv.insert(0, 'ID', range(1, len(output_csv) + 1))
            output_csv = output_csv.reset_index(drop=True)
            output_csv.to_csv("test_match_new.csv", index=False)
    if not test:
        print_file(f"Eval Average MRR@10: {total_rr/num_samples}")

def eval_towers(model_users, model_reviews, model_towers, grouped_data, user_val, batchsize):
    model_users.eval()
    model_reviews.eval()
    model_towers.eval()

    user_columns = user_val.columns
    user_cols_to_remove = ['user_id', 'accommodation_id']
    user_columns = [col for col in user_columns if col not in user_cols_to_remove]

    mrr_scores = []
    print_file("Starting model eval")
    with torch.no_grad():
        for name, group in grouped_data:
            users = group[user_columns].values
            users_list = [torch.tensor(u.astype(np.float32), dtype=torch.get_default_dtype()) for u in users]

            reviews_tensor_list = [t for t in group['review_embed'].values]
            reviews = torch.stack(reviews_tensor_list).to('cuda')
            review_embeds = model_reviews(reviews)

            targets = torch.arange(len(users_list), dtype=torch.long, device='cuda')


            for user_index, user in enumerate(users_list):
                target = targets[user_index]
                user_embed = model_users(user)
                user_repeated = user_embed.repeat(review_embeds.shape[0], 1)
                predictions_user = model_towers(user_repeated, review_embeds).reshape(review_embeds.shape[0])
                sorted_predictions = torch.argsort(predictions_user, descending=True)
                correct_index_rank = (sorted_predictions == target).nonzero(as_tuple=True)[0]
                if torch.numel(correct_index_rank) > 0:
                    rank = correct_index_rank.item() + 1
                    # Compute reciprocal rank if within top_k
                    if rank <= 10:
                        mrr_scores.append(1 / rank)
                    else: mrr_scores.append(0)

                else: mrr_scores.append(0)

        mrr_avg = np.mean(mrr_scores)
        print_file(f'MRR@10: {mrr_avg}')
        return mrr_avg



def eval_new(model_users, model_reviews, grouped_data, user_val):
    model_users.eval()
    model_reviews.eval()

    user_columns = user_val.columns
    user_cols_to_remove = ['user_id', 'accommodation_id']
    user_columns = [col for col in user_columns if col not in user_cols_to_remove]

    mrr_scores = []
    print_file("Starting model eval")
    with torch.no_grad():
        for name, group in grouped_data:
            users = group[user_columns].values
            users_list = [torch.tensor(u.astype(np.float32), dtype=torch.get_default_dtype()) for u in users]
            users = torch.stack(users_list).to('cuda')

            reviews_tensor_list = [t for t in group['review_embed'].values]
            reviews = torch.stack(reviews_tensor_list).to('cuda')

            user_embeds = model_users(users)
            review_embeds = model_reviews(reviews)

            similarity_matrix = user_embeds @ review_embeds.T

            targets = torch.arange(len(user_embeds), dtype=torch.long, device='cuda')
            
            # Compute ranks and MRR@10
            for i, target in enumerate(targets):
                # Sort similarities for each user in descending order
                sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
                # Find the rank of the correct review
                predicted_index =  (sorted_indices == target).nonzero(as_tuple=True)[0]
                if torch.numel(predicted_index) > 0:
                    rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
                else: rank = 11
                # Compute reciprocal rank if within top_k
                if rank <= 10:
                    mrr_scores.append(1 / rank)
                else:
                    mrr_scores.append(0)
        mrr_avg = np.mean(mrr_scores)
        print_file(f'MRR@10: {mrr_avg}')
        return mrr_avg


def save_model(model, optimizer, i, model_path):
    model_path = model_path + f"_epoch_{i}.pt"
    # torch.save({
    #     'epoch': i,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, model_path)
    torch.save(model, model_path)


def write_test_output_towers(model_users, model_reviews, model_towers, dataloader):
    output_csv = pd.DataFrame(columns=['accommodation_id', 'user_id', 'review_1', 'review_2', 'review_3', 'review_4', 'review_5', 'review_6', 'review_7'
                                           , 'review_8', 'review_9', 'review_10'])
    output_list = []
    print_file("Starting Test Output")
    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader, position=0, leave=True):
            i += 1
            user, actual_review_index, possible_user_reviews_embed, accomodation_id, possible_reviews_ids, user_id = batch

            if len(possible_user_reviews_embed.shape) > 2:
                possible_user_reviews_embed = possible_user_reviews_embed.squeeze()
            possible_user_reviews_embed = possible_user_reviews_embed.to('cuda')
            user = user.to('cuda')
            embedded_user = model_users(user)
            embedded_reviews = model_reviews(possible_user_reviews_embed)
            repeated_user = embedded_user.repeat(possible_user_reviews_embed.shape[0], 1)
            predictions = model_towers(repeated_user, embedded_reviews).reshape(embedded_reviews.shape[0])

            k = min(10, possible_user_reviews_embed.shape[0])
            topk_values, topk_indices = torch.topk(predictions, k=k, largest=True)

            index_list = topk_indices.tolist()
            if len(index_list) < 10:
                zero_index = [0] * (10 - len(index_list))
                index_list = index_list + zero_index
            review_ids = [possible_reviews_ids[idx][0] for idx in index_list]
            output_line = [accomodation_id.item(), user_id[0]] + review_ids
            output_csv.loc[len(output_csv)] = output_line
        if 'ID' not in output_csv.columns:
            output_csv.insert(0, 'ID', range(1, len(output_csv) + 1))
        output_csv = output_csv.reset_index(drop=True)
        output_csv.to_csv("test_match_towers.csv", index=False)

def write_test_output(model_users, model_reviews, dataloader):
    output_csv = pd.DataFrame(columns=['accommodation_id', 'user_id', 'review_1', 'review_2', 'review_3', 'review_4', 'review_5', 'review_6', 'review_7'
                                           , 'review_8', 'review_9', 'review_10'])
    output_list = []
    print_file("Starting Test Output")
    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader, position=0, leave=True):
            i += 1
            user, actual_review_index, possible_user_reviews_embed, accomodation_id, possible_reviews_ids, user_id = batch

            if len(possible_user_reviews_embed.shape) > 2:
                possible_user_reviews_embed = possible_user_reviews_embed.squeeze()
            possible_user_reviews_embed = possible_user_reviews_embed.to('cuda')
            user = user.to('cuda')
            embedded_user = model_users(user)
            embedded_reviews = model_reviews(possible_user_reviews_embed)
            similarity_vector = (embedded_user @ embedded_reviews.T).squeeze()

            k = min(10, possible_user_reviews_embed.shape[0])
            topk_values, topk_indices = torch.topk(similarity_vector, k=k, largest=True)

            index_list = topk_indices.tolist()
            if len(index_list) < 10:
                zero_index = [0] * (10 - len(index_list))
                index_list = index_list + zero_index
            review_ids = [possible_reviews_ids[idx][0] for idx in index_list]
            output_line = [accomodation_id.item(), user_id[0]] + review_ids
            output_csv.loc[len(output_csv)] = output_line
        if 'ID' not in output_csv.columns:
            output_csv.insert(0, 'ID', range(1, len(output_csv) + 1))
        output_csv = output_csv.reset_index(drop=True)
        output_csv.to_csv("test_match_new.csv", index=False)

def test_towers(embedder):
    users_train, _, _ = load_dataset('train', frac=args.frac)
    users_val, _, _ = load_dataset('val', frac=args.frac)

    users_test, reviews_test, _ = load_dataset('test')

    if args.num_groups:
        users_test = users_test.sample(frac=0.001)

    prepped_users = preprocess_users(users_train, users_val, users_test, which='test')

    del users_train, users_val

    if args.num_groups:
        reviews_test = reviews_test[reviews_test['accommodation_id'].isin(prepped_users['accommodation_id'])]
        prepped_reviews_test = preprocess_reviews(embedder, reviews_test)

    else:

        if not os.path.exists("Data/test.pkl"):
            prepped_reviews_test = preprocess_reviews(embedder, reviews_test)
            prepped_reviews_test.to_pickle("Data/test.pkl")
        else:
            prepped_reviews_test = pd.read_pickle("Data/test.pkl")

    model_users = torch.load("Models/Model_Embed_X_users_epoch_10.pt")
    model_reviews = torch.load("Models/Model_Embed_X_reviews_epoch_10.pt")

    model_towers = torch.load("Models/Model_Towers_epoch_final.pt")
    model_users.eval()
    model_reviews.eval()
    model_towers.eval()

    dataset = TestDataset(prepped_users, prepped_reviews_test)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, generator=torch.Generator(device='cuda'))

    write_test_output_towers(model_users, model_reviews, model_towers, dataloader)

def test(embedder):
    users_train, _, _ = load_dataset('train', frac=args.frac)
    users_val, _, _ = load_dataset('val', frac=args.frac)

    users_test, reviews_test, _ = load_dataset('test')

    # users_test = users_test.sample(frac=0.001)
    # reviews_test = reviews_test[reviews_test['accommodation_id'].isin(users_test['accommodation_id'])]

    prepped_users = preprocess_users(users_train, users_val, users_test, which='test')

    del users_train, users_val

    if not os.path.exists("Data/test.pkl"):
        prepped_reviews_test = preprocess_reviews(embedder, reviews_test)
        prepped_reviews_test.to_pickle("Data/test.pkl")
    else:
        prepped_reviews_test = pd.read_pickle("Data/test.pkl")


    model_users = torch.load("Models/Model_Embed_X_users_epoch_final.pt")
    model_reviews = torch.load("Models/Model_Embed_X_reviews_epoch_final.pt")


    model_users = model_users.to('cuda')
    model_reviews = model_reviews.to('cuda')
    model_users.eval()
    model_reviews.eval()

    dataset = TestDataset(prepped_users, prepped_reviews_test)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, generator=torch.Generator(device='cuda'))

    write_test_output(model_users, model_reviews, dataloader)




def train_loop_towers(model_users, model_reviews, model_towers, grouped_data, user_val, review_val, 
                 epochs, lr, print_every, save_every, model_path, batchsize, eval_every):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_towers.parameters(), lr=lr)
    grouped_eval_data = review_val.groupby('accommodation_id')
    if args.num_groups:
        list_groups = [item for item in grouped_eval_data]
        grouped_eval_data = random.choices(list_groups, k=args.num_groups)
    eval_towers(model_users, model_reviews, model_towers, grouped_eval_data, user_val, batchsize)
    print_file("Starting model training")
    user_columns = user_val.columns
    user_cols_to_remove = ['user_id', 'accommodation_id']
    last_avg_mrr = 0
    current_avg_mrr = 0
    user_columns = [col for col in user_columns if col not in user_cols_to_remove]
    for epoch in range(epochs):
        model_towers.train()
        total_loss = 0
        num_samples = 0
        loss = 0       
        iter = 0
        last_avg_mrr = current_avg_mrr

        for name, group in grouped_data:
            # users, reviews, targets = batch
            # batch = group.sample(n=batchsize, replace=True)
            for batch_group_index in range(0, len(group), batchsize):
                iter += 1
                batch = group.iloc[batch_group_index: batch_group_index+batchsize]
                users = batch[user_columns].values
                users_list = [torch.tensor(u.astype(np.float32), dtype=torch.get_default_dtype()) for u in users]
                users = torch.stack(users_list).to('cuda')
                
                reviews_tensor_list = [t for t in batch['review_embed'].values]
                reviews = torch.stack(reviews_tensor_list)
                reviews = reviews.to('cuda')
                with torch.no_grad():
                    reviews_embed = model_reviews(reviews)
                    users_embed = model_users(users)

                current_batchsize = len(reviews_embed)
                for user_index, user_embed in enumerate(users_embed):
                    targets = torch.zeros(current_batchsize, 1, dtype=torch.float32)
                    targets[user_index] = 1.
                    user_embed_repeated = user_embed.repeat(current_batchsize, 1)
                    user_embed_repeated.requires_grad = True
                    reviews_embed.requires_grad = True
                    predictions = model_towers(user_embed_repeated, reviews_embed)
                    loss = criterion(predictions, targets)

                    optimizer.zero_grad()  # Reset gradients
                    loss.backward()
                    optimizer.step()
                    if torch.isnan(loss):
                        print_file("nan loss")
                        exit()
                    loss_batch = loss.item()

                    total_loss += loss_batch
                    num_samples += 1

                if iter % print_every == 0:
                    print_file(f'Avg Loss at Iter {iter}: {total_loss / num_samples}')
        
        print_file(f'{epoch} Epoch Loss Avg: {total_loss / num_samples}')
        if epoch % save_every == 0:
            save_model(model_towers, optimizer, epoch, model_path)
        if epoch % eval_every == 0:
            current_avg_mrr = eval_towers(model_users, model_reviews, model_towers, grouped_eval_data, user_val, batchsize)
            # if current_avg_mrr < last_avg_mrr:
            #     save_model(model_users, optimizer, epoch, model_path + "_users_best")
            #     save_model(model_reviews, optimizer, epoch, model_path + "_reviews_best")
            #     exit()
    save_model(model_towers, optimizer, "final", model_path)

def train_loop_users_reviews(model_users, model_reviews, grouped_data, user_val, review_val, match_val, epochs, lr, grad_clip, print_every,
                   save_every, model_path, batchsize,eval_every, tower=False):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model_users.parameters()) + list(model_reviews.parameters()), lr=lr)
    # dataset_eval = EvalDataset(user_val, review_val, match_val)
    # dataloader_eval = DataLoader(dataset=dataset_eval, batch_size=1, shuffle=True, generator=torch.Generator(device='cuda'))
    # eval_model(model_users, model_reviews, dataloader_eval, reps=10000, tower=tower)

    grouped_eval_data = review_val.groupby('accommodation_id')
    eval_new(model_users, model_reviews, grouped_eval_data, user_val)
    print_file("Starting model training")
    user_columns = user_val.columns
    user_cols_to_remove = ['user_id', 'accommodation_id']
    last_avg_mrr = 0
    current_avg_mrr = 0
    user_columns = [col for col in user_columns if col not in user_cols_to_remove]
    for epoch in range(epochs):
        model_users.train()
        model_reviews.train()
        total_loss = 0
        num_samples = 0
        loss = 0       
        iter = 0
        last_avg_mrr = current_avg_mrr

        for name, group in grouped_data:
            # users, reviews, targets = batch
            iter += 1
            # batch = group.sample(n=batchsize, replace=True)
            for batch_group_index in range(0, len(group), batchsize):
                batch = group.iloc[batch_group_index: batch_group_index+batchsize]
                users = batch[user_columns].values
                users_list = [torch.tensor(u.astype(np.float32), dtype=torch.get_default_dtype()) for u in users]
                users = torch.stack(users_list).to('cuda')
                
                reviews_tensor_list = [t for t in batch['review_embed'].values]
                reviews = torch.stack(reviews_tensor_list)
                reviews = reviews.to('cuda')
                
                users.requires_grad = True
                reviews.requires_grad = True

                reviews_embed = model_reviews(reviews)
                users_embed = model_users(users)

                similarity_vector = users_embed @ reviews_embed.T

                targets = torch.arange(users_embed.shape[0], dtype=torch.long)

                loss = criterion(similarity_vector, targets)
                
                if torch.isnan(loss):
                    print_file("nan loss")
                    exit()
                optimizer.zero_grad()  # Reset gradients

                loss.backward()
                optimizer.step()
                loss_iter = loss.item()
                
                total_loss += loss_iter
                num_samples += 1
            if iter % print_every == 0:
                print_file(f'Avg Loss at Iter {iter}: {total_loss / num_samples}')
        
        print_file(f'{epoch} Epoch Loss Avg: {total_loss / num_samples}')
        if epoch % save_every == 0:
            save_model(model_users, optimizer, epoch, model_path + "_users")
            save_model(model_reviews, optimizer, epoch, model_path + "_reviews")
        if epoch % eval_every == 0:
            current_avg_mrr = eval_new(model_users, model_reviews, grouped_eval_data, user_val)
            # if current_avg_mrr < last_avg_mrr and epoch > 10:
            #     save_model(model_users, optimizer, "final", model_path + "_users")
            #     save_model(model_reviews, optimizer, "final", model_path + "_reviews")
            #     exit()
    save_model(model_users, optimizer, "final", model_path + "_users")
    save_model(model_reviews, optimizer, "final", model_path + "_reviews")
    # eval_model(model_users, model_reviews, dataloader_eval)





EMBED_DIM = 368


def load_and_prep(embedding_model):
    users_train, reviews_train, matches_train = load_dataset('train', frac=args.frac)
    users_val, reviews_val, matches_val = load_dataset('val', frac=args.frac)
    users_test, reviews_test, matches_test = load_dataset('test', frac=args.frac)
    reviews_train = reviews_train.fillna("None Given")
    reviews_val = reviews_val.fillna("None Given")

    del reviews_test, matches_test

    print_file("Prepping Users")
    # print_file(users_train[['guest_type']])
    prep_users_train = preprocess_users(users_train, users_val, users_test, which='train')
    prep_users_val = preprocess_users(users_train, users_val, users_test, which='val')

    
    del users_test


    # if args.num_groups:
    #     merged_train = merge_data(prep_users_train, reviews_train, matches_train).groupby('accommodation_id')
    #     merged_train_list = [g for n, g in merged_train]
    #     merged_train_list = merged_train_list[:args.num_groups]
    #     merged_train_concat = pd.concat(merged_train_list)
        
    #     merged_val = merge_data(prep_users_val, reviews_val, matches_val).groupby('accommodation_id')
    #     merged_val_list = [g for n, g in merged_val]
    #     merged_val_list = merged_val_list[:args.num_groups]
    #     merged_val_concat = pd.concat(merged_val_list)


    #     print_file("Finished Concat")

    #     prepped_reviews_train = preprocess_reviews(embedding_model, merged_train_concat)
    #     prepped_reviews_val = preprocess_reviews(embedding_model, merged_val_concat)

    # else:
    if os.path.exists("Data/train.pkl"):
        prepped_reviews_train = pd.read_pickle("Data/train.pkl")
        prepped_reviews_val = pd.read_pickle("Data/val.pkl")
    else:
        merged_train = merge_data(prep_users_train, reviews_train, matches_train)
        merged_val = merge_data(prep_users_val, reviews_val, matches_val)
        prepped_reviews_train = preprocess_reviews(embedding_model, merged_train)
        prepped_reviews_val = preprocess_reviews(embedding_model, merged_val)
        prepped_reviews_train.to_pickle("Data/train.pkl")
        prepped_reviews_val.to_pickle("Data/val.pkl")

    return prepped_reviews_train, prepped_reviews_val, prep_users_val, prepped_reviews_val, matches_val, prep_users_train


def train_users_reviews(embedding_model):
    USER_MLP_1 = 128
    USER_MLP_2 = 64
    USER_MLP_3 = 128
    USER_MLP_4 = 256
    USER_MLP_5 = 512

    USER_HIDDEN = REVIEW_HIDDEN = 768

    REVIEW_MLP_1 = 256
    REVIEW_MLP_2 = 512
    
    prepped_reviews_train, prepped_reviews_val, prep_users_val, prepped_reviews_val, matches_val, prep_users_train = load_and_prep(embedding_model)

    REVIEW_EMBED_DIM = prepped_reviews_train['review_embed'].iloc[0].shape[0]
    INPUT_DIM = prep_users_train.shape[1] - 2
    EPOCHS = 50
    
    LR = 5e-5
    BATCH_SIZE = 64
    GRAD_CLIP = 5.
    PRINT_EVERY = 4000
    SAVE_EVERY = 2
    EVAL_EVERY = 2
    MODEL_PATH = "Models/Model_Embed_Neo"

    model_users = UserEmbedder(input_dim=INPUT_DIM, hidden_dim=USER_HIDDEN, mlp_1=USER_MLP_1, mlp_2=USER_MLP_2, mlp_3=USER_MLP_3, mlp_4=USER_MLP_4, mlp_5=USER_MLP_5,
                 output_dim=EMBED_DIM, dropout=0.1)
    model_reviews = ReviewEmbedder(input_dim=REVIEW_EMBED_DIM, hidden_dim=REVIEW_HIDDEN, mlp_1=REVIEW_MLP_1, mlp_2=REVIEW_MLP_2,
                 output_dim=EMBED_DIM, dropout=0.1)

    # model_users = torch.load("Models/Model_Embed_Neo_users_epoch_final.pt")
    # model_reviews = torch.load("Models/Model_Embed_Neo_reviews_epoch_final.pt")
    
    print(f"max model seq length: {embedding_model.get_max_seq_length()}")
    tower = None
    grouped = prepped_reviews_train.groupby('accommodation_id_x')

    train_loop_users_reviews(model_users, model_reviews, grouped, prep_users_val, prepped_reviews_val, matches_val, EPOCHS, LR, GRAD_CLIP,
                   PRINT_EVERY, SAVE_EVERY, MODEL_PATH, BATCH_SIZE, EVAL_EVERY, tower=tower)


def train_towers(embedding_model):
    prepped_reviews_train, prepped_reviews_val, prep_users_val, prepped_reviews_val, matches_val, prep_users_train = load_and_prep(embedding_model)

    
    EPOCHS = 5
    
    LR = 3e-5 
    BATCH_SIZE = 32
    PRINT_EVERY = 4000
    SAVE_EVERY = 1
    EVAL_EVERY = 1
    MODEL_PATH = "Models/Model_Towers"

    TOWER_1 = 256
    HIDDEN_DIM = 64
    BRIDGE_1 = 128
    BRIDGE_2 = 64
    BRIDGE_3 = 32


    model_users = torch.load("Models/Model_Embed_Neo_users_epoch_16.pt")
    model_reviews = torch.load("Models/Model_Embed_Neo_reviews_epoch_16.pt")

    grouped = prepped_reviews_train.groupby('accommodation_id_x')

    if args.num_groups:
        group_list = [item for item in grouped]
        grouped = random.choices(group_list, k=args.num_groups)


    model_towers = TwoTowerModel(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, tower_1_dim=TOWER_1, bridge_dim_1=BRIDGE_1, bridge_dim_2=BRIDGE_2, bridge_dim_3=BRIDGE_3,)

    train_loop_towers(model_users, model_reviews, model_towers, grouped, prep_users_val, prepped_reviews_val, EPOCHS, LR, PRINT_EVERY, 
                      SAVE_EVERY, MODEL_PATH, BATCH_SIZE, EVAL_EVERY)
    

def main():

    embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", device='cuda', trust_remote_code=True)

    embedding_model.eval()

    train_towers(embedding_model)
    test_towers(embedding_model)

main()