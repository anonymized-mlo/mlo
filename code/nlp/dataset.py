import pandas as pd
import os
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
import torch.nn as nn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def load_profile(data_dir=None):
    if data_dir==None:
        data_dir = '~/luxemail/data'
    pair_profile = pd.read_csv(f'{data_dir}/pair_profile_with_attention_v2.csv')
    users = pair_profile.loc[:, ['user1', 'user2']]
    x = pair_profile.drop(columns=['dataset_type', 'user1', 'user2', 'travis_sentiment', 'email_embedding']).values
    x_col = pair_profile.drop(columns=['dataset_type', 'user1', 'user2', 'travis_sentiment', 'email_embedding']).columns
    x_index = pair_profile.index
    
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    new_profile = pd.DataFrame(x_scaled, columns=x_col, index=x_index)

    pair_profile = pd.concat([users, new_profile], axis=1)
    pair_profile = pair_profile.set_index(['user1','user2'])

    return pair_profile

def load_data(data_dir=None, save_dir=None, model_dir=None, force_create=False, electra_from_pretrained=False):
    if data_dir==None:
        data_dir = '~/luxemail/data'

    if save_dir==None:
        save_dir = 'split_data'

    if model_dir==None:
        model_dir = '../../trained_model'

    raw_data = load_raw_data(data_dir, save_dir, force_create)
    feature_data = load_profile_feature(data_dir, save_dir, force_create)
    liwc_embedding = load_liwc_embedding(data_dir, save_dir, force_create)
    vader_embedding = load_vader_embedding(data_dir, save_dir, force_create)
    electra_embedding = load_electra_embedding(data_dir, save_dir, model_dir, force_create, electra_from_pretrained)

    print('\nData load Complete')
    print(f'Train: {len(raw_data[0])}')
    print(f'Valid: {len(raw_data[1])}')
    print(f'Test: {len(raw_data[2])}')

    return raw_data, feature_data, liwc_embedding, vader_embedding, electra_embedding

def load_profile_feature(data_dir=None, save_dir=None, model_dir=None, force_create=False):
    target_file_list = [
        'X_train_profile_feature.pkl',
        'X_valid_profile_feature.pkl',
        'X_test_profile_feature.pkl',
    ]
    do_create = False

    if force_create:
        do_create=True
    else:
        for f in target_file_list:
            if f not in os.listdir(save_dir):
                do_create=True
                break

    if do_create==False:
        print(f'Load profile features from {save_dir}')
        ret = load_profile_feature_from_cache(save_dir)
    else:
        print(f'Create profile features in {save_dir}')
        ret = create_profile_feature(data_dir, save_dir, target_file_list)

    return ret

def load_profile_feature_from_cache(save_dir=None):
    X_train_profile_feature = pd.read_pickle(f'./{save_dir}/X_train_profile_feature.pkl')
    X_valid_profile_feature = pd.read_pickle(f'./{save_dir}/X_valid_profile_feature.pkl')
    X_test_profile_feature = pd.read_pickle(f'./{save_dir}/X_test_profile_feature.pkl')

    return X_train_profile_feature, X_valid_profile_feature, X_test_profile_feature

def create_profile_feature(data_dir=None, save_dir=None, target_file_list=None):
    def get_feature(pair_profile, P):
        features = []
        for pair in P:
            f = torch.Tensor(pair_profile.loc[pair, :])
            features.append(f)
        return features

    pair_profile = load_profile(data_dir)
    P_train = pd.read_pickle(f'./{save_dir}/P_train.pkl')
    P_valid = pd.read_pickle(f'./{save_dir}/P_valid.pkl')
    P_test = pd.read_pickle(f'./{save_dir}/P_test.pkl')

    X_train_profile_feature = get_feature(pair_profile, P_train)
    X_valid_profile_feature = get_feature(pair_profile, P_valid)
    X_test_profile_feature = get_feature(pair_profile, P_test)

    for f in target_file_list:
        if f in os.listdir(save_dir):
            print(f'Remove old file {f}')
            os.system(f'rm -rf ./{save_dir}/{f}')

    with open(f'./{save_dir}/X_train_profile_feature.pkl', 'wb') as f:
        pickle.dump(X_train_profile_feature, f)
    with open(f'./{save_dir}/X_valid_profile_feature.pkl', 'wb') as f:
        pickle.dump(X_valid_profile_feature, f)
    with open(f'./{save_dir}/X_test_profile_feature.pkl', 'wb') as f:
        pickle.dump(X_test_profile_feature, f)

    return X_train_profile_feature, X_valid_profile_feature, X_test_profile_feature

def load_electra_embedding(data_dir=None, save_dir=None, model_dir=None, force_create=False, electra_from_pretrained=False):
    assert(save_dir in os.listdir())
    if electra_from_pretrained:
        target_file_list = [
            'X_out_train_electra_from_pretrained.pkl',
            'X_out_valid_electra_from_pretrained.pkl',
            'X_out_test_electra_from_pretrained.pkl',
            'X_in_train_electra_from_pretrained.pkl',
            'X_in_valid_electra_from_pretrained.pkl',
            'X_in_test_electra_from_pretrained.pkl',
        ]

    else:
        target_file_list = [
            'X_out_train_electra.pkl',
            'X_out_valid_electra.pkl',
            'X_out_test_electra.pkl',
            'X_in_train_electra.pkl',
            'X_in_valid_electra.pkl',
            'X_in_test_electra.pkl',
        ]
    do_create = False

    if force_create:
        do_create=True
    else:
        for f in target_file_list:
            if f not in os.listdir(save_dir):
                do_create=True
                break

    if do_create==False:
        print(f'Load electra embedding from {save_dir}')
        ret = load_electra_embedding_from_cache(save_dir, electra_from_pretrained)
    else:
        print(f'Create electra embedding in {save_dir}')
        ret = create_electra_embedding(data_dir, save_dir, model_dir, target_file_list, electra_from_pretrained)
        
    return ret

def load_electra_embedding_from_cache(save_dir=None, electra_from_pretrained=False):
    if electra_from_pretrained:
        X_out_train_electra = pd.read_pickle(f'./{save_dir}/X_out_train_electra_from_pretrained.pkl')
        X_in_train_electra = pd.read_pickle(f'./{save_dir}/X_in_train_electra_from_pretrained.pkl')
        X_out_valid_electra = pd.read_pickle(f'./{save_dir}/X_out_valid_electra_from_pretrained.pkl')
        X_in_valid_electra = pd.read_pickle(f'./{save_dir}/X_in_valid_electra_from_pretrained.pkl')
        X_out_test_electra = pd.read_pickle(f'./{save_dir}/X_out_test_electra_from_pretrained.pkl')
        X_in_test_electra = pd.read_pickle(f'./{save_dir}/X_in_test_electra_from_pretrained.pkl')

    else:
        X_out_train_electra = pd.read_pickle(f'./{save_dir}/X_out_train_electra.pkl')
        X_in_train_electra = pd.read_pickle(f'./{save_dir}/X_in_train_electra.pkl')
        X_out_valid_electra = pd.read_pickle(f'./{save_dir}/X_out_valid_electra.pkl')
        X_in_valid_electra = pd.read_pickle(f'./{save_dir}/X_in_valid_electra.pkl')
        X_out_test_electra = pd.read_pickle(f'./{save_dir}/X_out_test_electra.pkl')
        X_in_test_electra = pd.read_pickle(f'./{save_dir}/X_in_test_electra.pkl')

    return X_out_train_electra, X_in_train_electra, X_out_valid_electra, X_in_valid_electra, X_out_test_electra, X_in_test_electra

def create_electra_embedding(data_dir=None, save_dir=None, model_dir=None, target_file_list=None, electra_from_pretrained=False):
    from transformers import ElectraModel, ElectraTokenizer
    BERT_HIDDEN = 768
    BERT_MAX_LEN = 512

    PRE_TRAINED_MODEL_NAME = 'google/electra-base-discriminator'
    tokenizer = ElectraTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    my_bert = ElectraModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)
    TRAINED_MODEL_PATH = f'{model_dir}/unsup_fine_tuning_ELECTRA.bin'

    class embeddingModel(nn.Module):
        def __init__(
            self,
            token_num: int,
            hidden_dim: BERT_HIDDEN,
            dropout: float=0.1
        ):
            super().__init__()
            self.bert = my_bert
            self.MLM = nn.Sequential(
                nn.Linear(BERT_HIDDEN, hidden_dim),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.classify = nn.Linear(hidden_dim, token_num)
        
        def forward(self, x, mask):
            out = self.bert(input_ids=x, attention_mask=mask)
            full_embedding = self.MLM(out['last_hidden_state'])
            cls_embedding = full_embedding[:, 0, :]
        
            out = self.classify(full_embedding)
            del full_embedding
        
            return out, cls_embedding

    def get_embeddings(senti_model, X):
        vec = []
    
        with torch.no_grad():
            for pair_emails in tqdm(X):
                embeddings = []

                for e in pair_emails:
                    encoding = tokenizer.encode_plus(
                        e,
                        add_special_tokens=True,
                        max_length=BERT_MAX_LEN,
                        return_token_type_ids=False,
                        padding='max_length',
                        return_attention_mask=True,
                        return_tensors='pt',
                        truncation=True,
                    )

                    input_ids = encoding['input_ids'].cuda()
                    attention_mask = encoding['attention_mask'].cuda()

                    _, embedding = senti_model(input_ids, attention_mask)
                    if electra_from_pretrained:
                        embedding = embedding.detach().cpu()
                    else:
                        embedding = embedding.cpu()

                    del input_ids, attention_mask
                    embeddings.append(embedding)

                embeddings = torch.cat(embeddings, dim=0)
                vec.append(embeddings)

        return vec

    X_out_train = pd.read_pickle(f'./{save_dir}/X_out_train.pkl')
    X_in_train = pd.read_pickle(f'./{save_dir}/X_in_train.pkl')
    X_out_valid = pd.read_pickle(f'./{save_dir}/X_out_valid.pkl')
    X_in_valid = pd.read_pickle(f'./{save_dir}/X_in_valid.pkl')
    X_out_test = pd.read_pickle(f'./{save_dir}/X_out_test.pkl')
    X_in_test = pd.read_pickle(f'./{save_dir}/X_in_test.pkl')

    senti_model = embeddingModel(tokenizer.vocab_size, BERT_HIDDEN, 0.1)
    senti_model = nn.DataParallel(senti_model)
    if not electra_from_pretrained:
        senti_model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    senti_model = senti_model.cuda()

    X_out_train_electra = get_embeddings(senti_model, X_out_train)
    X_in_train_electra = get_embeddings(senti_model, X_in_train)
    X_out_valid_electra = get_embeddings(senti_model, X_out_valid)
    X_in_valid_electra = get_embeddings(senti_model, X_in_valid)
    X_out_test_electra = get_embeddings(senti_model, X_out_test)
    X_in_test_electra = get_embeddings(senti_model, X_in_test)

    for f in target_file_list:
        if f in os.listdir(save_dir):
            print(f'Remove old file {f}')
            os.system(f'rm -rf ./{save_dir}/{f}')

    if electra_from_pretrained:
        with open(f'./{save_dir}/X_out_train_electra_from_pretrained.pkl', 'wb') as f:
            pickle.dump(X_out_train_electra, f)
        with open(f'./{save_dir}/X_in_train_electra_from_pretrained.pkl', 'wb') as f:
            pickle.dump(X_in_train_electra, f)
        with open(f'./{save_dir}/X_out_valid_electra_from_pretrained.pkl', 'wb') as f:
            pickle.dump(X_out_valid_electra, f)
        with open(f'./{save_dir}/X_in_valid_electra_from_pretrained.pkl', 'wb') as f:
            pickle.dump(X_in_valid_electra, f)
        with open(f'./{save_dir}/X_out_test_electra_from_pretrained.pkl', 'wb') as f:
            pickle.dump(X_out_test_electra, f)
        with open(f'./{save_dir}/X_in_test_electra_from_pretrained.pkl', 'wb') as f:
            pickle.dump(X_in_test_electra, f)
    else:
        with open(f'./{save_dir}/X_out_train_electra.pkl', 'wb') as f:
            pickle.dump(X_out_train_electra, f)
        with open(f'./{save_dir}/X_in_train_electra.pkl', 'wb') as f:
            pickle.dump(X_in_train_electra, f)
        with open(f'./{save_dir}/X_out_valid_electra.pkl', 'wb') as f:
            pickle.dump(X_out_valid_electra, f)
        with open(f'./{save_dir}/X_in_valid_electra.pkl', 'wb') as f:
            pickle.dump(X_in_valid_electra, f)
        with open(f'./{save_dir}/X_out_test_electra.pkl', 'wb') as f:
            pickle.dump(X_out_test_electra, f)
        with open(f'./{save_dir}/X_in_test_electra.pkl', 'wb') as f:
            pickle.dump(X_in_test_electra, f)

    return X_out_train_electra, X_in_train_electra, X_out_valid_electra, X_in_valid_electra, X_out_test_electra, X_in_test_electra

def load_vader_embedding(data_dir=None, save_dir=None, force_create=False):
    assert(save_dir in os.listdir())
    target_file_list = [
        'X_out_train_vader_aggregated.pkl',
        'X_out_valid_vader_aggregated.pkl',
        'X_out_test_vader_aggregated.pkl',
        'X_in_train_vader_aggregated.pkl',
        'X_in_valid_vader_aggregated.pkl',
        'X_in_test_vader_aggregated.pkl',
    ]
    do_create = False

    if force_create:
        do_create=True
    else:
        for f in target_file_list:
            if f not in os.listdir(save_dir):
                do_create=True
                break

    if do_create==False:
        print(f'Load vader embedding from {save_dir}')
        ret = load_vader_embedding_from_cache(save_dir)
    else:
        print(f'Create vader embedding in {save_dir}')
        ret = create_vader_embedding(data_dir, save_dir, target_file_list)
        
    return ret

def load_vader_embedding_from_cache(save_dir=None):
    X_out_train_vader = pd.read_pickle(f'./{save_dir}/X_out_train_vader_aggregated.pkl')
    X_in_train_vader = pd.read_pickle(f'./{save_dir}/X_in_train_vader_aggregated.pkl')
    X_out_valid_vader = pd.read_pickle(f'./{save_dir}/X_out_valid_vader_aggregated.pkl')
    X_in_valid_vader = pd.read_pickle(f'./{save_dir}/X_in_valid_vader_aggregated.pkl')
    X_out_test_vader = pd.read_pickle(f'./{save_dir}/X_out_test_vader_aggregated.pkl')
    X_in_test_vader = pd.read_pickle(f'./{save_dir}/X_in_test_vader_aggregated.pkl')

    return X_out_train_vader, X_in_train_vader, X_out_valid_vader, X_in_valid_vader, X_out_test_vader, X_in_test_vader

def create_vader_embedding(data_dir=None, save_dir=None, target_file_list=None):
    vader = SentimentIntensityAnalyzer()
    def aggregate(X):
        X_ag = []
        for p in X:
            at = ""
            for s in p:
                at = at + s
            X_ag.append(at)
        return X_ag

    def get_embeddings(X):
        vec = []
        for pair_emails in X:
            embeddings = []

            total_text = ''
            for e in pair_emails:
                total_text = total_text + e

            vader_score = vader.polarity_scores(total_text)
            embeddings = [
                vader_score['compound'],
                vader_score['pos'],
                vader_score['neg'],
                vader_score['neu']
            ]
            embeddings = torch.Tensor(embeddings)
            vec.append(embeddings)
        
        return vec

    X_out_train = pd.read_pickle(f'./{save_dir}/X_out_train.pkl')
    X_in_train = pd.read_pickle(f'./{save_dir}/X_in_train.pkl')
    X_out_valid = pd.read_pickle(f'./{save_dir}/X_out_valid.pkl')
    X_in_valid = pd.read_pickle(f'./{save_dir}/X_in_valid.pkl')
    X_out_test = pd.read_pickle(f'./{save_dir}/X_out_test.pkl')
    X_in_test = pd.read_pickle(f'./{save_dir}/X_in_test.pkl')

    X_out_train_aggregated = aggregate(X_out_train)
    X_in_train_aggregated = aggregate(X_in_train)
    X_out_valid_aggregated = aggregate(X_out_valid)
    X_in_valid_aggregated = aggregate(X_in_valid)
    X_out_test_aggregated = aggregate(X_out_test)
    X_in_test_aggregated = aggregate(X_in_test)

    print('Get train vader embeddings')
    X_out_train_vader_aggregated = get_embeddings(X_out_train_aggregated)
    X_in_train_vader_aggregated = get_embeddings(X_in_train_aggregated)
    print('Get valid vader embeddings')
    X_out_valid_vader_aggregated = get_embeddings(X_out_valid_aggregated)
    X_in_valid_vader_aggregated = get_embeddings(X_in_valid_aggregated)
    print('Get test vader embeddings')
    X_out_test_vader_aggregated = get_embeddings(X_out_test_aggregated)
    X_in_test_vader_aggregated = get_embeddings(X_in_test_aggregated)

    for f in target_file_list:
        if f in os.listdir(save_dir):
            print(f'Remove old file {f}')
            os.system(f'rm -rf ./{save_dir}/{f}')

    OUT_DIR = f'./{save_dir}/'
    with open(OUT_DIR + 'X_out_train_vader_aggregated.pkl', 'wb') as f:
        pickle.dump(X_out_train_vader_aggregated, f)
    with open(OUT_DIR + 'X_in_train_vader_aggregated.pkl', 'wb') as f:
        pickle.dump(X_in_train_vader_aggregated, f)
    with open(OUT_DIR + 'X_out_valid_vader_aggregated.pkl', 'wb') as f:
        pickle.dump(X_out_valid_vader_aggregated, f)
    with open(OUT_DIR + 'X_in_valid_vader_aggregated.pkl', 'wb') as f:
        pickle.dump(X_in_valid_vader_aggregated, f)
    with open(OUT_DIR + 'X_out_test_vader_aggregated.pkl', 'wb') as f:
        pickle.dump(X_out_test_vader_aggregated, f)
    with open(OUT_DIR + 'X_in_test_vader_aggregated.pkl', 'wb') as f:
        pickle.dump(X_in_test_vader_aggregated, f)

    return X_out_train_vader_aggregated, X_in_train_vader_aggregated, X_out_valid_vader_aggregated, X_in_valid_vader_aggregated, X_out_test_vader_aggregated, X_in_test_vader_aggregated


def load_liwc_embedding(data_dir=None, save_dir=None, force_create=False):
    assert(save_dir in os.listdir())
    target_file_list = [
        'X_out_train_liwc_aggregated.pkl',
        'X_out_valid_liwc_aggregated.pkl',
        'X_out_test_liwc_aggregated.pkl',
        'X_in_train_liwc_aggregated.pkl',
        'X_in_valid_liwc_aggregated.pkl',
        'X_in_test_liwc_aggregated.pkl',
    ]
    do_create = False

    if force_create:
        do_create=True
    else:
        for f in target_file_list:
            if f not in os.listdir(save_dir):
                do_create=True
                break

    if do_create==False:
        print(f'Load liwc embedding from {save_dir}')
        ret = load_liwc_embedding_from_cache(save_dir)
    else:
        print(f'Create liwc embedding in {save_dir}')
        ret = create_liwc_embedding(data_dir, save_dir, target_file_list)
        
    return ret

def load_liwc_embedding_from_cache(save_dir=None):
    X_out_train_liwc = pd.read_pickle(f'./{save_dir}/X_out_train_liwc_aggregated.pkl')
    X_in_train_liwc = pd.read_pickle(f'./{save_dir}/X_in_train_liwc_aggregated.pkl')
    X_out_valid_liwc = pd.read_pickle(f'./{save_dir}/X_out_valid_liwc_aggregated.pkl')
    X_in_valid_liwc = pd.read_pickle(f'./{save_dir}/X_in_valid_liwc_aggregated.pkl')
    X_out_test_liwc = pd.read_pickle(f'./{save_dir}/X_out_test_liwc_aggregated.pkl')
    X_in_test_liwc = pd.read_pickle(f'./{save_dir}/X_in_test_liwc_aggregated.pkl')

    return X_out_train_liwc, X_in_train_liwc, X_out_valid_liwc, X_in_valid_liwc, X_out_test_liwc, X_in_test_liwc

def create_liwc_embedding(data_dir=None, save_dir=None, target_file_list=None):
    def get_embeddings(P):
        vec_in = []
        vec_out = []

        for pair in P:
            reverse_pair = (pair[1], pair[0])
        
            embeddings_out = torch.Tensor(liwc_df.loc[pair])
            embeddings_in = torch.Tensor(liwc_df.loc[reverse_pair])

            vec_in.append(embeddings_in)
            vec_out.append(embeddings_out)
   
        return vec_out, vec_in

    liwc_df = pd.read_csv(f'{data_dir}/aggregated_email_liwc.csv')
    liwc_df = liwc_df.drop(columns=['text'])
    liwc_df = liwc_df.set_index(['src','trg'])

    x = liwc_df.values
    x_col = liwc_df.columns
    x_index = liwc_df.index

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    liwc_df = pd.DataFrame(x_scaled, columns=x_col, index=x_index)

    P_train = pd.read_pickle(f'{save_dir}/P_train.pkl')
    P_valid = pd.read_pickle(f'{save_dir}/P_valid.pkl')
    P_test = pd.read_pickle(f'{save_dir}/P_test.pkl')

    print('Get train liwc embeddings')
    X_out_train_liwc, X_in_train_liwc = get_embeddings(P_train)
    print('Get valid liwc embeddings')
    X_out_valid_liwc, X_in_valid_liwc = get_embeddings(P_valid)
    print('Get test liwc embeddings')
    X_out_test_liwc, X_in_test_liwc = get_embeddings(P_test)

    for f in target_file_list:
        if f in os.listdir(save_dir):
            print(f'Remove old file {f}')
            os.system(f'rm -rf ./{save_dir}/{f}')

    OUT_DIR = f'./{save_dir}/'
    with open(OUT_DIR + 'X_out_train_liwc_aggregated.pkl', 'wb') as f:
        pickle.dump(X_out_train_liwc, f)
    with open(OUT_DIR + 'X_in_train_liwc_aggregated.pkl', 'wb') as f:
        pickle.dump(X_in_train_liwc, f)
    with open(OUT_DIR + 'X_out_valid_liwc_aggregated.pkl', 'wb') as f:
        pickle.dump(X_out_valid_liwc, f)
    with open(OUT_DIR + 'X_in_valid_liwc_aggregated.pkl', 'wb') as f:
        pickle.dump(X_in_valid_liwc, f)
    with open(OUT_DIR + 'X_out_test_liwc_aggregated.pkl', 'wb') as f:
        pickle.dump(X_out_test_liwc, f)
    with open(OUT_DIR + 'X_in_test_liwc_aggregated.pkl', 'wb') as f:
        pickle.dump(X_in_test_liwc, f)

    return X_out_train_liwc, X_in_train_liwc, X_out_valid_liwc, X_in_valid_liwc, X_out_test_liwc, X_in_test_liwc


def load_raw_data(data_dir=None, save_dir=None, force_create=False):
    if force_create==False and save_dir in os.listdir():
        print(f'Load raw data from {save_dir}')
        ret = load_raw_data_from_cache(save_dir)
    else:
        print(f'Create raw data in {save_dir}')
        ret = create_raw_data(data_dir, save_dir)
        
    return ret

def load_raw_data_from_cache(save_dir=None):
    P_train = pd.read_pickle(f'./{save_dir}/P_train.pkl')
    P_valid = pd.read_pickle(f'./{save_dir}/P_valid.pkl')
    P_test = pd.read_pickle(f'./{save_dir}/P_test.pkl')

    X_out_train = pd.read_pickle(f'./{save_dir}/X_out_train.pkl')
    X_out_valid = pd.read_pickle(f'./{save_dir}/X_out_valid.pkl')
    X_out_test = pd.read_pickle(f'./{save_dir}/X_out_test.pkl')

    X_in_train = pd.read_pickle(f'./{save_dir}/X_in_train.pkl')
    X_in_valid = pd.read_pickle(f'./{save_dir}/X_in_valid.pkl')
    X_in_test = pd.read_pickle(f'./{save_dir}/X_in_test.pkl')

    y_train = pd.read_pickle(f'./{save_dir}/y_train.pkl')
    y_valid = pd.read_pickle(f'./{save_dir}/y_valid.pkl')
    y_test = pd.read_pickle(f'./{save_dir}/y_test.pkl')

    return P_train, P_valid, P_test, X_out_train, X_out_valid, X_out_test, X_in_train, X_in_valid, X_in_test, y_train, y_valid, y_test

def create_raw_data(data_dir=None, save_dir=None):
    email_path = f'{data_dir}/pair2email.pkl'
    senti_path = f'{data_dir}/travis_sentiment.pkl'
    profile_path = f'{data_dir}/pair_profile_with_senti.csv'

    corpus_dict = pd.read_pickle(email_path)
    travis_sent = pd.read_pickle(senti_path)
    profile = pd.read_csv(profile_path)

    max_email_num = 10000
    seed = 1234

    n = 0
    P = []
    y = []
    X_in = []
    X_out = []

    for pair in travis_sent:
        if pair not in corpus_dict:
            continue

        reverse_pair = (pair[1], pair[0])
        if reverse_pair not in corpus_dict:
            continue

        emails_out = corpus_dict[pair]
        emails_out = sorted(emails_out, key=lambda e: e['date'])
        email_text_out = [e['text'] for e in emails_out if type(e['text'])==str]
        email_text_out = email_text_out[:min(len(email_text_out), max_email_num)]

        emails_in = corpus_dict[reverse_pair]
        emails_in = sorted(emails_in, key=lambda e: e['date'])
        email_text_in = [e['text'] for e in emails_in if type(e['text'])==str]
        email_text_in = email_text_in[:min(len(email_text_in), max_email_num)]

        if len(email_text_in) == 0 or len(email_text_out) == 0:
            continue

        label = travis_sent[pair]

        n += 1
        P.append(pair)
        y.append(label)
        X_out.append(email_text_out)
        X_in.append(email_text_in)

    P_train, P_test_and_valid, \
    X_in_train, X_in_test_and_valid, \
    X_out_train, X_out_test_and_valid, \
    y_train, y_test_and_valid = train_test_split(
        P, X_in, X_out, y, test_size=0.3, random_state=seed
    )

    P_test, P_valid, \
    X_in_test, X_in_valid, \
    X_out_test, X_out_valid, \
    y_test, y_valid = train_test_split(
        P_test_and_valid, X_in_test_and_valid, X_out_test_and_valid, y_test_and_valid, test_size=0.5, random_state=seed
    )

    #print(n)
    #print(len(X_out_train))
    #print(len(X_out_valid))
    #print(len(X_out_test))

    
    new_P_train = []
    new_X_in_train = []
    new_X_out_train = []
    new_y_train = []

    for i in range(len(P_train)):
        p = P_train[i]
        x_in = X_in_train[i]
        x_out = X_out_train[i]
        y = y_train[i]
    
        if len(profile.loc[((profile['user1']==p[0]) & (profile['user2']==p[1])), :]) == 0:
            continue
    
        new_P_train.append(p)
        new_X_in_train.append(x_in)
        new_X_out_train.append(x_out)
        new_y_train.append(y)

    new_P_valid = []
    new_X_in_valid = []
    new_X_out_valid = []
    new_y_valid = []

    for i in range(len(P_valid)):
        p = P_valid[i]
        x_in = X_in_valid[i]
        x_out = X_out_valid[i]
        y = y_valid[i]
    
        if len(profile.loc[((profile['user1']==p[0]) & (profile['user2']==p[1])), :]) == 0:
            continue
    
        new_P_valid.append(p)
        new_X_in_valid.append(x_in)
        new_X_out_valid.append(x_out)
        new_y_valid.append(y)

    new_P_test = []
    new_X_in_test = []
    new_X_out_test = []
    new_y_test = []

    for i in range(len(P_test)):
        p = P_test[i]
        x_in = X_in_test[i]
        x_out = X_out_test[i]
        y = y_test[i]
    
        if len(profile.loc[((profile['user1']==p[0]) & (profile['user2']==p[1])), :]) == 0:
            continue
    
        new_P_test.append(p)
        new_X_in_test.append(x_in)
        new_X_out_test.append(x_out)
        new_y_test.append(y)

    if 'split_data' in os.listdir():
        print('Remove old raw data.')
        os.system(f'rm -rf ./{save_dir}')
    os.system(f'mkdir ./{save_dir}')

    OUT_DIR = f'./{save_dir}/'
    with open(OUT_DIR + 'P_train.pkl', 'wb') as f:
        pickle.dump(new_P_train, f)
    with open(OUT_DIR + 'P_valid.pkl', 'wb') as f:
        pickle.dump(new_P_valid, f)
    with open(OUT_DIR + 'P_test.pkl', 'wb') as f:
        pickle.dump(new_P_test, f)
    
    with open(OUT_DIR + 'X_out_train.pkl', 'wb') as f:
        pickle.dump(new_X_out_train, f)
    with open(OUT_DIR + 'X_out_valid.pkl', 'wb') as f:
        pickle.dump(new_X_out_valid, f)
    with open(OUT_DIR + 'X_out_test.pkl', 'wb') as f:
        pickle.dump(new_X_out_test, f)
    
    with open(OUT_DIR + 'X_in_train.pkl', 'wb') as f:
        pickle.dump(new_X_in_train, f)
    with open(OUT_DIR + 'X_in_valid.pkl', 'wb') as f:
        pickle.dump(new_X_in_valid, f)
    with open(OUT_DIR + 'X_in_test.pkl', 'wb') as f:
        pickle.dump(new_X_in_test, f)
    
    with open(OUT_DIR + 'y_train.pkl', 'wb') as f:
        pickle.dump(new_y_train, f)
    with open(OUT_DIR + 'y_valid.pkl', 'wb') as f:
        pickle.dump(new_y_valid, f)
    with open(OUT_DIR + 'y_test.pkl', 'wb') as f:
        pickle.dump(new_y_test, f)
    

    return new_P_train, new_P_valid, new_P_test, new_X_out_train, new_X_out_valid, new_X_out_test, new_X_in_train, new_X_in_valid, new_X_in_test, new_y_train, new_y_valid, new_y_test
