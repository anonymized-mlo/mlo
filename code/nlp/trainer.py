import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, classification_report

class EmailEncoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        num_layers: int=2,
        dropout: float=0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            input_dim,
            nhead=2,
            dim_feedforward=128,
            dropout=dropout,
            activation='gelu',
        )
        self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x
    ):
        out = x
        pad = (x[:,:,0]==0.)
        out = self.encoders(src=out.transpose(0,1), src_key_padding_mask=pad)
        out = out.transpose(0,1)
        del pad

        return out

class MloClassifier(nn.Module):
    def __init__(
        self,
        param_dict=None,
    ):
        super().__init__()
        position_dict = {
            'feature': 0,
            'liwc_out': 1,
            'liwc_in': 2,
            'vader_out': 3,
            'vader_in': 4,
            'electra_out': 5,
            'electra_in': 6,
            'liwc_leaves': 7,
            'vader_leaves': 8,
        }

        flag_dict = param_dict['flags']
        self.flag_dict = flag_dict
        #self.position_dict = calculate_positions(flag_dict)
        self.position_dict = position_dict
        print('\nPosition Dict')
        print(self.position_dict)

        model_param_dict = param_dict['model_param']
        input_dim_dict = param_dict['input_dim']
        rep_dim_dict = param_dict['rep_dim']


        total_rep_dim = sum([rep_dim_dict[i] for i in flag_dict if flag_dict[i]])

        if model_param_dict['activation']=='relu':
            self.activation=nn.ReLU()
        elif model_param_dict['activation']=='tanh':
            self.activation=nn.Tanh()
        else:
            assert(0)
        self.dropout = nn.Dropout(model_param_dict['dropout'])

        if flag_dict['electra']:
            self.use_attention = model_param_dict['use_attention']
            if self.use_attention:
                self.electra_out_encoder = EmailEncoder(
                    input_dim=input_dim_dict['electra'],
                    num_layers=model_param_dict['num_layers'],
                    dropout=model_param_dict['dropout']
                )
                self.electra_in_encoder = EmailEncoder(
                    input_dim=input_dim_dict['electra'],
                    num_layers=model_param_dict['num_layers'],
                    dropout=model_param_dict['dropout']
                )
            self.electra_linear = nn.Linear(2 * input_dim_dict['electra'], rep_dim_dict['electra'])

        if flag_dict['vader']:
            self.vader_linear = nn.Linear(input_dim_dict['vader'] * 2, rep_dim_dict['vader'])

        if flag_dict['vader_leaves']:
            self.vader_leaves_linear = nn.Linear(input_dim_dict['vader_leaves'], rep_dim_dict['vader_leaves'])

        if flag_dict['liwc']:
            self.liwc_linear = nn.Linear(input_dim_dict['liwc'] * 2, rep_dim_dict['liwc'])

        if flag_dict['liwc_leaves']:
            self.liwc_leaves_linear = nn.Linear(input_dim_dict['liwc_leaves'], rep_dim_dict['liwc_leaves'])

        if flag_dict['feature']:
            self.feature_linear_1 = nn.Linear(input_dim_dict['feature'], model_param_dict['dim_features'])
            self.feature_linear_2 = nn.Linear(model_param_dict['dim_features'], model_param_dict['dim_features'])
            self.feature_linear_3 = nn.Linear(model_param_dict['dim_features'], rep_dim_dict['feature'])

        if flag_dict['final']:
            self.final_linear = nn.Linear(total_rep_dim, rep_dim_dict['final'])
            self.classify_linear = nn.Linear(rep_dim_dict['final'], 1)
        else:
            self.classify_linear = nn.Linear(total_rep_dim, 1)

    def get_electra_embedding(
        self,
        electra_out,
        electra_in,
    ):
        if self.use_attention:
            rep_electra_out = electra_out + self.electra_out_encoder(electra_out)
            rep_electra_in = electra_in + self.electra_in_encoder(electra_in)
        else:
            rep_electra_out = electra_out
            rep_electra_in = electra_in
        rep_electra = torch.cat((torch.mean(rep_electra_out, dim=1), torch.mean(rep_electra_in, dim=1)), 1)
        rep_electra = self.activation(self.electra_linear(rep_electra))

        return rep_electra

    def get_vader_embedding(
        self,
        vader_out,
        vader_in,
    ):
        rep_vader = torch.cat([vader_out, vader_in], 1)
        rep_vader = self.activation(self.vader_linear(rep_vader))

        return rep_vader

    def get_vader_leaves_embedding(
        self,
        vader_leaves,
    ):
        rep_vader_leaves = torch.flatten(vader_leaves, start_dim=1)
        rep_vader_leaves = self.activation(self.vader_leaves_linear(rep_vader_leaves))

        return rep_vader_leaves

    def get_liwc_embedding(
        self,
        liwc_out,
        liwc_in,
    ):
        rep_liwc = torch.cat([liwc_out, liwc_in], 1)
        rep_liwc = self.activation(self.liwc_linear(rep_liwc))

        return rep_liwc

    def get_liwc_leaves_embedding(
        self,
        liwc_leaves,
    ):
        rep_liwc_leaves = torch.flatten(liwc_leaves, start_dim=1)
        rep_liwc_leaves = self.activation(self.liwc_leaves_linear(rep_liwc_leaves))

        return rep_liwc_leaves

    def get_feature_embedding(
        self,
        feature: torch.Tensor,
    ):
        rep_feature = self.dropout(self.feature_linear_1(feature))
        rep_feature = self.dropout(self.feature_linear_2(rep_feature)) + rep_feature
        rep_feature = self.activation(self.feature_linear_3(rep_feature))

        return rep_feature

    def forward(
        self,
        inputs,
    ):
        embeddings = []

        if self.flag_dict['feature']:
            feature = inputs[self.position_dict['feature']]
            rep_feature = self.get_feature_embedding(feature)
            embeddings.append(rep_feature)

        if self.flag_dict['liwc']:
            liwc_out = inputs[self.position_dict['liwc_out']]
            liwc_in = inputs[self.position_dict['liwc_in']]
            rep_liwc = self.get_liwc_embedding(liwc_out, liwc_in)
            embeddings.append(rep_liwc)

        if self.flag_dict['vader']:
            vader_out = inputs[self.position_dict['vader_out']]
            vader_in = inputs[self.position_dict['vader_in']]
            rep_vader = self.get_vader_embedding(vader_out, vader_in)
            embeddings.append(rep_vader)

        if self.flag_dict['electra']:
            electra_out = inputs[self.position_dict['electra_out']]
            electra_in = inputs[self.position_dict['electra_in']]
            rep_electra = self.get_electra_embedding(electra_out, electra_in)
            embeddings.append(rep_electra)

        if self.flag_dict['liwc_leaves']:
            liwc_leaves = inputs[self.position_dict['liwc_leaves']]
            rep_liwc_leaves = self.get_liwc_leaves_embedding(liwc_leaves)
            embeddings.append(rep_liwc_leaves)

        if self.flag_dict['vader_leaves']:
            vader_leaves = inputs[self.position_dict['vader_leaves']]
            rep_vader_leaves = self.get_vader_leaves_embedding(vader_leaves)
            embeddings.append(rep_vader_leaves)

        embeddings = torch.cat(embeddings, dim=1)
        if self.flag_dict['final']:
            embeddings = self.activation(embeddings)
            embeddings = self.dropout(embeddings)
            embeddings = self.final_linear(embeddings)
        
        #logit = self.classify_linear(self.dropout(self.activation(embeddings))).squeeze(dim=1)
        #logit = self.classify_linear(self.dropout(embeddings)).squeeze(dim=1)
        #logit = self.classify_linear(self.activation(embeddings)).squeeze(dim=1)
        logit = self.classify_linear(embeddings).squeeze(dim=1)

        return logit, embeddings

def train(
    model=None,
    optimizer=None,
    scheduler=None,
    criterion=None,
    train_dataloader=None
):
    model.train()
    train_loss = []
    for d in train_dataloader:
        inputs, label = d
        inputs = [i.cuda() for i in inputs]
        label = label.cuda()

        logit, _ = model(inputs)

        loss = criterion(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

    avg_train_loss = sum(train_loss) / len(train_loss)
    return model, optimizer, scheduler, avg_train_loss

@torch.no_grad()
def evaluate(
    model=None,
    criterion=None,
    eval_dataloader=None,
    test=False
):
    model.eval()
    eval_loss = []
    labels = []
    probs = []
    preds = []
    for d in eval_dataloader:
        inputs, label = d
        inputs = [i.cuda() for i in inputs]
        label = label.cuda()

        logit, _ = model(inputs)
        pred = torch.where(logit > 0, 1, 0)
        prob = torch.sigmoid(logit)

        loss = criterion(logit, label)
        eval_loss.append(loss.item())
        labels.extend(label.int().cpu().tolist())
        preds.extend(pred.cpu().tolist())
        probs.extend(prob.cpu().tolist())


    avg_eval_loss = sum(eval_loss) / len(eval_loss)
    pred_auc = roc_auc_score(labels, preds)
    prob_auc = roc_auc_score(labels, probs)

    if test:        
        class_names = ['Negative', 'Positive']
        print(classification_report(labels, preds, target_names=class_names, digits=4))
    
    return avg_eval_loss, pred_auc, prob_auc
