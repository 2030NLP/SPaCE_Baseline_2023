import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer


class Task1Model(nn.Module):
    def __init__(self, params):
        super(Task1Model, self).__init__()
        self.bert_model = BertModel.from_pretrained(params['base_model'])
        bert_output_dim = self.bert_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.tag_num = (6 + 1) # 6 types & 0 for plain text
        self.tag_layer = nn.Linear(bert_output_dim, self.tag_num)  

        self.tag_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.tokenizer = AutoTokenizer.from_pretrained(params['base_model'])

        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and params['cuda'] else "cpu"
        )
        
        if (params['load_model_path'] is not None):
            self.load_model(params['load_model_path'])
    

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def predict(self,
        input_ids, 
        token_type_ids, 
        attention_mask, 
    ):
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )

        token_embeddings = outputs.last_hidden_state
        tag_predictions = self.tag_layer(token_embeddings)
        tag_predictions = torch.argmax(tag_predictions, dim=2)
        return tag_predictions


    def forward(self, 
        input_ids, 
        token_type_ids, 
        attention_mask, 
        tag_labels,
    ):  
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )
        token_embeddings = outputs.last_hidden_state
        tag_predictions = self.tag_layer(token_embeddings)
        tag_loss = self.tag_criterion(tag_predictions.view(-1, self.tag_num), tag_labels.view(-1))

        return tag_loss