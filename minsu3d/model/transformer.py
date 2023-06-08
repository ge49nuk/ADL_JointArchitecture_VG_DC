import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=134):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



    
class Transformer(nn.Module):
    def __init__(self, max_text_len, dim_model=64, dim_ptfeats=32, dim_wdfeats=768, size_vocab=30522, nlayers=2, dropout_p=0.1, nhead=1):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.dim_ptfeats = dim_ptfeats
        self.dim_wdfeats = dim_wdfeats
        self.max_text_len = max_text_len
        self.size_vocab = size_vocab
        
        # Layers
        # Transform point and word to have dimension=dim_model
        self.point_to_model = nn.Linear(self.dim_ptfeats, self.dim_model)
        self.word_to_model = nn.Linear(self.dim_wdfeats, self.dim_model)
        # Transformer encoder
        self.positional_encoder = PositionalEncoding(self.dim_model, dropout_p, max_len=self.max_text_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        # Decoder layer
        self.grdhead = nn.Sequential(nn.Linear(self.dim_model, self.dim_ptfeats),nn.Linear(self.dim_ptfeats,1))         # One score for each box token
        self.caphead = nn.Linear(self.dim_model, self.size_vocab)  # (#Vocab) scores for each text token
        # Define loss criterion
        self.loss_criterion = nn.CrossEntropyLoss()


        #TEST
        self.dim_test = 32
        self.point_to_test = nn.Linear(self.dim_ptfeats, self.dim_test)
        self.word_to_emb = nn.Sequential(nn.Linear(self.max_text_len * self.dim_wdfeats, self.dim_test))
        self.lin_Test = nn.Sequential(
                                      nn.Linear((self.max_text_len+1) * self.dim_test, 256), 
                                      nn.BatchNorm1d(256),
                                      nn.ReLU(),
                                    #   nn.Dropout(),
                                      nn.Linear(256,256), 
                                      nn.BatchNorm1d(256),
                                      nn.ReLU(),
                                    #   nn.Dropout(),
                                      nn.Linear(256,256), 
                                      nn.BatchNorm1d(256),
                                      nn.ReLU(),
                                    #   nn.Dropout(),
                                      nn.Linear(256,2))

        # self.init_weights()

    def forward(self, data_dict, output_dict):
        # Get box tokens:
        proposals_idx = output_dict["proposals_idx"]
        point_features = (output_dict["point_features"])[proposals_idx[:, 1]]

        unique_proposals_idx, counts = proposals_idx[:, 0].unique(return_counts=True)
        num_proposals = len(unique_proposals_idx)

        for i, count in enumerate(counts[:-1]):
            counts[i+1] += count
        counts = counts[:-1]

        point_features_chunks = torch.tensor_split(point_features, counts.cpu(), dim=0)
        box_tokens = torch.stack([chunk.mean(dim=0) for chunk in point_features_chunks], dim=0)
        assert box_tokens.size() == (num_proposals, self.dim_ptfeats)
    

        # #LINEAR TEST MODEL
        # assert box_tokens.size() == (num_proposals, self.dim_test)
        # text_tokens = output_dict["descr_embedding"][0]
        # assert text_tokens.shape == (self.max_text_len, self.dim_wdfeats)
        # text_tokens = text_tokens[:,:self.dim_test*2:2].reshape((self.max_text_len, self.dim_test))

        # global_box_token = box_tokens.mean(dim=0, keepdim=True)
        # global_visual_cue = text_tokens

        # global_visual_cue[:] += global_box_token
        # assert global_visual_cue.shape == (self.max_text_len, self.dim_test)

        # VG_tokens = torch.zeros((num_proposals, (self.max_text_len+1) * self.dim_test))
        # for i in range(num_proposals):
        #     VG_token = torch.cat((box_tokens[i], global_visual_cue.flatten()))
        #     VG_tokens[i] = VG_token
    

        # out = self.lin_Test(VG_tokens.to("cuda"))

        # return {"VG_scores": out}

        
        box_tokens = self.point_to_model(box_tokens)
        assert box_tokens.size() == (num_proposals, self.dim_model)
        # prepared: (box_tokens, unique_proposals_idx)
        
        # Get text tokens:
        text_tokens = output_dict["descr_embedding"][0]  # word embeddings start with [CLS]
        # word_ids = data_dict["descr_token"][0]           # word ids from vocab
        assert text_tokens.shape == (self.max_text_len, self.dim_wdfeats)
        text_tokens = self.word_to_model(text_tokens)
        assert text_tokens.shape == (self.max_text_len, self.dim_model)
        text_tokens = self.positional_encoder(text_tokens.unsqueeze(1))
        text_tokens = text_tokens[:,0]
        assert text_tokens.shape == (self.max_text_len, self.dim_model)
        # prepared: (queried_obj, text_tokens, word_ids)
    

        # Visual grounding pass
        global_box_token = box_tokens.mean(dim=0, keepdim=True)
        global_visual_cue = text_tokens
        global_visual_cue[:] += global_box_token
        VG_tokens = torch.cat((box_tokens, global_visual_cue), dim=0)
        assert VG_tokens.size() == (num_proposals + self.max_text_len, self.dim_model)
        output_VG_tokens = self.transformer_encoder(VG_tokens)
        output_box_tokens = output_VG_tokens[:num_proposals]
        assert output_box_tokens.size() == (num_proposals, self.dim_model)
        VG_scores = self.grdhead(output_box_tokens).flatten()
        assert VG_scores.size() == (num_proposals,)
        
        # # Dense captioning pass
        # queried_box_token = box_tokens[queried_obj]
        # queried_box_token = queried_box_token.view(1, -1)
        # captioning_cue = text_tokens + queried_box_token
        # DC_tokens = torch.cat((box_tokens, captioning_cue), dim=0)
        # assert DC_tokens.size() == (num_proposals + self.max_text_len, self.dim_model)
        # mask = self.get_seq2seq_mask(num_proposals, self.max_text_len)
        # output_DC_tokens = self.transformer_encoder(DC_tokens, mask.to("cuda"))
        # output_text_tokens = output_DC_tokens[num_proposals:]
        # assert output_text_tokens.size() == (self.max_text_len, self.dim_model)
        # DC_scores = self.caphead(output_text_tokens)
        # assert DC_scores.size() == (self.max_text_len, self.size_vocab)
        
        return {"VG_scores": VG_scores}#, "DC_scores": DC_scores}
    
    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.transformer_encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.grdhead.bias)
        nn.init.zeros_(self.caphead.bias)
        nn.init.uniform_(self.grdhead.weight, -initrange, initrange)
        nn.init.uniform_(self.caphead.weight, -initrange, initrange)
    

    def get_seq2seq_mask(self, num_proposals, size) -> torch.tensor:
        mask_upper_left = torch.full((num_proposals, num_proposals), float(0.0))
        mask_upper_right = torch.full((num_proposals, size), float('-inf'))
        mask_upper = torch.cat((mask_upper_left, mask_upper_right), dim=1)

        mask_bottom_left = torch.full((size, num_proposals), float(0.0))
        mask_bottom_right = torch.triu(torch.ones(size, size)) # Upper triangular matrix
        mask_bottom_right = mask_bottom_right.float().masked_fill(mask_bottom_right == 0, float(0.0)).masked_fill(mask_bottom_right == 1, float('-inf')).transpose(0, 1)
        mask_bottom = torch.cat((mask_bottom_left, mask_bottom_right), dim=1)

        mask = torch.cat((mask_upper, mask_bottom), dim=0)
        return mask
        # -----box tokens---- ---text tokens---
        # 0.0 0.0 0.0 0.0 0.0 -inf -inf -inf -inf
        # 0.0 0.0 0.0 0.0 0.0 -inf -inf -inf -inf
        # 0.0 0.0 0.0 0.0 0.0 -inf -inf -inf -inf
        # 0.0 0.0 0.0 0.0 0.0 -inf -inf -inf -inf
        # 0.0 0.0 0.0 0.0 0.0  0.0 -inf -inf -inf
        # 0.0 0.0 0.0 0.0 0.0  0.0  0.0 -inf -inf
        # 0.0 0.0 0.0 0.0 0.0  0.0  0.0  0.0 -inf
        # 0.0 0.0 0.0 0.0 0.0  0.0  0.0  0.0 -inf