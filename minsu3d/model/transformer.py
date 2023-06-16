import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # dim_model: the embedding dimension
        # max_len: the max. length of the incoming sequence
        # dropout_p: dropout probability
        
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        # pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1) # For batched input
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # token_embedding: size(sequence_len, dim_emb)
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])



    
class Transformer(nn.Module):
    def __init__(self, dim_model=512, dim_ptfeats=32, dim_wdfeats=768, max_text_len=134, num_cls=18, size_vocab=30522, dropout_p=0.1, nhead=1, nlayers=2):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.dim_ptfeats = dim_ptfeats
        self.dim_wdfeats = dim_wdfeats
        self.size_vocab = size_vocab
        self.num_cls = num_cls
        
        # Layers
        # Transform point and word to have dimension=dim_model
        self.point_to_model = nn.Sequential(
            nn.Linear(self.dim_ptfeats, 64),
            nn.Linear(64, self.dim_model)
        )
        self.word_to_model = nn.Sequential(
            nn.Linear(self.dim_wdfeats, 640),
            nn.Linear(640, self.dim_model)
        )
        # Transformer encoder
        self.positional_encoder = PositionalEncoding(self.dim_model, dropout_p, max_len=max_text_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        # Decoder layer: One score for each box token
        self.grdhead = nn.Sequential(
            nn.Linear(self.dim_model, 64),
            nn.Linear(64, self.dim_ptfeats),
            nn.Linear(self.dim_ptfeats, 1)
        )
        # Decoder layer: (#Vocab) scores for each text token
        self.caphead = nn.Sequential(
            nn.Linear(self.dim_model, 640),
            nn.Linear(640, self.dim_ptfeats),
            nn.Linear(self.dim_ptfeats, self.size_vocab)
        )
        # Decoder layer: (#Classes) scores for each [CLS] token
        self.clshead = nn.Sequential(
            nn.Linear(self.dim_model, 512),
            nn.Linear(512, 256),
            nn.Linear(256, self.num_cls)
        )

        # Define loss criterion
        self.loss_criterion = nn.CrossEntropyLoss()

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
        
        box_tokens = self.point_to_model(box_tokens)
        assert box_tokens.size() == (num_proposals, self.dim_model)
        # Prepared: (box_tokens, unique_proposals_idx)
        
        # Get text tokens:
        text_tokens = output_dict["descr_embedding"][0]  # word embeddings from BERT (start/end with [CLS]/[SEP], variable length)
        num_tokens = data_dict["num_descr_tokens"][0]
        text_tokens = text_tokens[:(num_tokens-1)] # Taken the first N-1 tokens as input, which means ignoring [SEP]
        len_text_tokens = text_tokens.shape[0]
        assert text_tokens.shape == (len_text_tokens, self.dim_wdfeats)
        text_tokens = self.word_to_model(text_tokens)
        assert text_tokens.shape == (len_text_tokens, self.dim_model)
        text_tokens = self.positional_encoder(text_tokens)
        assert text_tokens.shape == (len_text_tokens, self.dim_model)
        # Prepared: (text_tokens)
    
        # Get target proposals:
        target_proposals = output_dict["target_proposals"]
        num_target_proposals = len(target_proposals)
        # Prepared: (target_proposals, num_target_proposals)


        # Visual grounding pass
        global_box_token = box_tokens.mean(dim=0, keepdim=True)
        global_visual_cue = text_tokens + global_box_token
        VG_tokens = torch.cat((box_tokens, global_visual_cue), dim=0)
        assert VG_tokens.size() == (num_proposals + len_text_tokens, self.dim_model)
        output_VG_tokens = self.transformer_encoder(VG_tokens)
        output_box_tokens = output_VG_tokens[:num_proposals]
        assert output_box_tokens.size() == (num_proposals, self.dim_model)
        Match_scores = (self.grdhead(output_box_tokens)).flatten()
        assert Match_scores.size() == (num_proposals,)
        # Compute Matching loss
        Match_targets = torch.zeros((Match_scores.shape))
        for p in target_proposals:
            Match_targets[p] = 1.0 / num_target_proposals
        Match_loss = self.loss_criterion(Match_scores, Match_targets.to("cuda"))
        # Compute CLS loss
        encoded_cls_token = output_VG_tokens[num_proposals]
        CLS_scores = self.clshead(encoded_cls_token)
        CLS_target = data_dict["target_class"] # One-hot vector
        CLS_loss = self.loss_criterion(CLS_scores, CLS_target.to("cuda"))
        

        # Dense captioning pass
        DC_loss = 0.0
        if num_target_proposals > 0:
            for i, target_proposal in enumerate(target_proposals):
                target_box_token = box_tokens[i]
                target_box_token = target_box_token.view(1, self.dim_model)
                captioning_cue = text_tokens + target_box_token
                assert captioning_cue.size() == (len_text_tokens, self.dim_model)
                DC_tokens = torch.cat((box_tokens, captioning_cue), dim=0)
                assert DC_tokens.size() == (num_proposals + len_text_tokens, self.dim_model)
                mask = self.get_seq2seq_mask(num_proposals, len_text_tokens)
                output_DC_tokens = self.transformer_encoder(DC_tokens, mask.to("cuda"))
                output_text_tokens = output_DC_tokens[num_proposals:]
                assert output_text_tokens.size() == (len_text_tokens, self.dim_model)
                DC_scores = self.caphead(output_text_tokens)
                assert DC_scores.size() == (len_text_tokens, self.size_vocab)
                # Compute DC loss
                target_word_ids = data_dict["descr_tokens"][0]
                target_word_ids = target_word_ids[1:num_tokens] # Taken the last N-1 tokens as target
                target_word_ids = nn.functional.one_hot(target_word_ids, self.size_vocab)
                target_word_ids = torch.tensor(target_word_ids, dtype=torch.float32)
                DC_loss += self.loss_criterion(DC_scores, target_word_ids)
            DC_loss /= num_target_proposals

        return {"Match_scores": Match_scores, "Match_loss": Match_loss,
                "CLS_scores": CLS_scores, "CLS_loss": CLS_loss,
                "DC_scores": DC_scores, "DC_loss": DC_loss}
    
    
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
        mask_bottom_right = mask_bottom_right.float().masked_fill(mask_bottom_right==0, float('-inf')).masked_fill(mask_bottom_right==1, float(0.0)).transpose(0, 1)
        mask_bottom = torch.cat((mask_bottom_left, mask_bottom_right), dim=1)

        mask = torch.cat((mask_upper, mask_bottom), dim=0)
        return mask
        #-box tokens- ------text tokens------
        # 0.0 0.0 0.0 -inf -inf -inf -inf -inf
        # 0.0 0.0 0.0 -inf -inf -inf -inf -inf
        # 0.0 0.0 0.0 -inf -inf -inf -inf -inf
        # 0.0 0.0 0.0  0.0 -inf -inf -inf -inf
        # 0.0 0.0 0.0  0.0  0.0 -inf -inf -inf
        # 0.0 0.0 0.0  0.0  0.0  0.0 -inf -inf
        # 0.0 0.0 0.0  0.0  0.0  0.0  0.0 -inf
        # 0.0 0.0 0.0  0.0  0.0  0.0  0.0  0.0
