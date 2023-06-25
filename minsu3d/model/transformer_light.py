import torch
import torch.nn as nn
import math
import sys

class PositionalEncoder(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # dim_model: the embedding dimension
        # max_len: the max. length of the incoming sequence
        
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
        pos_encoding = pos_encoding.unsqueeze(0) # For batched input
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # token_embedding: size(batch, sequence_len, dim_emb)
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding)

    
class Transformer_Light(nn.Module):
    def __init__(self, dim_model=512, dim_ptfeats=32, dim_wdfeats=768, max_text_len=134, num_cls=18, size_vocab=30522, dropout_p=0.1, nhead=1, nlayers=2,
                 samples_per_box=256):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.dim_ptfeats = dim_ptfeats
        self.dim_wdfeats = dim_wdfeats
        self.size_vocab = size_vocab
        self.num_cls = num_cls
        self.samples_per_box = samples_per_box
        
        # Transform point and word to have dimension=dim_model
        self.point_to_model = nn.Sequential(
            nn.Linear(self.dim_ptfeats*self.samples_per_box, 1024),
            nn.Linear(1024, self.dim_model)
        )
        self.word_to_model = nn.Sequential(
            nn.Linear(self.dim_wdfeats, 640),
            nn.Linear(640, self.dim_model)
        )
        # Transformer encoder
        self.positional_encoder = PositionalEncoder(self.dim_model, dropout_p, max_text_len) # Not 100% sure correct
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
        self.loss_criterion_bce = nn.BCEWithLogitsLoss()
        
        # Initialize weights
        # self.init_weights()
        self.seen = []

    def forward(self, data_dict):
        proposals_batched = data_dict['proposals']
        batch_splits = data_dict['batch_splits']
        target_proposals = data_dict['target_proposals']
        target_proposal_splits = data_dict['target_proposal_splits']
        text_embeddings = data_dict['text_embeddings'].detach().clone()
        target_word_ids = data_dict['target_word_ids']
        num_tokens = data_dict['num_tokens']
        target_classes = data_dict['target_classes']

        # Transform text embedding to have dim_model
        text_embeddings = self.word_to_model(text_embeddings)
        text_embeddings = self.positional_encoder(text_embeddings)

        # Transform proposal to dim_model:
        proposals_batched = torch.tensor_split(proposals_batched, batch_splits[1:-1], dim=0)

        proposals_t = []
        for props in proposals_batched:
            proposals_t.append(self.point_to_model(props.to("cuda")))
        # scenes_sampled = []
        # for scene in scenes:
        #     proposals = torch.zeros((scene.shape[0], self.dim_ptfeats * self.samples_per_box))
        #     for i,proposal in enumerate(scene):
        #         num_samples = self.samples_per_box
        #         feature_means = torch.mean(proposal, dim=1)
        #         sampled_indices = torch.topk(feature_means, num_samples)[1]
        #         sampled_chunk = proposal[sampled_indices]
        #         proposals[i] = sampled_chunk.flatten()
        #     proposals = self.point_to_model(proposals.to("cuda"))
        #     scenes_sampled.append(proposals)
        # scenes = scenes_sampled

        # Get target_proposals
        best_proposals = torch.tensor_split(target_proposals, target_proposal_splits[1:-1], dim=0)

        num_scenes = len(proposals_t) # = batch size
        num_instances = [props.shape[0] for props in proposals_t] # Number of instances in each scene

        Match_scores_list = []
        CLS_scores_list = []
        DC_scores_list = []
        Match_loss = 0.0
        CLS_loss = 0.0
        DC_loss = 0.0
        for i, props in enumerate(proposals_t):
            num_proposals = num_instances[i]
            len_text_tokens = num_tokens[i] - 1
            box_tokens = props
            text_tokens = text_embeddings[i][:len_text_tokens]  # word embeddings from BERT (start with [CLS], without [SEP])
            # print(text_embeddings[i].shape)

            # Visual grouding pass
            global_box_token = box_tokens.mean(dim=0, keepdim=True)
            global_visual_cue = text_tokens + global_box_token
            VG_tokens = torch.cat((box_tokens, global_visual_cue), dim=0)
            assert VG_tokens.size() == (num_proposals + len_text_tokens, self.dim_model)
            output_VG_tokens = self.transformer_encoder(VG_tokens)
            output_box_tokens = output_VG_tokens[:num_proposals]
            assert output_box_tokens.size() == (num_proposals, self.dim_model)
            Match_scores = (self.grdhead(output_box_tokens)).flatten()
            Match_scores_list.append(Match_scores)
            assert Match_scores.size() == (num_proposals,)
            # Compute Matching loss
            Match_targets = torch.zeros((Match_scores.shape))
            num_target_proposals = best_proposals[i].size()[0]
            
            for p in best_proposals[i]:
                Match_targets[p] = 1.0 / num_target_proposals
            scene_loss = self.loss_criterion(Match_scores, Match_targets.to("cuda"))
            # print(scene_loss, best_proposals[i], torch.argmax(Match_scores))
            Match_loss += scene_loss
            # Match_loss += self.loss_criterion(Match_scores, Match_targets.to("cuda"))
            # Compute CLS loss
            encoded_cls_token = output_VG_tokens[num_proposals]
            CLS_scores = self.clshead(encoded_cls_token)
            CLS_scores_list.append(CLS_scores)
            scene_loss = self.loss_criterion(CLS_scores, target_classes[i])
            CLS_loss += scene_loss/num_proposals
            # CLS_loss += self.loss_criterion(CLS_scores, target_classes[i])
            # Dense Captioning pass
            for target_proposal in best_proposals[i]:
                target_box_token = box_tokens[target_proposal]
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
                DC_scores_list.append(DC_scores)
                assert DC_scores.size() == (len_text_tokens, self.size_vocab)
                # Compute DC loss
                DC_loss += self.loss_criterion(DC_scores, target_word_ids[i][1:(len_text_tokens+1)])
        
        Match_loss /= num_scenes
        CLS_loss /= num_scenes
        DC_loss /= num_scenes
     
        return {"Match_scores": Match_scores_list, "Match_loss": Match_loss,
                "CLS_scores": CLS_scores_list, "CLS_loss": CLS_loss,
                "DC_scores": DC_scores_list, "DC_loss": DC_loss}
    
    
    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.transformer_encoder.weight, -initrange, initrange)
    #     nn.init.uniform_(self.grdhead.weight, -initrange, initrange)
    #     nn.init.uniform_(self.caphead.weight, -initrange, initrange)
    #     nn.init.uniform_(self.clshead.weight, -initrange, initrange)
    #     nn.init.zeros_(self.grdhead.bias)
    #     nn.init.zeros_(self.caphead.bias)
    #     nn.init.zeros_(self.clshead.bias)
    

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