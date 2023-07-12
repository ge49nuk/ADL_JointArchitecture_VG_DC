import torch
import torch.nn as nn
import numpy as np
import random
from transformers import BertTokenizer, BertModel


class Transformer_Light(nn.Module):
    def __init__(self, dim_model=900, dim_ptfeats=1536, dim_wdfeats=768, max_text_len=134, num_cls=18, size_vocab=30522, dropout_p=0.2, nhead=1, nlayers=3):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.dim_ptfeats = dim_ptfeats
        self.dim_wdfeats = dim_wdfeats
        self.max_text_len = max_text_len
        self.size_vocab = size_vocab
        self.num_cls = num_cls
        
        # Transform point and word to have dimension=dim_model
        self.instance_to_model = nn.Sequential(
            nn.Linear(self.dim_ptfeats, 1024),
            nn.Linear(1024, self.dim_model)
        )
        self.word_to_model = nn.Sequential(
            nn.Linear(self.dim_wdfeats, self.dim_model)
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_model, nhead=nhead, dropout=dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        # Decoder layer: One score for each box token
        self.grdhead = nn.Sequential(
            nn.Linear(self.dim_model, 256),
            nn.Linear(256, 1)
        )
        # Decoder layer: (#Vocab) scores for each text token
        self.caphead = nn.Sequential(
            nn.Linear(self.dim_model, self.size_vocab)
        )
        # Decoder layer: (#Classes) scores for each [CLS] token
        self.clshead = nn.Sequential(
            nn.Linear(self.dim_model, 256),
            nn.Linear(256, self.num_cls)
        )

        # Define loss criterion
        self.loss_criterion_VG = nn.CrossEntropyLoss()
        self.loss_criterion_DC = nn.CrossEntropyLoss(label_smoothing=0.1)
        # self.loss_criterion_bce = nn.BCEWithLogitsLoss()
        

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased").to('cuda')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

        self.disturb = False

    def forward(self, data_dict):
        instances = data_dict['instances']
        scene_splits = data_dict['scene_splits']
        target_proposals = data_dict['target_proposals']
        target_proposal_splits = data_dict['target_proposal_splits']
        text_embeddings = data_dict['text_embeddings'].detach().clone()
        target_word_ids = data_dict['target_word_ids']
        num_tokens = data_dict['num_tokens']
        target_classes = data_dict['target_classes']

        # Transform point features and text embedding to have dim_model
        instances = self.instance_to_model(instances)
        text_embeddings = self.word_to_model(text_embeddings)

        # Get scenes:
        scenes = torch.tensor_split(instances, scene_splits[1:-1], dim=0)
        # Get target_proposals
        best_proposals = torch.tensor_split(target_proposals, target_proposal_splits[1:-1], dim=0)

        num_scenes = len(scenes) # = batch size
        num_instances = torch.diff(torch.tensor(scene_splits)) # Number of instances in each scene

        Match_scores_list = []
        CLS_scores_list = []
        DC_scores_list = []
        Match_loss = 0.0
        CLS_loss = 0.0
        DC_loss = 0.0

        for i, scene in enumerate(scenes):
            num_proposals = num_instances[i]
            len_text_tokens = num_tokens[i] - 1
            box_tokens = scene
            text_tokens = text_embeddings[i][:len_text_tokens]  # word embeddings from BERT (start with [CLS], without [SEP])

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
            Match_loss += self.loss_criterion_VG(Match_scores, Match_targets.to("cuda"))
            # Compute CLS loss
            encoded_cls_token = output_VG_tokens[num_proposals]
            CLS_scores = self.clshead(encoded_cls_token)
            CLS_scores_list.append(CLS_scores)
            CLS_loss += self.loss_criterion_VG(CLS_scores, target_classes[i])

            # Dense Captioning pass
            gt_ids = target_word_ids[i][1:(len_text_tokens+1)]
            if gt_ids[1] == 2003: # Only train with e.g. "this is ..." or "it is ..." etc.
                for target_proposal in best_proposals[i]:
                    target_box_token = box_tokens[target_proposal]
                    target_box_token = target_box_token.view(1, self.dim_model)

                    if self.disturb:
                        # Disturb the input and feed the transformer again
                        maxlen = min(len_text_tokens, 28)
                        replaced_idx = random.randint(1, maxlen-2)
                        input_len = replaced_idx + 1

                        input_text_ids = target_word_ids[i][:input_len]
                        input_text_embs = self.word_to_model(self.bert(torch.stack([input_text_ids]))[0])
                        input_text_tokens = input_text_embs[0]
                        captioning_cue = input_text_tokens + target_box_token
                        assert captioning_cue.size() == (input_len, self.dim_model)
                        DC_tokens = torch.cat((box_tokens, captioning_cue), dim=0)
                        assert DC_tokens.size() == (num_proposals + input_len, self.dim_model)
                        mask = self.get_seq2seq_mask(num_proposals, input_len)
                        output_DC_tokens = self.transformer_encoder(DC_tokens, mask.to("cuda"))
                        output_text_tokens = output_DC_tokens[num_proposals:]
                        assert output_text_tokens.size() == (input_len, self.dim_model)
                        DC_scores = self.caphead(output_text_tokens)

                        empty_cap = np.array(["[PAD]"] * 134, dtype=np.dtype('U15'))
                        empty_cap[0] = "[CLS]"
                        empty_cap = self.tokenizer.convert_tokens_to_ids(empty_cap)
                        empty_cap = torch.tensor(empty_cap).to('cuda')

                        predicted_ids = DC_scores.argmax(dim=-1)
                        new_input_ids = empty_cap[:input_len]
                        new_input_ids[:input_len] = target_word_ids[i][:input_len] # Copy GT words
                        new_input_ids[replaced_idx] = predicted_ids[replaced_idx - 1] # Replace the last GT word with predicted word
                        new_input_embs = self.word_to_model(self.bert(torch.stack([new_input_ids]))[0])
                        new_text_tokens = new_input_embs[0]
                        new_captioning_cue = new_text_tokens + target_box_token
                        assert new_captioning_cue.size() == (input_len, self.dim_model)
                        new_DC_tokens = torch.cat((box_tokens, new_captioning_cue), dim=0)
                        assert new_DC_tokens.size() == (num_proposals + input_len, self.dim_model)
                        mask = self.get_seq2seq_mask(num_proposals, input_len)
                        new_output_DC_tokens = self.transformer_encoder(new_DC_tokens, mask.to("cuda"))
                        new_output_text_tokens = new_output_DC_tokens[num_proposals:]
                        assert new_output_text_tokens.size() == (input_len, self.dim_model)
                        DC_scores = self.caphead(new_output_text_tokens)
                        assert DC_scores.size() == (input_len, self.size_vocab)
                        DC_scores_list.append(DC_scores)
                        # Compute CE loss
                        assert DC_scores.size() == (input_len, self.size_vocab)
                        DC_loss += self.loss_criterion_DC(DC_scores, target_word_ids[i][1:(input_len+1)])
                    else:
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
                        # Compute CE loss
                        assert DC_scores.size() == (len_text_tokens, self.size_vocab)
                        DC_loss += self.loss_criterion_DC(DC_scores, target_word_ids[i][1:(len_text_tokens+1)])
        
        Match_loss /= num_scenes
        CLS_loss /= num_scenes
        DC_loss /= num_scenes
     
        return {"Match_scores": Match_scores_list, "Match_loss": Match_loss,
                "CLS_scores": CLS_scores_list, "CLS_loss": CLS_loss,
                "DC_scores": DC_scores_list, "DC_loss": DC_loss}
    

    # Only called for testing
    def feed_VG(self, data_dict):
        instances = data_dict['instances']
        scene_splits = data_dict['scene_splits']
        text_embeddings = data_dict['text_embeddings'].detach().clone()
        num_tokens = data_dict['num_tokens']

        # Transform point features and text embedding to have dim_model
        instances = self.instance_to_model(instances)
        text_embeddings = self.word_to_model(text_embeddings)

        # Get scenes:
        scenes = torch.tensor_split(instances, scene_splits[1:-1], dim=0)

        num_instances = torch.diff(torch.tensor(scene_splits)) # Number of instances in each scene

        Match_scores_list = []
        for i, scene in enumerate(scenes):
            num_proposals = num_instances[i]
            len_text_tokens = num_tokens[i] - 1
            box_tokens = scene
            text_tokens = text_embeddings[i][:len_text_tokens]  # word embeddings from BERT (start with [CLS], without [SEP])

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
                
        return Match_scores_list
    

    # Only called for testing
    def feed_DC(self, data_dict, idx):
        instances = data_dict['instances']
        scene_splits = data_dict['scene_splits']
        target_proposals = data_dict['target_proposals']
        target_proposal_splits = data_dict['target_proposal_splits']
        text_embeddings = data_dict['text_embeddings'].detach().clone()
        num_tokens = data_dict['num_tokens']

        # Transform point features and text embedding to have dim_model
        instances = self.instance_to_model(instances)
        text_embeddings = self.word_to_model(text_embeddings)

        # Get scenes:
        scenes = torch.tensor_split(instances, scene_splits[1:-1], dim=0)
        # Get target_proposals
        best_proposals = torch.tensor_split(target_proposals, target_proposal_splits[1:-1], dim=0)

        num_instances = torch.diff(torch.tensor(scene_splits)) # Number of instances in each scene

        num_proposals = num_instances[idx]
        len_text_tokens = num_tokens[idx] - 1
        box_tokens = scenes[idx]
        text_tokens = text_embeddings[idx][:len_text_tokens]  # word embeddings from BERT (start with [CLS], without [SEP])
        
        # Dense Captioning pass
        target_proposal = best_proposals[idx][0] # Use only the first best proposal
        target_box_token = box_tokens[target_proposal]
        target_box_token = target_box_token.view(1, self.dim_model)

        # #############################################################################
        # ############## Only for inference using teacher forcing scheme ##############
        # caption_ids = []
        # captioning_cue = text_tokens + target_box_token
        # assert captioning_cue.size() == (len_text_tokens, self.dim_model)
        # DC_tokens = torch.cat((box_tokens, captioning_cue), dim=0)
        # assert DC_tokens.size() == (num_proposals + len_text_tokens, self.dim_model)
        # mask = self.get_seq2seq_mask(num_proposals, len_text_tokens)
        # mask = mask.to("cuda")

        # # input_cap = np.array(["[PAD]"] * len_text_tokens, dtype=np.dtype('U15'))
        # # input_cap[0] = "[CLS]"
        # # input_cap = self.tokenizer.convert_tokens_to_ids(input_cap)
        # # input_cap_ids = torch.tensor(input_cap).to('cuda')
        # # for l in range(len_text_tokens-1):
        # #     text_tokens = self.word_to_model(self.bert(torch.stack([input_cap_ids]))[0])[0]
        # #     captioning_cue = text_tokens + target_box_token
        # #     DC_tokens = torch.cat((box_tokens, captioning_cue), dim=0)
        # #     output_DC_tokens = self.transformer_encoder(DC_tokens, mask)
        # #     output_text_tokens = output_DC_tokens[num_proposals:]
        # #     DC_scores = self.caphead(output_text_tokens)
        # #     input_cap_ids[l+1] = DC_scores.argmax(dim=-1)[l]
            
        # #     next_word_id = DC_scores.argmax(dim=-1)[l]
        # #     caption_ids.append(next_word_id)

        # for l in range(len_text_tokens-1):
        #     output_DC_tokens = self.transformer_encoder(DC_tokens, mask)
        #     output_text_tokens = output_DC_tokens[num_proposals:]
        #     assert output_text_tokens.size() == (len_text_tokens, self.dim_model)
        #     DC_scores = self.caphead(output_text_tokens)
            
        #     next_word_id = DC_scores.argmax(dim=-1)[l]
        #     caption_ids.append(next_word_id)
        
        # predicted_tokens = self.tokenizer.convert_ids_to_tokens(caption_ids)
        # predicted_str = self.tokenizer.convert_tokens_to_string(predicted_tokens)
        # predicted_str = predicted_str.replace('[SEP]', '.')
        # predicted_str = predicted_str.replace(' \' s ', ' \'s ')
        # candidate_descr = "sos " + predicted_str + " eos"
        # caption = candidate_descr
        # ############## Only for inference using teacher forcing scheme ##############
        # #############################################################################

        empty_cap = np.array(["[PAD]"] * 134, dtype=np.dtype('U15'))
        empty_cap[0] = "[CLS]"
        empty_cap = self.tokenizer.convert_tokens_to_ids(empty_cap)
        empty_cap = torch.tensor(empty_cap).to('cuda')
        best_pasts = [empty_cap] # Tracking the best past word sequences (in word ids)
        best_past_embs = self.word_to_model(self.bert(torch.stack([empty_cap]))[0])
        best_scores = [0] # Scores of best past sequences

        for l in range(self.max_text_len-1): # Generate words one by one
            tmp_best_pasts = []
            tmp_best_scores = []
            for j, past in enumerate(best_pasts): # past: (134, id)
                model_embeddings = best_past_embs[j]
                len_text_tokens = l + 1 # Number of words to generate
                text_tokens = model_embeddings[:len_text_tokens]  # word embeddings from BERT (start with [CLS], without [SEP])

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

                # Select words with highest scores
                next_word_scores = DC_scores[l]
                
                last_id = past[l]
                next_word_scores[last_id] = 0 # Not predict the same last word
                if l > 0:
                    second_last_id = past[l-1]
                    next_word_scores[second_last_id] = 0 # Not predict the same second last word
                    if l > 1:
                        third_last_id = past[l-2]
                        next_word_scores[third_last_id] = 0 # Not predict the same third last word
                

                sm = nn.Softmax(dim=-1)
                next_word_scores = sm(next_word_scores)
                
                _, top_ids = torch.topk(next_word_scores, 20, sorted=False) # Select the best next words
                top_scores = next_word_scores[top_ids]
                top_scores = top_scores + 0.3 * best_scores[j] # Accumulate past score
                new_best_pasts = torch.stack([past for k in range(20)])
                new_best_pasts[:, l+1] = top_ids

                tmp_best_pasts.append(new_best_pasts)
                tmp_best_scores.append(top_scores)

            # Concat the best past sequences
            new_best_pasts = torch.cat(tmp_best_pasts, dim=0)
            best_scores = torch.cat(tmp_best_scores, dim=0)

            best_idx = best_scores.argmax(dim=-1)
            if l == 20 or new_best_pasts[best_idx, l+1] == 102: # 102: [SEP]
                predicted_ids = new_best_pasts[best_idx][1:l+1]
                predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_ids)
                predicted_str = self.tokenizer.convert_tokens_to_string(predicted_tokens)
                predicted_str = predicted_str.replace('[SEP]', '.')
                predicted_str = predicted_str.replace(' \' s ', ' \'s ')
                candidate_descr = "sos " + predicted_str + " eos"
                caption = candidate_descr
                break
            num_best = min(new_best_pasts.shape[0], 64)
            _, best_indices = torch.topk(best_scores, num_best, sorted=False) # Select the best sequences
            best_pasts = new_best_pasts[best_indices] 
            best_past_embs = self.bert(best_pasts)[0]
            best_past_embs = self.word_to_model(best_past_embs)
                
        return caption
    

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
    
    def start_disturb(self):
        self.disturb = True
    
    def disable_disturb(self):
        self.disturb = False
