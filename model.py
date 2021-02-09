import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel
import dgl
import dgl.nn.pytorch as dglnn

from utils import MODEL_CLASSES


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reinitialize learnable parameters. '''
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attntion)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class BiAttention(nn.Module):

    '''
    bi-attention layer (Seo et al., 2017)

    Placed on top of pre-trained encoder (e.g., RoBERTa, BERT) to fuse information from both the query and the context
    '''

    def __init__(self, args, hidden_size, dropout=0.1):
        super(BiAttention, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.simW = nn.Linear(self.hidden_size * 3, 1, bias=False)  # nn.Linear(hidden_size * 6, 1, bias=False)

    def forward(self, encoder_out, question_ends):
        query = encoder_out[:, : question_ends + 1, :]
        context = encoder_out[:, question_ends + 1 :, :]
        
        # print("Query hidden : ", query.shape)
        # print("Context hidden : ", context.shape)

        T = context.size(1)
        J = query.size(1)

        sim_shape = (self.args.train_batch_size, T, J, self.hidden_size)  # (self.args.train_batch_size, T, J, 2 * self.hidden_size)
        context_embed = context.unsqueeze(2)  # (N, T, 1, 2d)
        context_embed = context_embed.expand(sim_shape)  # (N, T, J, 2d)
        query_embed = query.unsqueeze(1)  # (N, 1, J, 2d)
        query_embed = query_embed.expand(sim_shape)  # (N, T, J, 2d)
        elemwise_mul = torch.mul(context_embed, query_embed)  # (N, T, J, 2d)
        concat_sim_input = torch.cat((context_embed, query_embed, elemwise_mul), 3)  # (N, T, J, 6d) - [h ; u ; h o u]

        S = self.simW(concat_sim_input).view(self.args.train_batch_size, T, J)  # (N, T, J)
        # print("S : ", S.shape)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), query)  # bmm( (N, T, J), (N, J, 2d) ) = (N, T, 2d)
        # Query2Context
        b = F.softmax(torch.max(S, 2)[0], dim=-1)  # Apply the "maximum function (max_col) across the column"
        q2c = torch.bmm(b.unsqueeze(1), context)
        q2c = q2c.repeat(1, T, 1)  # (N, T, 2d) - tiled `T` times

        # G: query-aware representation of each context word
        G = torch.cat((context, c2q, context.mul(c2q), context.mul(q2c)), -1)  # (N, T, 8d)
        # print("G: ", G.shape)

        return query, G


class ContextEncoder(BertPreTrainedModel):  # TODO: Need RoBERTa-Large
    def __init__(self, args, config):
        super().__init__(config)
        self.args = args
        self.config = config
        _, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.encoder = self.model_class.from_pretrained(self.args.model_name_or_path, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        encoded_out = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return encoded_out


class GatedAttention(nn.Module):
    def __init__(self, args, config):
        super(GatedAttention, self).__init__()
        self.args = args
        self.config = config
        self.context_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2, bias=True)
        self.graph_mlp = nn.Linear(self.config.hidden_size, self.config.hidden_size * 2, bias=True)
        self.gate_linear = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size * 4, bias=True)
    
    def forward(self, context_rep, graph_rep):
        '''
        context_rep : M (in the HGN paper)
        graph_rep : H' (in the HGN paper)
        context_initial : C (in the HGN paper)

        '''
        ctx_dot = F.relu(self.context_mlp(context_rep))
        graph_dot = F.relu(self.graph_mlp(graph_rep))

        # print("ctx_dot (shape): ", ctx_dot.shape)
        # print("graph_dot (shape): ", graph_dot.shape)

        C_merged = torch.bmm(ctx_dot, graph_dot.permute(0, 2, 1).contiguous())
        ctx2nodes = F.softmax(C_merged, dim=-1)  # Context-to-node (attention)
        # print("C_merged: ", C_merged.shape)
        # print("ctx2nodes: ", ctx2nodes.shape)
        # print("graph_rep: ", graph_rep.shape)

        H_attn = torch.bmm(ctx2nodes, graph_rep)  # (B, N, M) x (B, M, H) = (B, N, H)
        ctx_graph = torch.cat((context_rep, H_attn), dim=-1)
        gated1 = F.sigmoid(self.gate_linear(ctx_graph))  # (B, N, 2H) ; (B, N, H) = (B, N, 3H)
        gated2 = F.tanh(self.gate_linear(ctx_graph))  # (B, N, 2H) ; (B, N, H) = (B, N, 3H)
        G = gated1 * gated2
        # print("gated1: ", gated1.shape)
        # print("gated2: ", gated2.shape)
        # print("G: ", G.shape)
        return G



class NumericHGN(nn.Module):
    def __init__(self, args, config):
        super(NumericHGN, self).__init__()
        self.args = args
        self.config = config
        self.encoder = ContextEncoder(self.args, config)
        self.bi_attn = BiAttention(args, self.config.hidden_size)
        self.bi_attn_linear = nn.Linear(self.config.hidden_size * 4, self.config.hidden_size)
        self.bi_lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, bidirectional=True)
        self.para_node_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.sent_node_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.ent_node_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        
        # https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.HeteroGraphConv
        self.gat = dglnn.HeteroGraphConv({
            "ps" : dglnn.GATConv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            "se" : dglnn.GATConv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            # TODO: Need (i) bi-directional edges and (ii) more edge types (e.g., question-paragraph, paragraph-paragraph, etc.)
        }, aggregate='sum')  # TODO: May need to change aggregate function (test it!) - ‘sum’, ‘max’, ‘min’, ‘mean’, ‘stack’.

        self.gated_attn = GatedAttention(self.args, self.config)

        self.para_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, args.num_paragraphs))
        self.sent_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, args.num_sentences))
        self.ent_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, args.num_entities))
        self.start_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, args.max_seq_len))
        self.end_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, args.max_seq_len))
        self.answer_type_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, 3))

    def forward(self, input_ids, attention_mask, token_type_ids, labels, graph_out, question_ends):
        '''
        Args

        input_ids : [Question ; Context] tokenized by tokenizer (e.g., BertTokenizer)
        attention_mask : attention_mask
        token_type_ids : token_type_ids
        labels : labels in tuple: (para_lbl, sent_lbl, span_idx, answer_type_lbl)
        graph_out = (g, node_idx, span_dict)
        g : dgl.graph - the hierarchical graph neural network
        question_ends : question end index (e.g., first occurrence of [SEP] token in [Q;C] input)

        '''
        para_lbl, sent_lbl, answer_type_lbl, span_idx = labels
        g, node_idx, span_dict = graph_out

        encoder_out = self.encoder(input_ids, attention_mask, token_type_ids)
        seq_out = encoder_out[0]
        Q, C = self.bi_attn(seq_out, question_ends)
        C = self.bi_attn_linear(C)
        C = C.permute(1, 0, 2)  # Change dimension from `batch-first` to `sequence-first`
        h0 = torch.randn(2, self.args.train_batch_size, self.config.hidden_size).to(C.device)  # TODO: Think about new ways to initialize self.bi_lstm's h0 and c0
        c0 = torch.randn(2, self.args.train_batch_size, self.config.hidden_size).to(C.device)
        print("Q: ", Q.shape)
        print("C: ", C.shape)
        M, (hn, cn) = self.bi_lstm(C, (h0, c0))
    
        # Extract from `M` with the given spans of paragraphs, sentences and entities
        # (i) The hidden state of the backward LSTM at the start position
        # (ii) And the hidden state of the forward LSTM at the end position
        print("M (shape) : ", M.shape)
        print("h_n (shape) : ", hn.shape)
        print("c_n (shape) : ", cn.shape)

        # TODO: Need to extract paragraph, sentence and entity representations from `M`
        g, node_idx, span_dict = graph_out
        question_idx = span_dict['question']
        para_idx = span_dict['paragraph']
        sent_idx = span_dict['sentence']
        ent_idx = span_dict['entity']

        print("Question_idx: ", question_idx)
        print("Paragraph_idx: ", para_idx)
        print("Sentence_idx: ", sent_idx)
        print("Entity_idx: ", ent_idx)

        print("g : ", g)

        for q_idx in question_idx:
            start_q, _ = q_idx
            # assert end_q == question_ends  # TODO: Rebuild cached_data later on to fix this issue
            end_q = question_ends.item()

        # Extract spans from `M`
        M_temp = M.squeeze(-2)  # TODO: This only works for `batch_size = 1` case
        para_node_input = torch.zeros(len(para_idx), self.config.hidden_size * 2).to(M.device)
        sent_node_input = torch.zeros(len(sent_idx), self.config.hidden_size * 2).to(M.device)
        ent_node_input = torch.zeros(len(ent_idx), self.config.hidden_size * 2).to(M.device)
        for i, p_span in enumerate(para_idx):
            para_node_input[i] = torch.cat((M_temp[p_span[0]][self.config.hidden_size:], M_temp[p_span[1]][:self.config.hidden_size]))
        for i, s_span in enumerate(sent_idx):
            sent_node_input[i] = torch.cat((M_temp[s_span[0]][self.config.hidden_size:], M_temp[s_span[1]][:self.config.hidden_size]))
        for i, e_span in enumerate(ent_idx):
            ent_node_input[i] = torch.cat((M_temp[e_span[0]][self.config.hidden_size:], M_temp[e_span[1]][:self.config.hidden_size]))
        
        # Max-pooling `Q` for question representation
        Q_temp = Q.squeeze(0)
        q, _ = torch.max(Q_temp, dim=0)
        print("q (shape): ", q.shape)

        # Construct node representations
        para_rep = self.para_node_mlp(para_node_input)
        sent_rep = self.sent_node_mlp(sent_node_input)
        ent_rep = self.ent_node_mlp(ent_node_input)

        print("para_initial_embed: ", para_rep.shape)
        print("sent_initial_embed: ", sent_rep.shape)
        print("ent_initial_embed: ", ent_rep.shape)

        # Initialize the paragraph, sentence and entity nodes with the node representations
        in_feats = {"question": q, "paragraph": para_rep, "sentence": sent_rep, "entity": ent_rep}
        g_out = self.gat(g, (in_feats, in_feats))  # TODO: Why input tuple (이 부분 dgl documentation에 존재하지 않음 - Open an Issue on their official GitHub)
        graph_rep = []
        for k, v in g_out.items():
            if len(graph_rep) == 0:
                graph_rep = v
            else:
                graph_rep = torch.cat((graph_rep, v), dim=0)

        print("graph_rep (shape): ", graph_rep.shape)
        print("g (etype): ", g.etype)
        print("g_out: ", g_out)
        print("g_out (length): ", len(g_out))
        print("g_out (keys): ", g_out.keys())
        print("g_out - entities: ", g_out["entity"].shape)
        print("g_out - sentence: ", g_out["sentence"].shape)

        # Gated Attention ; TODO: Implement a separate class for gated attention mechanism
        M_perm = M.permute(1, 0, 2)
        graph_rep_perm = graph_rep.permute(1, 0, 2)
        gated_rep = self.gated_attn(M_perm, graph_rep_perm)



class BiDAF(nn.Module):
    def __init__(self, args):
        super(BiDAF, self).__init__()
        self.embd_size = args.w_embd_size
        self.d = self.embd_size * 2 # word_embedding + char_embedding
        # self.d = self.embd_size # only word_embedding

        self.char_embd_net = CharEmbedding(args)
        self.word_embd_net = WordEmbedding(args)
        self.highway_net = Highway(self.d)
        self.ctx_embd_layer = nn.GRU(self.d, self.d, bidirectional=True, dropout=0.2, batch_first=True)

        self.W = nn.Linear(6*self.d, 1, bias=False)

        self.modeling_layer = nn.GRU(8*self.d, self.d, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)

        self.p1_layer = nn.Linear(10*self.d, 1, bias=False)
        self.p2_lstm_layer = nn.GRU(2*self.d, self.d, bidirectional=True, dropout=0.2, batch_first=True)
        self.p2_layer = nn.Linear(10*self.d, 1)

    def build_contextual_embd(self, x_c, x_w):
        # 1. Caracter Embedding Layer
        char_embd = self.char_embd_net(x_c) # (N, seq_len, embd_size)
        # 2. Word Embedding Layer
        word_embd = self.word_embd_net(x_w) # (N, seq_len, embd_size)
        # Highway Networks for 1. and 2.
        embd = torch.cat((char_embd, word_embd), 2) # (N, seq_len, d=embd_size*2)
        embd = self.highway_net(embd) # (N, seq_len, d=embd_size*2)

        # 3. Contextual  Embedding Layer
        ctx_embd_out, _h = self.ctx_embd_layer(embd)
        return ctx_embd_out

    def forward(self, ctx_w, ctx_c, query_w, query_c):
        batch_size = ctx_w.size(0)
        T = ctx_w.size(1)   # context sentence length (word level)
        J = query_w.size(1) # query sentence length   (word level)

        # 1. Caracter Embedding Layer
        # 2. Word Embedding Layer
        # 3. Contextual  Embedding Layer
        embd_context = self.build_contextual_embd(ctx_c, ctx_w)     # (N, T, 2d)
        embd_query   = self.build_contextual_embd(query_c, query_w) # (N, J, 2d)

        # 4. Attention Flow Layer
        # Make a similarity matrix
        shape = (batch_size, T, J, 2*self.d)            # (N, T, J, 2d)
        embd_context_ex = embd_context.unsqueeze(2)     # (N, T, 1, 2d)
        embd_context_ex = embd_context_ex.expand(shape) # (N, T, J, 2d)
        embd_query_ex = embd_query.unsqueeze(1)         # (N, 1, J, 2d)
        embd_query_ex = embd_query_ex.expand(shape)     # (N, T, J, 2d)
        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex) # (N, T, J, 2d)
        cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3) # (N, T, J, 6d), [h;u;h◦u]
        S = self.W(cat_data).view(batch_size, T, J) # (N, T, J)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), embd_query) # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        # Query2Context
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1) # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), embd_context) # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, T, 1) # (N, T, 2d), tiled T times

        # G: query aware representation of each context word
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2) # (N, T, 8d)

        # 5. Modeling Layer
        M, _h = self.modeling_layer(G) # M: (N, T, 2d)

        # 6. Output Layer
        G_M = torch.cat((G, M), 2) # (N, T, 10d)
        p1 = F.softmax(self.p1_layer(G_M).squeeze(), dim=-1) # (N, T)

        M2, _ = self.p2_lstm_layer(M) # (N, T, 2d)
        G_M2 = torch.cat((G, M2), 2) # (N, T, 10d)
        p2 = F.softmax(self.p2_layer(G_M2).squeeze(), dim=-1) # (N, T)

        return p1, p2