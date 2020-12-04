from ltp import LTP
from ddparser import DDParser


class Base_Parse(object):
    def __init__(self):
        pass

    def get_dependency_path(self, tokens: str, ent_h: dict, ent_t: dict, word: list, head: list, deprel: list):
        """get head entity to root dependency path, and get tail entity 
        to root dependency path

        Args:
            tokens (str): token list or sentence
            ent_h (dict): entity dict like {'pos': [hpos, tpos], 'entity': Time}
            ent_t (dict): entity dict like {'pos': [hpos, tpos], 'entity': Time}
            word (list): words
            head (list): head word, word[head[i] - 1] is the head word of word[i] 
            deprel (list): dependecy relation, deprel[i] is the dependency relation of word[head[i] - 1] to word[i]

        Returns:
            ent_h_path (list): DSP path of head entity to root
            ent_t_path (list): DSP path of tail entity to root
        """
        # construct map between word and token
        token2word = {}
        word2token = {}
        tpos, tlen = 0, 0
        wpos, wlen = 0, len(word[0])
        for i in range(len(tokens)):
            if tlen >= wlen:
                wpos += 1
                wlen += len(word[wpos])
                tpos = i
            tlen += len(tokens[i])
            token2word[i] = wpos
            word2token[wpos] = tpos

        # get entity word pos
        ent_h_word_pos_1 = token2word[ent_h['pos'][0]]
        ent_h_word_pos_2 = token2word[ent_h['pos'][1] - 1]
        ent_t_word_pos_1 = token2word[ent_t['pos'][0]]
        ent_t_word_pos_2 = token2word[ent_t['pos'][1] - 1]

        # get head entity dependency path to root
        ent_h_path = [ent_h_word_pos_2]
        while head[ent_h_path[-1]] != 0:
            ent_h_path.append(head[ent_h_path[-1]] - 1)
        for i in range(len(ent_h_path)):
            ent_h_path[i] = word2token[ent_h_path[i]]
        pos = ent_h_word_pos_1
        while pos < ent_h_word_pos_2:
            if head[pos] - 1 <= ent_h_word_pos_2 and deprel[pos] == 'ATT':
                pos = head[pos] - 1
            else:
                break
        if ent_h_word_pos_1 < pos and pos == ent_h_word_pos_2:
            ent_h_path[0] = word2token[ent_h_word_pos_1]

        # get tail entity dependency path to root
        ent_t_path = [ent_t_word_pos_2]
        while head[ent_t_path[-1]] != 0:
            ent_t_path.append(head[ent_t_path[-1]] - 1)
        for i in range(len(ent_t_path)):
            ent_t_path[i] = word2token[ent_t_path[i]]
        pos = ent_t_word_pos_1
        while pos < ent_t_word_pos_2:
            if head[pos] - 1 <= ent_t_word_pos_2 and deprel[pos] == 'ATT':
                pos = head[pos] - 1
            else:
                break
        if ent_t_word_pos_1 < pos and pos == ent_t_word_pos_2:
            ent_t_path[0] = word2token[ent_t_word_pos_1]
        
        return ent_h_path, ent_t_path


class DDP_Parse(Base_Parse):
    def __init__(self):
        super(DDP_Parse, self).__init__()
        self.ddp = DDParser()
    
    def parse(self, tokens, ent_h, ent_t):
        """get head entity to root dependency path, and get tail entity 
        to root dependency path

        Args:
            tokens (list or str): token list or sentence
            ent_h (dict): entity dict like {'pos': [hpos, tpos]}
            ent_t (dict): entity dict like {'pos': [hpos, tpos]}

        Returns:
            ent_h_path (list): DSP path of head entity to root
            ent_t_path (list): DSP path of tail entity to root
        """
        if isinstance(tokens, list):
            sent = ''.join(tokens)
        else:
            sent = tokens
        parse_dict = self.ddp.parse(sent)[0] # word, head, deprel
        # print(tokens[ent_h['pos'][0]:ent_h['pos'][1]], tokens[ent_t['pos'][0]:ent_t['pos'][1]])
        word = parse_dict['word']
        head = parse_dict['head']
        deprel = parse_dict['deprel']
        ent_h_path, ent_t_path = self.get_dependency_path(tokens, ent_h, ent_t, word, head, deprel)
        
        return ent_h_path, ent_t_path


class LTP_Parse(Base_Parse):
    def __init__(self):
        super(LTP_Parse, self).__init__()
        self.ltp = LTP()
    
    def parse(self, tokens, ent_h: dict, ent_t: dict):
        """get head entity to root dependency path, and get tail entity 
        to root dependency path

        Args:
            tokens (list or str): sentence
            ent_h (dict): entity dict like {'pos': [hpos, tpos]}
            ent_t (dict): entity dict like {'pos': [hpos, tpos]}

        Returns:
            ent_h_path (list): DSP path of head entity to root
            ent_t_path (list): DSP path of tail entity to root
        """
        if isinstance(tokens, list):
            sent = ''.join(tokens)
        else:
            sent = tokens
        seg, hidden = self.ltp.seg([sent])
        parse_tuple = self.ltp.dep(hidden)[0] # word, head, deprel
        parse_tuple = list(zip(*parse_tuple)) # [(tail token id), (head token id), (DSP relation)], id start from 1
        word = seg[0]
        head = parse_tuple[1]
        deprel = parse_tuple[2]
        ent_h_path, ent_t_path = self.get_dependency_path(tokens, ent_h, ent_t, word, head, deprel)
        
        return ent_h_path, ent_t_path
        
        
        

        
        
        
        
