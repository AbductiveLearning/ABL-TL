import torch
from itertools import product


class KnowledgeBase:
    def __init__(self, name='ConjEq'):
        self.name = name
        self.rules = {}
        if name == 'ConjEq':
            self.rlen = 3
            self.tnum = 1
            self.snum = 2
            self.generate_conj_eq()
        elif name == 'Conjunction':
            self.rlen = 2
            self.tnum = 2
            self.snum = 2
            self.generate_conjunction()
        elif name == 'TripletComp':
            self.rlen = 3
            self.tnum = 1
            self.snum = 2
            self.generate_triplet_comp()
        elif name == 'Addition':
            self.rlen = 2
            self.tnum = 19
            self.snum = 10
            self.generate_addition()
        self.symbol_set = torch.tensor(range(self.snum))
        self.num_rules = {}
        self.total_rules = 0
        for k, v in self.rules.items():
            self.num_rules[k] = len(v)
            self.total_rules += len(v)
        rank, matrix = self.rank_criterion()
        self.rank = rank
        self.matrix = matrix
        
    def generate_conj_eq(self):
        conj = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 1),
        ]
        self.rules[0] = torch.tensor(conj)
    
    def generate_conjunction(self):
        conj0 = [
            (0, 0),
            (0, 1),
            (1, 0),
        ]
        conj1 = [
            (1, 1),
        ]
        self.rules[0] = torch.tensor(conj0)
        self.rules[1] = torch.tensor(conj1)
        
    def generate_triplet_comp(self):
        tri_comp = [
            (0, 0, 1),
            (1, 1, 0),
        ]
        self.rules[0] = torch.tensor(tri_comp)
    
    def generate_addition(self):
        digits = list(product(list(range(self.snum)), repeat=self.rlen))
        digits = torch.tensor(digits)
        targets = digits.sum(dim=1)
        for t in targets.unique().tolist():
            self.rules[t] = digits[targets == t]
    
    def rank_criterion(self):
        py = torch.zeros(self.snum)
        for j in range(self.snum):
            count, total = 0, 0
            for t, trules in self.rules.items():
                count += (trules == j).sum()
                total += int(torch.prod(torch.tensor(trules.shape)))
            py[j] = count / total

        matrix = []
        for t, trules in self.rules.items():
            pY = 1 / self.total_rules
            pt = self.num_rules[t] / self.total_rules
            pt_given_Y = 1    # For Y in S(t), otherwise ptau_given_Y = 0
            pY_given_t = pt_given_Y * pY / pt
            rlen = trules.shape[1]
            piota = 1 / rlen
            py_given_tau_iota = torch.zeros(self.snum, rlen)
            ptau_iota_given_y = torch.zeros(self.snum, rlen)
            for j in range(self.snum):
                for k in range(rlen):
                    prob = 0
                    for Y in trules:
                        prob += (Y[k] == j) * pY_given_t
                    py_given_tau_iota[j][k] = prob
                    ptau_iota_given_y[j][k] = prob * pt * piota / py[j]
            matrix.append(ptau_iota_given_y)
            rank = torch.linalg.matrix_rank(torch.cat(matrix, dim=1)).item()
            print('Upon concept #{}, rank is {}.'.format(t, rank))
        
        matrix = torch.cat(matrix, dim=1)
        rank = torch.linalg.matrix_rank(matrix).item()
        print('Number of classes: {}. \t Rank: {}. \n'.format(self.snum, rank))
        print()
        return rank, matrix
    
    def cuda(self):
        for k, v in self.rules.items():
            self.rules[k] = v.cuda()
        self.symbol_set = self.symbol_set.cuda()
        self.matrix = self.matrix.cuda()
    
    def cpu(self):
        for k, v in self.rules.items():
            self.rules[k] = v.cpu()
        self.symbol_set = self.symbol_set.cpu()
        self.matrix = self.matrix.cpu()
    
    def logic_forward(self, facts: torch.Tensor):
        res = -1
        for target_concept, target_facts in self.rules.items():
            if facts.shape[0] != target_facts.shape[1]:
                continue
            indicator = torch.eq(facts, target_facts).all(dim=1)
            if torch.any(indicator):
                res = target_concept
                break
        return res
    
    def __call__(self, facts):
        if len(facts.shape) == 1:
            res = self.logic_forward(facts)
            return torch.tensor(res, device=facts.device)
        elif len(facts.shape) == 2:
            res = []
            for fact in facts:
                res.append(self.logic_forward(fact))
            return torch.tensor(res, device=facts.device)

         
if __name__ == '__main__':
    from pprint import pprint
    
    name = 'TripletComp'
    kb = KnowledgeBase(name)
    pprint(kb.rules)
    
    name = 'ConjEq'
    kb = KnowledgeBase(name)
    pprint(kb.rules)

    kb.cuda()
    facts = torch.tensor([[0, 0, 1], [0, 1, 1]]).cuda()
    res = kb(facts)
    print(res)

    kb.cpu()
    facts = torch.tensor([[1, 1, 1], [0, 1, 1]]).cpu()
    res = kb(facts)
    print(res)

    kb.cuda()
    facts = torch.tensor([[0, 0, 0], [0, 1, 0]]).cuda()
    res = kb(facts)
    print(res)

    print()
    print()
    name = 'Conjunction'
    kb = KnowledgeBase(name)
    pprint(kb.rules)

    kb.cuda()
    facts = torch.tensor([[0, 0], [1, 1]]).cuda()
    res = kb(facts)
    print(res)
    kb.cpu()
    facts = torch.tensor([0, 1]).cpu()
    res = kb(facts)
    print(res)
    facts = torch.tensor([1, 1]).cpu()
    res = kb(facts)
    print(res)
    facts = torch.tensor([[1, 1]]).cpu()
    res = kb(facts)
    print(res)


    print()
    print()
    name = 'Addition'
    kb = KnowledgeBase(name)
    facts = torch.tensor([[0, 1], [0, 8], [4, 9], [3, 1]]).cpu()
    res = kb(facts)
    print(res)