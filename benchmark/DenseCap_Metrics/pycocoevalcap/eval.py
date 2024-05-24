from .bleu.bleu import Bleu
from .cider.cider import Cider
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .spice.spice import Spice
from .wmd.wmd import WMD

def eval(gts,res):
    scorer = Bleu(n=4)
    s1, _ = scorer.compute_score(gts, res)
    
    scorer = Cider()
    s2, _ = scorer.compute_score(gts, res)

    scorer = Meteor()
    s3, _ = scorer.compute_score(gts, res)

    scorer = Rouge()
    s4, _ = scorer.compute_score(gts, res)

    scorer = Spice()
    s5, _ = scorer.compute_score(gts, res)

    scorer = WMD()
    s6, _ = scorer.compute_score(gts, res)

    return {'bleu':s1,'cider':s2,'meteor':s3,'rouge':s4,'spice':s5,'wmd':s6}

def get_bleu(gts,res):
    scorer = Bleu(n=4)
    s, _ = scorer.compute_score(gts, res)
    return s

def get_meteor(gts, res):
    scorer = Meteor()
    s, _ = scorer.compute_score(gts, res)
    return s

def get_cider(gts, res):
    scorer = Cider()
    s, _ = scorer.compute_score(gts, res)
    return s

def get_rouge(gts, res):
    scorer = Rouge()
    s, _ = scorer.compute_score(gts, res)
    return s


def get_spice(gts, res):
    scorer = Spice()
    s, _ = scorer.compute_score(gts, res)
    return s


def get_wmd(gts, res):
    scorer = WMD()
    s, _ = scorer.compute_score(gts, res)
    return s
