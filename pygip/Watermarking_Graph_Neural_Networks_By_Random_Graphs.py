from pygip.protect import *
from pygip.protect.Defense import Watermark_sage

attack_name = int(input("Please choose the number:\n1.ModelExtractionAttack0\n2.ModelExtractionAttack1\n3.ModelExtractionAttack2\n4.ModelExtractionAttack3\n5.ModelExtractionAttack4\n6.ModelExtractionAttack5\n"))
dataset_name = int(input("Please choose the number:\n1.Cora\n2.Citeseer\n3.PubMed\n"))
if (dataset_name == 1):
    defense = Watermark_sage(Cora(),0.25)
    defense.watermark_attack(Cora(), attack_name, dataset_name)
elif (dataset_name == 2):
    defense = Watermark_sage(Citeseer("dgl","./"),0.25)
    defense.watermark_attack(Citeseer("dgl","./"), attack_name, dataset_name)
elif (dataset_name == 3):
    defense = Watermark_sage(PubMed("dgl","./"),0.25)
    defense.watermark_attack(PubMed("dgl","./"), attack_name, dataset_name)
