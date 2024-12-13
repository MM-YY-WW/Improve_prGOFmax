#!/bin/bash
exec > results.txt 2>&1
#------------------------------------------------Evaluating the original baseline performance-------------------------------------

# echo "### Evaluating the original baseline performance ###"
# python fmax_evaluation.py --ontology 'BP' --model 'DeepFri' --best_threshold 0.07
# python fmax_evaluation.py --ontology 'CC' --model 'DeepFri' --best_threshold 0.05
# python fmax_evaluation.py --ontology 'MF' --model 'DeepFri' --best_threshold 0.09

# python fmax_evaluation.py --ontology 'BP' --model 'PFresGO' --best_threshold 0.09
# python fmax_evaluation.py --ontology 'CC' --model 'PFresGO' --best_threshold 0.04
# python fmax_evaluation.py --ontology 'MF' --model 'PFresGO' --best_threshold 0.16

# python fmax_evaluation.py --ontology 'BP' --model 'HEAL' --best_threshold 0.05
# python fmax_evaluation.py --ontology 'CC' --model 'HEAL' --best_threshold 0.02
# python fmax_evaluation.py --ontology 'MF' --model 'HEAL' --best_threshold 0.07


# #------------------------------------------------Evaluating on harder test set------------------------------------------------------

# echo "### Evaluating on harder test set ###"
# python fmax_evaluation.py --ontology 'BP' --model 'DeepFri' --test_subset --best_threshold 0.07
# python fmax_evaluation.py --ontology 'CC' --model 'DeepFri' --test_subset --best_threshold 0.05
# python fmax_evaluation.py --ontology 'MF' --model 'DeepFri' --test_subset --best_threshold 0.09

# python fmax_evaluation.py --ontology 'BP' --model 'PFresGO' --test_subset --best_threshold 0.09
# python fmax_evaluation.py --ontology 'CC' --model 'PFresGO' --test_subset --best_threshold 0.04
# python fmax_evaluation.py --ontology 'MF' --model 'PFresGO' --test_subset --best_threshold 0.16

# python fmax_evaluation.py --ontology 'BP' --model 'HEAL' --test_subset --best_threshold 0.05
# python fmax_evaluation.py --ontology 'CC' --model 'HEAL' --test_subset --best_threshold 0.02
# python fmax_evaluation.py --ontology 'MF' --model 'HEAL' --test_subset --best_threshold 0.07

#----------------------------------------------Results after combining with GoBER--------------------------------------------------------

echo "### Results after combining with GoBERT ###"
# python fmax_evaluation.py --ontology 'BP' --model 'DeepFri' --replace_gobert --best_threshold 0.07
# python fmax_evaluation.py --ontology 'CC' --model 'DeepFri' --replace_gobert --best_threshold 0.05
# python fmax_evaluation.py --ontology 'MF' --model 'DeepFri' --replace_gobert --best_threshold 0.09

python fmax_evaluation.py --ontology 'BP' --model 'PFresGO' --replace_gobert --best_threshold 0.09
python fmax_evaluation.py --ontology 'CC' --model 'PFresGO' --replace_gobert --best_threshold 0.04
python fmax_evaluation.py --ontology 'MF' --model 'PFresGO' --replace_gobert --best_threshold 0.16

python fmax_evaluation.py --ontology 'BP' --model 'HEAL' --replace_gobert --best_threshold 0.05 
# python fmax_evaluation.py --ontology 'CC' --model 'HEAL' --replace_gobert --best_threshold 0.02 --best_gobert_threshold 0.82 --best_gobert_ratio 0.1
# python fmax_evaluation.py --ontology 'CC' --model 'HEAL' --replace_gobert --best_threshold 0.02 --best_gobert_ratio 0.1
python fmax_evaluation.py --ontology 'MF' --model 'HEAL' --replace_gobert --best_threshold 0.07 

echo "### All evaluations completed ###"
