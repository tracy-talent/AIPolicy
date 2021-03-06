bash run_bert_dist_with_dsp.sh kbp37 entity_dist_context_dsp 0
bash run_bert_dist_with_dsp.sh kbp37 entity_dist_dsp 0
bash run_bert_dist.sh kbp37 entity_dist_context 0
bash run_bert_dist.sh kbp37 entity_dist 0
bash run_bert.sh kbp37 entity 0
bash run_bert_dist_with_dsp.sh semeval entity_dist_context_dsp 0
bash run_bert_dist_with_dsp.sh semeval entity_dist_dsp 0
bash run_bert_dist.sh semeval entity_dist_context 0
bash run_bert_dist.sh semeval entity_dist 0
bash run_bert.sh semeval entity 0
cp ./bert_entity_dist_alltype.out /data/labs/relation
sh ~/shutdown.sh
