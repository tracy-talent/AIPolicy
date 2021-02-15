echo "without pinyin, ctb"
bash run_bert_bmes_lexicon_att_mtl_span_attr_boundary.sh resume ctb 0 0.1 13
echo "without pinyin, sgns"
bash run_bert_bmes_lexicon_att_mtl_span_attr_boundary.sh resume sgns 0 0.5 11
echo "without lexicon and pinyin"
bash run_bert_mtl_span_attr_boundary.sh resume 0 0.1
echo "B-Loc M-Loc E-Loc tagging scheme, ctb"
bash run_bert_lstm_crf_bmes_lexicon_pinyin_word_att.sh resume ctb word2vec 0 0.1 13
echo "B-Loc M-Loc E-Loc tagging scheme, sgns"
bash run_bert_lstm_crf_bmes_lexicon_pinyin_word_att.sh resume sgns word2vec 0 0.5 11
echo "Loc O O/O O Loc tagging scheme, ctb"
bash run_bert_bmes_lexicon_pinyin_word_att_mtl_attr_boundary.sh resume ctb word2vec 0 0.1 13
echo "Loc O O/O O Loc tagging scheme, sgns"
bash run_bert_bmes_lexicon_pinyin_word_att_mtl_attr_boundary.sh resume sgns word2vec 0 0.5 11
