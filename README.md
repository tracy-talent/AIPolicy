# AIPolicy
PASA Information Extraction Toolkit for AIPolicy

### 项目结构
```
.
├── config.ini                              配置文件      
├── input                                   数据和模型输入目录
│   ├── benchmark                           基线目录
│   │   ├── entity                          实体数据集目录
│   │   │   └── policy                      实体-政策数据集
│   │   └── relation                        关系数据集目录
│   │       └── test-policy                 关系-政策数据集
│   └── pretrain                            预训练模型目录
│       └── hfl-chinese-bert-wwm-ext        中文Bert预训练模型
│	
├── labs                                    实验目录
│   ├── entity                              实体-训练脚本目录
│   └── relation                            关系-训练脚本目录
│
├── LICENSE                                 证书
├── output                                  输出目录
│   ├── entity                              实体输出目录
│   │   └── ckpt                            实体checkpoint目录
│   │       └── policy_bmoes                policy数据集模型保存目录
│   └── relation                            关系输出目录
│       └── ckpt                            关系checkpoint目录
│           └── test-policy                 test-policy数据集模型保存目录
│
├── pasaie                                  信息抽取模块
│   ├── losses                              损失函数模块
│   │   ├── autoweighted_loss.py			
│   │   ├── dice_loss.py					
│   │   ├── focal_loss.py					
│   │   └── label_smoothing.py				
│   ├── metrics                             度量模块
│   │   ├── basestats.py                    基本性能评估类
│   │   └── metrics.py                      其它性能评估函数
│   ├── module                              神经网络模块
│   │   ├── nn                              神经网络基础单元目录
│   │   │   ├── cnn.py						
│   │   │   ├── linear.py					
│   │   │   ├── lstm.py						
│   │   │   └── rnn.py						
│   │   └── pool                            池化层目录
│   │       ├── avg_pool.py                 平均池化
│   │       └── max_pool.py                 最大池化
│   ├── pasaap                              文章、句子、要求解析模块
│   │   ├── framework                       framework模块
│   │   │   └── article_parser.py           文章分解
│   │   └── tools                           工具模块
│   │       ├── node.py                     要求间逻辑树类
│   │       ├── plot.py                     画图函数
│   │       └── search_sentences.py         句子搜寻
│   ├── pasaner                             实体抽取模块
│   │   ├── decoder                         解码器模块
│   │   │   └── crf.py
│   │   ├── encoder                         编码器模块
│   │   │   ├── base_encoder.py
│   │   │   ├── base_wlf_encoder.py
│   │   │   ├── bert_bilstm_encoder.py
│   │   │   ├── bert_encoder.py
│   │   │   ├── bilstm_encoder.py
│   │   │   ├── bilstm_wlf_encoder.py
│   │   │   └── xlnet_encoder.py
│   │   ├── framework                       框架模块
│   │   │   ├── data_loader.py              数据载入
│   │   │   ├── model_crf.py                流程控制函数-基线
│   │   │   ├── mtl_span_attr.py            流程控制函数-基于span多标签
│   │   │   ├── span_based_ner.py           流程控制函数-基于span
│   │   │   └── xlnet_crf.py                流程控制函数-Xlnet代替Bert
│   │   ├── model                           分类器模块
│   │   │   ├── bilstm_crf.py
│   │   │   ├── bilstm_crf_span_attr.py
│   │   │   └── span_cls.py
│   │   └── pretrain.py                     NER预训练模型接口
│   ├── pasare                              关系分类模块
│   │   ├── encoder                         编码器模块
│   │   │   ├── base_encoder.py
│   │   │   ├── bert_encoder.py
│   │   │   ├── cnn_encoder.py
│   │   │   └── pcnn_encoder.py
│   │   ├── framework                       框架模块
│   │   │   ├── bag_re.py                   流程控制函数-基于远程监督
│   │   │   ├── data_loader.py              数据载入
│   │   │   ├── sentence_re.py              流程控制函数-基线
│   │   │   └── utils.py                    辅助函数
│   │   ├── model                           分类器模块
│   │   │   ├── bag_attention.py
│   │   │   ├── bag_average.py
│   │   │   ├── base_model.py
│   │   │   └── softmax_nn.py
│   │   ├── pretrain.py                     RE预训练模型接口
│   │   └── utils                           辅助模块
│   │       ├── dataset_split.py            数据集划分（已丢弃）
│   │       ├── eval.py                     性能评估函数（已丢弃）
│   │       └── relation_statistic.py       关系数据集指标统计
│   ├── tokenization                        分词模块
│   │   ├── basic_tokenizer.py
│   │   ├── bert_tokenizer.py
│   │   ├── toolkit_tokenizer.py
│   │   ├── utils.py
│   │   ├── word_piece_tokenizer.py
│   │   └── word_tokenizer.py
│   └── utils                               辅助模块
│       ├── adversarial.py                  对抗函数
│       ├── corpus.py                       语料生成
│       ├── dependency_parse.py             依存句法函数
│       ├── embedding.py                    加载预训练embedding
│       ├── entity_extract.py               实体抽取
│       ├── entity_tag.py                   NER标签模式转换（如BIO => BMOES）
│       ├── log.py                          日志
│       ├── sampler.py                      采样函数
│       ├── seed.py                         pytorch随机种子
│       └── timedec.py                      时间统计
├── README.md							
├── requirements.txt
└── test                                    测试模块
    ├── main.py                             主入口，项目pipeline
    └── test_inference.py                   测试接口
```