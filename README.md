# VAE-Transformer
This project aims to overcome the data scarcity obstacle in Turkish natural language processing projects by generating sentences with similar meanings from sentences in Turkish datasets. For this purpose, various natural language processing models were compared, and sentence generation was performed using variational autoencoders (VAE). Specifically, RNN and GRU models were compared, and BERT and Transformer architectures were examined. The success of the model was measured by enriching Turkish datasets and improving classification success rates. The results show that enriching the dataset significantly improves classification performance. The detailed comparison and results of the methods used in the project are considered an important step for the development of Turkish natural language processing projects.

## FOLDER ORGANIZATION

      data---
            |
            -- corpus.train.txt
            |
            -- corpus.valid.txt

## MODEL TRAINING

```bash
python trainer.py \
       --data_name "corpus" \
       --data_dir "data" \ 
       --epochs 3 \
       --learning_rate 0.001 \
       --batch_size 64
```

## AUGMENTATION

```bash
python augment.py \
       --data_name "{data_name}" \
       --save_model_path "models" \ 
       --checkpoint "{model_path}" \
       ---max_sentence_length 60 \
       --rate 0.5
```

## DEMO

```bash
python demo.py \
       --checkpoint "{model_path}" \
       --data_name "{data_name}" \ 
       --save_model_path "models" \
       --noise "False" \
```

