import sentencepiece as spm
text_path = "./data/trans_data2/train_ja_en_text.txt"
model_name = "./data/en_ja_16000"
vocab_size = 8000
spm.SentencePieceTrainer.Train(f"--input={text_path} --model_prefix={model_name} --vocab_size={vocab_size}")