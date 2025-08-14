import torch
from transformers import BertTokenizer, BertForMaskedLM

# 1. Load pre-trained BERT model + tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# 2. Input sentence with a masked token
sentence = "The capital of France is [MASK]."

# 3. Tokenize the input
inputs = tokenizer(sentence, return_tensors="pt")

# 4. Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# 5. Find the predicted token for the [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"Predicted word: {predicted_token}")
