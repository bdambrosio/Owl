from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/bruce/Downloads/models/Specter-2-base')

#load base model
model = AutoModel.from_pretrained('/home/bruce/Downloads/models/Specter-2-base')

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter('/home/bruce/Downloads/models/Specter-2', source="hf", load_as="proximity", set_active=True)
#model.load_adapter('/home/bruce/Downloads/models/Specter-2')
#other possibilities: allenai/specter2_<classification|regression|adhoc_query>

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# concatenate title and abstract
text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
# preprocess the input
inputs = tokenizer(text_batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=512)
output = model(**inputs)
# take the first token in the batch as the embedding
embeddings = output.last_hidden_state[:, 0, :]
print(embeddings.shape)
