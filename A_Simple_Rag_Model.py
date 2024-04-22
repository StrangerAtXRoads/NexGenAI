from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration 
 
# Initialize the Retriever & the model
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True) ## Uses a pretrained DPR dataset (wiki_dpr) https://huggingface.co/datasets/wiki_dpr
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever) 

# A sample query to ask the model
query = "How long have Australia held on to the Ashes?" 


tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")  ## Get the tokenizer from the pretrained model
tokenized_text = tokenizer(query, return_tensors='pt', max_length=100, truncation=True) ## Encode/Tokenize the query


model_generated_tokens = model.generate(input_ids=tokenized_text["input_ids"], max_new_tokens=1000) ## Find the relavant information from the dataset (tokens)

print(tokenizer.batch_decode(model_generated_tokens, skip_special_tokens=True)[0]) ## Decode the data to find the answer