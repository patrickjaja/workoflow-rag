ultra think: build a application that is providing an python fastapi that allows to            
                                                                                                  
   1. search for data (search string will be embedded, api returns 10 results, these results are  
   ranked by relevance)                                                                           
                                                                                                  
   2. creates a index to data uploaded to a minio bucket                                          
                                                                                                  
   3. creates a collection in a vector database to search inside that data using hybrid search    
   (semantic or based on a smart text index that can be created using LLM or other technique)     
                                                                                                  
   My suggestion: This data is read from a minio bucket, and this data can be structured (JSON,   
   CSV) or unstructured (PDF, TXT). This data is loaded into a qdrant vector store with meta      
   information and different index fields based on the chunk and dataset.                         
                                                                                                  
   We also need an api endpoint to upload new files (filetypes json, txt, csv, txt) that will     
   put into that index (vector collection). To create the embeddings to data chunks, use these    
   azure api                                                                                      
                                                                                                  
   endpoint = "https://oai-cec-de-germany-west-central.openai.azure.com/"                         
   model_name = "text-embedding-3-large"                                                          
   deployment = "text-embedding-3-large"                                                          
                                                                                                  
   api_version = "2024-02-01"                                                                     
   api_key =                                                                                      
   "xxx"         
                                                                                                  
   When you need LLM access for i.e. text indexing you can use this credentials                   
                                                                                                  
   model_name=gpt-4o-mini                                                                         
   endpoint="https://oai-cec-de-germany-west-central.openai.azure.com/openai/deployments/gpt-4o-  
   mini"                                                                                          
   api_key="xxxx"                                                                                              
                                                                                                  
                                                                                                  
   Conclusion:                                                                                    
   Please build me a onpremise application, its api i would like to connect as a tool into an     
   n8n agent for a RAG pattern - so the agent is able to read the provided chunks. Setup should   
   be onpremise, but services or architecture is not final - you can improve it. Build me a       
   docker-setup, the FULL application as described above. Test it based on data in folder         
   example_data/                                                                                  
                                                                                                  
   I think the challenge will be to combine different search types to provide the highest         
   accurency possible. I also thought about a connection to neo4j - but not sure how to archive   
   the data and logic relation. You can decide on the final concept.       
