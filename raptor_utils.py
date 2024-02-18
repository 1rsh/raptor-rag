import tiktoken
from tqdm.auto import tqdm
from transformers import pipeline
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma


def text_chunker(document: str, token_limit: int = 128):
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    chunks = []
    tokens = enc.encode(document, disallowed_special=())

    while tokens:
        chunk = tokens[:token_limit]
        chunk_text = enc.decode(chunk)
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )
        
        if last_punctuation != -1 and len(tokens) > token_limit:
            chunk_text = chunk_text[: last_punctuation + 1]
            
        cleaned_text = chunk_text.replace("\n", " ").strip()
        
        if cleaned_text and (not cleaned_text.isspace()):
            chunks.append(cleaned_text)
            
        tokens = tokens[len(enc.encode(chunk_text, disallowed_special=())):]

    return chunks

def lang_doc_chunker(documents: list, token_limit: int = 128):
    chunks = []
    
    for _document in documents:
        chunks.extend(text_chunker(_document, token_limit))
        
    return chunks


def raptor_chunk(documents: list, token_limit: int = 128, tree_depth: int = 3, strategy: int = 0):
    
    documents = [_document.page_content for _document in documents]
    
    chunks = {}
    chunks[tree_depth] = lang_doc_chunker(documents, token_limit)
    tree_depth -= 1
    
    summarizer = pipeline('summarization', model="t5-small", tokenizer="t5-small", framework="tf")
    
    while tree_depth > 0:
        
        documents = [''.join(x) for x in zip(chunks[tree_depth + 1][0::2], chunks[tree_depth + 1][1::2])]
        
        new_documents = []
        for _document in tqdm(documents, desc = f"Constructing tree layer {tree_depth}"):
            new_documents.append(summarizer(_document, max_length = len(_document)//8)[0]['summary_text'])
        
        token_limit *= 2
        
        chunks[tree_depth] = lang_doc_chunker(new_documents, token_limit)
        
        documents = new_documents
        
        tree_depth -= 1
        
    if strategy == 0: 
        tree_traversal = {}
        for key in chunks.keys():
            tree_traversal[key] = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks[key]]
        return tree_traversal
    else:
        collapsed = []
        for key in chunks.keys():
            collapsed.extend(chunks[key])
        
        return [Document(page_content=node, metadata={"source": "local"}) for node in collapsed]


def raptor_db(documents: list, tree_depth: int = 3, strategy: int = 0, chunk_token_limit: int = 128, embeddings = HuggingFaceHubEmbeddings()):
    
    chunks = raptor_chunk(documents, chunk_token_limit, tree_depth, strategy)
        
    if strategy == 1:
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        
        return vectorstore
    else:
        vectorstore = [Chroma.from_documents(documents=chunks[key], embedding=embeddings) for key in chunks.keys()]
        return vectorstore


def raptor_retrieval(query, vectorstore, top_k: int = 3, embeddings = HuggingFaceHubEmbeddings()):

    if type(query) == str:
        embedding_vector = embeddings.embed_query(query)
    else:
        embedding_vector = query

    if type(vectorstore) != list:

        retrieved = vectorstore.similarity_search_by_vector(embedding_vector, k = top_k)
    else:
        retrieved = []
        for vs in vectorstore:
            retrieved.append(vs.similarity_search_by_vector(embedding_vector, k = top_k))

    return retrieved

