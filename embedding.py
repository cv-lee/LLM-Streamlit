from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_vectorstore():
    # 문서 로드
    loader = DirectoryLoader("./data", glob="**/*.txt")
    documents = loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(model_name='BM-K/KoSimCSE-roberta-multitask')

    # 벡터 저장소 생성
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 벡터 저장소 저장
    vectorstore.save_local("./data/embedding_data")

if __name__ == "__main__":
    create_vectorstore()
    print("Vector store created and saved successfully.")
