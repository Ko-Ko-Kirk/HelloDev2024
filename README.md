# 20240912 Helle World Dev Conf 工作坊 --「使用 LangServe 快速部署 AI」

## 說明

這是 20240912 在 Hello World Dev Conference 的實戰工作坊「使用 LangServe 快速部署 AI」的內容，帶大家快速建立 LangServe 的 Web API，並且進行容器化部署。

## 準備工作

1. 先給這個 repo 一個星星。
2. 先追蹤臉書粉專「大魔術熊貓工程師」。
3. 先確認大家是不是裝好了 Poetry: `poetry --version`。
4. 如果還沒裝好的，那麼來不及了，回去跟著這個 repo 操作吧！
5. https://python-poetry.org/docs/#installing-with-the-official-installer\
6. 是否已經到 Qdrant Cloud 開好帳號了，然後拿到自己的連接方式了？ https://qdrant.tech/
7. 是否已經加了工作坊上的 LINE 官方帳號 `@231yjujq`，拿到公用 Azure OpenAI 的 GPT-4o 的 key 了呢？會後還會透過這個送出投影片哦！
8. 取得政府托育補助的 pdf： https://www.sfaa.gov.tw/SFAA/Pages/ashx/File.ashx?FilePath=~/File/Attach/10802/File_191388.pdf


## 安裝第一個 LangChain 虛擬環境

1. 打開你的 VS code。
2. `poetry new HelloDev2024`
3. `cd HelloDev2024`
4. 在 `pyproject.toml` 改 Python 版本到 3.11 後，`poetry install`。
5. `poetry add langchain-cli langchain-openai qdrant-client langchain-community pypdf`。


## 寫好第一個 LangChain 的 RAG

建立一個新檔案： `hellodev2024/rag.py`

```
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

loader = PyPDFLoader("./qa.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(    
    chunk_size=500,
    chunk_overlap=100)

chunks = splitter.split_documents(docs)

embeddings_model = AzureOpenAIEmbeddings(
    api_key="xx",
    azure_deployment="text-small", 
    openai_api_version="2024-06-01",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
)


qdrant = Qdrant.from_documents(
    chunks,
    embeddings_model,
    url="https://xx.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="xx-xx-xx",
    collection_name="subsidy_qa",
    force_recreate=True,
)


retriever = qdrant.as_retriever(search_kwargs={"k": 3})


model = AzureChatOpenAI(
    api_key="xx",
    openai_api_version="2024-06-01",
    azure_deployment="gpt-4o",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")


document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "請問第二胎補助加發多少，共為多少錢？"})

print(response["answer"])


```

進入虛擬環境： `poetry shell`
跑起來： `python HelloDev2024/rag.py`

找加入準公共的保母的話、每個月就 13000 台幣的補助，看了有沒有覺得生小孩沒有壓力了呢？


##  建立 LangServe 的虛擬環境（第二個虛擬環境）

我們這裡用 langchain-cli （前面有安裝）建立起來 langserve，先在第一個虛擬環境裡裝上 langchain-cli，然後離開，再進到 langserve 裡建立 langserve 的虛擬環境。

1. `peotry shell`
2. `langchain app new langserveapp`，直接空白跳過要你安裝的內容。
3. 用  `exit` 離開目前的虛擬環境
4. `cd langserveapp`
5. `poetry install`
6. `poetry add langchain-openai qdrant-client langchain-community`


兩個虛擬環境切換，要注意不要搞錯囉！

這裡會碰到 pydantic 的版本問題，記得去 pyproject.toml 裡調整為 pydantic = ">2" ，因為現在 LangServe 需要 1 版 pydantic 正常顯示文件，可是其他的套件需要 2 版以上的 pydantic。不過這個問題之後就會修復。

最後，再 `poetry shell` 進到我們這個 langserve 的虛擬環境中。


## 把你寫好的 RAG 整合到 LangServe 中。

這裡我們不再重新 embedding 了。坊間一堆範例都是教你重新 embedding，但是其實做一次就好了。各位同學來上這個工作坊真的是上對了。

建立檔案：`rag/rag_chain.py`

```
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.pydantic_v1 import BaseModel



embeddings_model = AzureOpenAIEmbeddings(
    api_key="xx",
    azure_deployment="text-small", 
    openai_api_version="2024-06-01",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
)

client = QdrantClient(
	url="https://xx.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="xx")
collection_name = "subsidy_qa"
qdrant = Qdrant(client, collection_name, embeddings_model)

retriever = qdrant.as_retriever(search_kwargs={"k": 3})

model = AzureChatOpenAI(
    api_key="xx",
    openai_api_version="2024-06-01",
    azure_deployment="gpt-4o",
    azure_endpoint="https://chatgpteastus.openai.azure.com/",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")


document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Add typing for input
class Question(BaseModel):
    input: str


rag_chain = retrieval_chain.with_types(input_type=Question)

```

新增檔案，讓他變成 python package：`rag/__init__.py`

```
from rag.rag_chain import rag_chain

__all__ = ["rag_chain"]
```

修改 app 資料夾裡的 `server.py`

```
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag import rag_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

```

啟動 LangServe `langchain serve`


Invoke 這個路由就是把 API 叫起來的方式，你就可以用這個 API 整合到你要串 LINE 的 webhook 裡。
```
curl --location --request POST 'http://localhost:8000/rag/invoke' \
--header 'Content-Type: application/json' \
--data-raw '{
    "input": {
        "input": "父母離婚，該由誰申請補助"
    }
}'
```

Stream 的話就是用串流模式，字會一個一個的出來，建議做自己的產品就要用這個方式。（注意 LINE 還不支援串流模式，他是用 loading animation）
```
curl --location --request POST 'http://localhost:8000/rag/stream' \
--header 'Content-Type: application/json' \
--data-raw '{
    "input": {
        "input": "父母離婚，該由誰申請補助"
    }
}'


```

可以進到 play ground  `http://127.0.0.1:8000/rag/playground/` 來試玩

我們也可以去 Qdrant dashboard 裡，觀察 Qdrant 向量資料庫。

## 容器化部署

修改 dockerfile 如下：

```
FROM python:3.11-slim

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

COPY ./rag ./rag

RUN poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080

```

`docker build . -t hellodev-api:latest`

在專案裡開一個資料夾叫 docker-compose，然後新增檔案 `docker-compose.yaml`

```
version: '3'
services:
  langhchain-app:
    image: hellodev-api:latest
    ports: 
      - 8080:8080
    environment: # 如果你有抽離成環境變數的話再用
      QDRANT_HOST: qdrant
      QDRANT_KEY: xx
      AOAI_KEY: xx
```

可以 VS code 裡安裝 Docker 的延伸模組，比較方便操作 docker compose up。

確定本地端可以跑起來代表已經容器化部署成功了。

工作坊到這裡結束囉！有什麼問題，可以透過臉書粉專「大魔術熊貓工程師」找到我。新書「LangChain 的奇幻旅程」的預購資訊，也會在臉書粉專貼出哦！
