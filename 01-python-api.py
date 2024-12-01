def ollama_chat():
    import ollama
    response = ollama.chat(model='qwen2:0.5b', messages=[
    {
        'role': 'user',
        'content': '为什么天空是蓝色的？',
    },
    ])
    print(response['message']['content'])

# 流式响应
def ollama_stream():
    import ollama

    stream = ollama.chat(
        model='qwen2:0.5b',
        messages=[{'role': 'user', 'content': '为什么天空是蓝色的？'}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

# langchain
def ollama_langchain():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama
    template = """
    你是一个乐于助人的AI，擅长于解决回答各种问题。
    问题：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOllama(model="qwen2:0.5b", temperature=0)

    chain = prompt | model
    print(chain.invoke({"question": "你比GPT4厉害吗？"}))

def ollama_langchain_stream():
    from langchain_ollama import ChatOllama

    model = ChatOllama(model="qwen2:0.5b", temperature=0.7)

    messages = [
        ("human", "你好呀"),
    ]

    for chunk in model.stream(messages):
        print(chunk.content, end='', flush=True)

def simple_rag():
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings
    # 初始化 Ollama 模型和嵌入
    llm = ChatOllama(model="llama3.1")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 准备文档
    text = """
    Datawhale 是一个专注于数据科学与 AI 领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。
    Datawhale 以" for the learner，和学习者一起成长"为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。
    同时 Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。
    如果你想在Datawhale开源社区发起一个开源项目，请详细阅读Datawhale开源项目指南[https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md]
    """

    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_text(text)

    # 创建向量存储
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # 创建提示模板
    template = """只能使用下列内容回答问题:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 创建检索-问答链
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # 使用链回答问题
    question = "我想为datawhale贡献该怎么做？"
    response = chain.invoke(question)

if __name__ == '__main__':
    # ollama_chat
    # ollama_chat()
    # ollama_stream
    # ollama_stream()
    # ollama_langchain 
    # ollama_langchain()
    # ollama_langchain_stream
    ollama_langchain_stream()
    # simple_rag 模型下载速度太慢了



