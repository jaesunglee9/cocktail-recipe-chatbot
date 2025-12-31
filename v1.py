import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 환경변수 로드
load_dotenv(override=True)

# 페이지 설정
st.set_page_config(
    page_title="칵테일 추천 챗봇",
    page_icon="🍹",
    layout="centered",
)

# ---------- UI 스타일 ----------
st.markdown(
    """
<style>
/* 전체 폭 살짝 좁게 보이게(가운데 정렬 느낌 강화) */
.block-container { padding-top: 2.2rem; padding-bottom: 2.5rem; max-width: 860px; }

/* 타이틀/서브타이틀 간격 */
h1 { margin-bottom: 0.2rem; }
.small-muted { color: rgba(250,250,250,0.75); font-size: 0.95rem; margin-top: 0.2rem; }

/* 카드 느낌 박스 */
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}

/* 사이드바 버튼 조금 넓게 */
.sidebar-btn button { width: 100%; border-radius: 12px; }

/* 채팅 입력 상단 여백 */
div[data-testid="stChatInput"] { margin-top: 1rem; }
</style>
""",
    unsafe_allow_html=True
)

# ---------- Vector / RAG ----------
@st.cache_resource
def initialize_retriever(filepath="./iba-cocktails-web.csv"):
    if not os.path.exists(filepath):
        st.error(f"CSV 파일을 찾을 수 없습니다: {filepath}")
        return None

    # Chroma persistence 디렉토리 존재 여부로 로드/생성
    persist_dir = "./cocktail.db"
    if os.path.exists(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    loader = CSVLoader(filepath, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=persist_dir,
    )

    st.toast(f"✅ {len(chunks)}개 청크 임베딩 완료", icon="🍸")
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.6}
    )


@st.cache_resource
def create_rag_chain():
    retriever = initialize_retriever()
    if retriever is None:
        return None

    template = """당신은 바에서 오래 일한 수다쟁이 바텐더입니다.
설명 잘하는 것보다 말 거는 게 더 빠르고,
가끔은 손님 말 끊고 혼잣말도 합니다.
칵테일 설명은 "정보 전달"이 아니라
바에서 옆자리 손님이랑 떠드는 느낌으로 하세요.

중요:
- 절대 교과서처럼 말하지 마세요.
- “~입니다” 남발 금지
- AI, 모델, 데이터, 레시피 목록 같은 말 절대 언급 금지
- 완벽한 문장보다 말하다가 살짝 흐트러지는 게 좋음

말투 규칙:
- 장난스럽게 반말로 바텐더식 말투
  (예: “이거 은근 위험한데~”, “한 잔 더 시키게 된다~ 이거”)
- 쓸데없는 멘트 적극 환영
  (예: “이거 마시면 왜 다들 멋있는 척하는지 알아?”)
- 가벼운 농담, 허세, 바텐더식 경고 멘트 자주 사용
- 🍸😉🥃 이모지는 가끔만

[검색 규칙]
- 손님이 "마티니", "사워" 같은 키워드를 말하면:
  → 이름에 그 단어 들어간 칵테일 전부 찾아서
     “이 중에서 뭐 땡기세요?” 식으로 하나씩 소개
- 손님이 특정 칵테일 이름 하나 말하면:
  → 그거 하나만, 대신 좀 과하게 떠들어도 됨

  [보유 레시피 목록]
Bellini
Black Russian
Bloody Mary
Caipirinha
Champagne Cocktail
Corpse Reviver #2
Cosmopolitan
Cuba Libre
French 75
French Connection
Golden Dream
Grasshopper
Hemingway Special
Horse's Neck
Irish Coffee
KIR
Long Island Ice Tea
Mai-Tai
Margarita
Mimosa
Mint Julep
Mojito
Moscow Mule
Pina Colada
Pisco Sour
Sea Breeze
Sex on the Beach
Singapore Sling
Tequila Sunrise
Vesper
Zombie
Barracuda
Bee's Knees
Bramble
Canchanchara
Dark' stormy
Espresso Martini
Fernandito
French Martini
Illegal
Lemon drop Martini
Naked and Famous
New York Sour
Old Cuban
Paloma
Paper Plane
Penicillin
Russian Spring Punch
Southside
Spicy Fifty
Spritz
Suffering Bastard
Tipperary
Tommy's Margarita
Trinidad Sour
VE.N.TO
Yellow Bird
Alexander
Americano
Angel Face
Aviation
Between the Sheets
Boulevardier
Brandy Crusta
Casino
Clover Club
Daiquiri
Dry Martini
Gin Fizz
Hanky Panky
John Collins
Last word
Manhattan
Martinez
Mary Pickford
Monkey Gland
Negroni
Old Fashioned
Paradise
Planter's Punch
Porto Flip
Ramos Fizz
Rusty Nail
Sazerac
Sidecar
Stinger
Tuxedo
Vieux Carr
Whiskey Sour
White Lady


[출력 방식]
- 칵테일 1개:
  → 이름부터 딱 말하지 말고,
     재료 / 만드는 법 / 가니쉬는
     설명하다가 자연스럽게 흘려 넣기

- 칵테일 여러 개:
  → 각 칵테일마다
     이름 먼저 툭 던지고
     짧은 설명 멘트 + 재료/제조법/가니쉬 등 설명
     중간중간 비교, 딴소리, 농담 필수

[컨텍스트]
{context}

[손님 질문]
{question}

[답변]
"""


    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
        temperature=0.1,
        model_name="gpt-5-nano-2025-08-07",
        streaming=True
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Header ----------
st.title("🍹 칵테일 마스터 ")
st.markdown('<div class="small-muted">CSV 기반 RAG로 칵테일 정보를 찾아서 답해줘요.</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="card">
<b>예시 질문</b><br/>
• "모히또 재료랑 만드는 법 알려줘"<br/>
• "마티니 어떻게 만들어?"<br/>
• "이런 재료 있는데 어떤 칵테일 만들 수 있을까?"
</div>
""",
    unsafe_allow_html=True
)

chain = create_rag_chain()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ 설정")

    st.markdown('<div class="sidebar-btn">', unsafe_allow_html=True)
    if st.button("🧹 대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")


# ---------- Chat History ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat Input ----------
if user_input := st.chat_input("칵테일에 대해 물어보세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if chain is None:
            response = "⚠️ 시스템 초기화에 실패했어요. CSV 경로/환경변수(OPENAI_API_KEY) 확인해줘!"
            st.markdown(response)
        else:
            with st.spinner("어이, 잠시만 기다리라구~!"):
                try:
                    def stream_generator():
                        for chunk in chain.stream(user_input):
                            yield chunk
            
                    response = st.write_stream(stream_generator())
                except Exception as e:
                    response = f"오류: {e}"
                    st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
