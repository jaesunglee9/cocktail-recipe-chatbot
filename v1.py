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

# ========== Page ==========
st.set_page_config(
    page_title="🍹 Cocktail Master",
    page_icon="🍹",
    layout="wide",
)

# ========== Global Style ==========
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');

* { font-family: 'Noto Sans KR', sans-serif !important; }

/* 전체 배경 - 바 느낌 */
.stApp {
  background: linear-gradient(180deg,
    rgba(20, 20, 30, 0.95) 0%,
    rgba(40, 30, 20, 0.95) 100%),
    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800"><rect fill="%23140a05" width="1200" height="800"/><rect fill="%231a0f0a" y="400" width="1200" height="400"/></svg>');
  background-size: cover;
  background-attachment: fixed;
}

.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* 헤더 */
.bar-header {
  position: fixed;
  top: 0; left: 0; right: 0;
  padding: 15px 30px;
  background: rgba(20, 20, 30, 0.95);
  border-bottom: 2px solid rgba(139, 90, 43, 0.5);
  z-index: 200;
  text-align: center;
  backdrop-filter: blur(10px);
}
.bar-title {
  font-size: 2rem;
  font-weight: 700;
  color: #f4d03f;
  text-shadow: 0 0 20px rgba(244, 208, 63, 0.5),
               0 0 40px rgba(244, 208, 63, 0.3);
  margin: 0;
  letter-spacing: 2px;
}
.bar-subtitle {
  color: rgba(244, 208, 63, 0.7);
  font-size: 0.9rem;
  margin-top: 5px;
}

/* 바 카운터 */
.bar-counter {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  height: 200px;
  background: linear-gradient(180deg,
    rgba(101, 67, 33, 0.9) 0%,
    rgba(139, 90, 43, 1) 40%,
    rgba(101, 67, 33, 1) 100%);
  border-top: 8px solid rgba(139, 90, 43, 0.8);
  box-shadow: 0 -10px 50px rgba(0,0,0,0.8),
              inset 0 5px 20px rgba(255,255,255,0.1);
  z-index: 100;
}
.bar-counter::before {
  content: '';
  position: absolute;
  top: -8px; left: 0; right: 0;
  height: 4px;
  background: linear-gradient(90deg,
    transparent 0%,
    rgba(255,215,0,0.3) 20%,
    rgba(255,215,0,0.5) 50%,
    rgba(255,215,0,0.3) 80%,
    transparent 100%);
}

/* 바 장식 */
.bar-glasses {
  position: absolute;
  bottom: 10px;
  left: 5%;
  font-size: 2rem;
  opacity: 0.6;
  animation: float 3s ease-in-out infinite;
}
.bar-glasses-right {
  left: auto;
  right: 5%;
  animation-delay: 1.5s;
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

/* 채팅 영역 (✅ 입력창 위까지만) */
.chat-area {
  position: fixed;
  top: 80px;
  bottom: 320px;                 /* ✅ 입력창+바카운터 안전거리 */
  left: 50%;
  transform: translateX(-50%);
  width: 90%;
  max-width: 1000px;

  overflow-y: auto;
  padding: 20px 20px 160px 20px; /* ✅ 아래 여유 */
  z-index: 90;                   /* 바(100) 아래, 입력창(150) 아래 */
}

/* 스크롤바 */
.chat-area::-webkit-scrollbar { width: 8px; }
.chat-area::-webkit-scrollbar-track {
  background: rgba(0,0,0,0.2);
  border-radius: 10px;
}
.chat-area::-webkit-scrollbar-thumb {
  background: rgba(139, 90, 43, 0.6);
  border-radius: 10px;
}

/* 말풍선 공통 */
.speech-bubble {
  position: relative;
  max-width: 540px;
  padding: 18px 22px;
  margin: 22px auto;
  border-radius: 20px;
  line-height: 1.6;
  font-size: 0.95rem;
  animation: fadeIn 0.25s ease-in;
  word-break: break-word;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* 손님 말풍선 */
.user-bubble {
  background: #667eea;
  color: white;
  margin-left: auto;
  margin-right: 80px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
.user-bubble::after {
  content: '';
  position: absolute;
  right: -20px;
  top: 20px;
  width: 0; height: 0;
  border: 15px solid transparent;
  border-left-color: #667eea;
  border-right: 0;
}
.user-avatar {
  position: absolute;
  right: -55px;
  top: 10px;
  font-size: 2.5rem;
  filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
}

/* 바텐더 말풍선 */
.bartender-bubble {
  background: #2d2d2d;
  color: #f4f4f4;
  border: 2px solid rgba(244, 208, 63, 0.5);
  margin-left: 80px;
  margin-right: auto;
  box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.bartender-bubble::before {
  content: '';
  position: absolute;
  left: -20px;
  top: 20px;
  width: 0; height: 0;
  border: 15px solid transparent;
  border-right-color: #2d2d2d;
  border-left: 0;
}
.bartender-avatar {
  position: absolute;
  left: -55px;
  top: 10px;
  font-size: 2.5rem;
  filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
}

/* 입력창 */
div[data-testid="stChatInput"] {
  position: fixed !important;
  bottom: 220px !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: 90% !important;
  max-width: 700px !important;
  z-index: 150 !important;

  background: rgba(255, 255, 255, 0.95) !important;
  border: 2px solid rgba(244, 208, 63, 0.6) !important;
  border-radius: 30px !important;
  padding: 8px 15px !important;
  box-shadow: 0 8px 30px rgba(0,0,0,0.7),
              0 0 20px rgba(244, 208, 63, 0.2) !important;
  backdrop-filter: blur(10px) !important;
}
div[data-testid="stChatInput"] > div {
  background: transparent !important;
  border: none !important;
}
div[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: #000000 !important;
  font-size: 1rem !important;
  padding: 12px 20px !important;
  border: none !important;
  font-weight: 500 !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
  color: rgba(0, 0, 0, 0.5) !important;
  font-weight: 500 !important;
}
div[data-testid="stChatInput"] button { color: #667eea !important; }

/* 예시 질문 카드 */
.example-questions {
  width: 100%;
  max-width: 800px;
  margin: 0 auto 30px auto;
  background: rgba(244, 208, 63, 0.1);
  border: 2px solid rgba(244, 208, 63, 0.3);
  border-radius: 20px;
  padding: 20px 30px;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}
.example-title {
  color: #f4d03f;
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 15px;
  text-align: center;
  text-shadow: 0 0 10px rgba(244, 208, 63, 0.5);
}
.example-item {
  background: rgba(255, 255, 255, 0.05);
  color: #f4f4f4;
  padding: 12px 18px;
  margin: 8px 0;
  border-radius: 12px;
  border-left: 4px solid #f4d03f;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.95rem;
}
.example-item:hover {
  background: rgba(244, 208, 63, 0.2);
  transform: translateX(5px);
  border-left-color: #fff;
}

/* 쉐이커 */
.shaker-container {
  position: fixed;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  z-index: 300;
  text-align: center;
}
.shaker {
  font-size: 100px;
  animation: shake 0.5s infinite;
  filter: drop-shadow(0 10px 20px rgba(244, 208, 63, 0.5));
}
@keyframes shake {
  0%, 100% { transform: rotate(-15deg) translateY(0); }
  25%      { transform: rotate(15deg) translateY(-10px); }
  50%      { transform: rotate(-15deg) translateY(0); }
  75%      { transform: rotate(15deg) translateY(-10px); }
}
.shaker-text {
  margin-top: 20px;
  font-size: 1.8rem;
  font-weight: 700;
  color: #f4d03f;
  text-shadow: 0 0 20px rgba(244, 208, 63, 0.8),
               0 0 40px rgba(244, 208, 63, 0.5);
  animation: pulse 1s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.8; transform: scale(1.05); }
}

/* 초기화 버튼 */
.reset-btn {
  position: fixed;
  top: 15px;
  right: 20px;
  z-index: 250;
}
.reset-btn button {
  background: rgba(244, 208, 63, 0.2) !important;
  color: #f4d03f !important;
  border: 2px solid rgba(244, 208, 63, 0.4) !important;
  border-radius: 10px !important;
  padding: 8px 20px !important;
  font-weight: 600 !important;
  transition: all 0.3s !important;
}
.reset-btn button:hover {
  background: rgba(244, 208, 63, 0.3) !important;
  transform: scale(1.05) !important;
}

/* 사이드바 숨기기 */
section[data-testid="stSidebar"] { display: none; }

/* Streamlit 기본 요소 숨기기 */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ========== Retriever ==========
@st.cache_resource
def initialize_retriever(filepath="./iba-cocktails-web.csv"):
    if not os.path.exists(filepath):
        st.error("CSV 파일이 없습니다.")
        return None

    persist_dir = "./cocktail.db"
    if os.path.exists(persist_dir):
        vs = Chroma(
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
        return vs.as_retriever(search_kwargs={"k": 10})

    loader = CSVLoader(filepath, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=persist_dir,
    )
    return vs.as_retriever(search_kwargs={"k": 5})


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
- 재료는 한국어로 얘기하는 게 좋겠고 칵테일 이름은 영어로 해도 상관없어 
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

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ========== Session ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== Header ==========
st.markdown("""
<div class="bar-header">
    <div class="bar-title">🍹 Cocktail Master Bar 🍹</div>
    <div class="bar-subtitle">어서오세요, 무슨 칵테일 드릴까요?</div>
</div>
""", unsafe_allow_html=True)

# ========== 초기화 버튼 ==========
col1, col2, col3 = st.columns([8, 1, 1])
with col3:
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("🔄 새로 시작"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Chat Area ==========
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    
    # 예시 질문 (대화 시작 전에만 표시)
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="example-questions">
            <div class="example-title">💬 이렇게 물어보세요!</div>
            <div class="example-item">🍸 "모히또 재료랑 만드는 법 알려줘"</div>
            <div class="example-item">🥃 "마티니 어떻게 만들어?"</div>
            <div class="example-item">🍹 "위스키로 만드는 칵테일 추천해줘"</div>
            <div class="example-item">🍋 "새콤한 칵테일 뭐 있어?"</div>
        </div>
        """, unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <div class="speech-bubble user-bubble">
                    <div class="user-avatar">😵‍💫</div>
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <div class="speech-bubble bartender-bubble">
                    <div class="bartender-avatar">🕴️</div>
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Bar Counter ==========
st.markdown("""
<div class="bar-counter">
    <div class="bar-glasses">🍸 🥃 🍷</div>
    <div class="bar-glasses bar-glasses-right">🍹 🥂 🍾</div>
</div>
""", unsafe_allow_html=True)

# ========== Input Area ==========
chain = create_rag_chain()

# 입력창을 가장 아래에 배치
if user_input := st.chat_input("칵테일에 대해 물어보세요... 🍸"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if chain is None:
        response = "죄송해요, 지금 시스템 점검 중이에요 😅"
    else:
        # 쉐이커 애니메이션 표시
        shaker_placeholder = st.empty()
        shaker_placeholder.markdown("""
        <div class="shaker-container">
            <div class="shaker">🍸</div>
            <div class="shaker-text">칵테일 만드는 중...</div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            response = ""
            for chunk in chain.stream(user_input):
                response += chunk
        except Exception as e:
            response = f"앗, 실수로 잔을 깨뜨렸네요... 😵 ({str(e)})"
        finally:
            # 애니메이션 제거
            shaker_placeholder.empty()
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
