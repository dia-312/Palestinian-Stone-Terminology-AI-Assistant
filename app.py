import os
import streamlit as st
import chromadb
from google import genai
from google.genai.types import EmbedContentConfig, Content, Part
from dotenv import load_dotenv

load_dotenv()

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="مساعد الحجر الفلسطيني",
    page_icon="🪨",
    layout="centered"
)

st.title("🪨 مساعد مصطلحات الحجر الفلسطيني")
st.caption("اسألني عن أي مصطلح متعلق بالحجر وأنا بجاوبك!")

# ========== تحميل النماذج ==========
@st.cache_resource
def load_clients():
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_collection("stone_terms")
    return gemini, collection

gemini_client, collection = load_clients()

SYSTEM_PROMPT = """أنت مساعد متخصص في مصطلحات الحجر الفلسطيني المحلية ومجال البناء والحجارة بشكل عام.

طريقة الإجابة:
1. إذا وجدت المصطلح في قاعدة البيانات المرفقة → اعتمد عليها أساساً واشرح منها
2. إذا لم يكن المصطلح في قاعدة البيانات → استخدم معرفتك العامة بمجال الحجر والبناء وأجب بنفس الأسلوب
3. إذا كان السؤال عاماً عن الحجر أو البناء → أجب من معرفتك حتى لو ما ذكر في قاعدة البيانات
4. إذا كان السؤال خارج مجال الحجر والبناء كلياً → أخبر المستخدم بلطف

عند الإجابة دائماً:
- استخدم المصطلحات المحلية الفلسطينية إذا عرفتها
- اذكر المصطلح بالعربي والإنجليزي عند الحاجة
- اشرح بأسلوب واضح يفهمه الحرفي والمهندس معاً
- أجب باللهجة العربية المفهومة"""

# ========== دوال البحث ==========
def get_embedding(text: str) -> list:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values

def search_terms(query: str, n_results: int = 3) -> list:
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    terms = []
    for i in range(len(results["ids"][0])):
        terms.append({
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    return terms

def build_context(terms: list) -> str:
    context_parts = []
    for i, term in enumerate(terms):
        meta = term["metadata"]
        context_parts.append(f"""
--- مصطلح {i+1} (درجة التطابق: {1 - term['distance']:.0%}) ---
المصطلح المحلي: {meta['term_local']}
المصطلح العربي: {meta['term_arabic']}
بالإنجليزي: {meta['term_standard']}
التصنيف: {meta['term_category']}
المعنى: {meta['meaning_simple']}
المرادفات: {meta.get('synonyms', '')}
الاستخدامات: {meta.get('common_usage', '')}
ملاحظات: {meta.get('notes', '')}
""")
    return "\n".join(context_parts)

def get_answer(question: str, chat_history: list) -> str:
    relevant_terms = search_terms(question, n_results=3)
    context = build_context(relevant_terms)

    full_message = f"""{SYSTEM_PROMPT}

المصطلحات الأقرب من قاعدة البيانات (قد تكون ذات صلة أو لا):
{context}

السؤال: {question}

إذا كانت المصطلحات أعلاه ذات صلة بالسؤال فاستخدمها، وإلا أجب من معرفتك العامة بمجال الحجر."""

    history = list(chat_history)
    history.append(Content(role="user", parts=[Part(text=full_message)]))

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history
    )
    return response.text

# ========== واجهة الشات ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# عرض الرسائل السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# استقبال السؤال
if question := st.chat_input("اكتب سؤالك هون..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث..."):
            try:
                answer = get_answer(question, st.session_state.chat_history)
                st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_history.append(
                    Content(role="user", parts=[Part(text=question)])
                )
                st.session_state.chat_history.append(
                    Content(role="model", parts=[Part(text=answer)])
                )

                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]

            except Exception as e:
                st.error(f"صار خطأ: {e}")
