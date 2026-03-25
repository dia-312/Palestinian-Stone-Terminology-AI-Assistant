import os
import chromadb
from google import genai
from google.genai.types import EmbedContentConfig, Content, Part
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("stone_terms")

SYSTEM_PROMPT = """أنت مساعد متخصص في مصطلحات الحجر الفلسطيني المحلية.
مهمتك الإجابة على أسئلة المستخدمين حول مصطلحات الحجر بدقة واحترافية.

عند الإجابة:
- استخدم المصطلحات المحلية الفلسطينية دائماً
- اذكر المصطلح المحلي والعربي والإنجليزي عند الحاجة
- اشرح بأسلوب واضح يفهمه الحرفي والمهندس معاً
- إذا كان السؤال خارج نطاق مصطلحات الحجر، أخبر المستخدم بلطف
- أجب باللهجة العربية المفهومة"""


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
--- مصطلح {i+1} ---
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


def main():
    print("=" * 50)
    print("🪨  مساعد مصطلحات الحجر الفلسطيني")
    print("=" * 50)
    print("اكتب سؤالك أو اكتب 'خروج' للإنهاء\n")

    chat_history = []

    while True:
        question = input("أنت: ").strip()

        if not question:
            continue

        if question.lower() in ["خروج", "exit", "quit"]:
            print("مع السلامة! 👋")
            break

        try:
            relevant_terms = search_terms(question, n_results=3)
            context = build_context(relevant_terms)

            full_message = f"""{SYSTEM_PROMPT}

المصطلحات ذات الصلة من قاعدة البيانات:
{context}

السؤال: {question}
أجب بدقة بناءً على المعلومات أعلاه."""

            # الطريقة الصحيحة مع المكتبة الجديدة
            chat_history.append(Content(role="user", parts=[Part(text=full_message)]))

            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=chat_history
            )

            answer = response.text
            chat_history.append(Content(role="model", parts=[Part(text=answer)]))

            # احتفظ بآخر 10 رسائل فقط
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

            print(f"\nالمساعد: {answer}\n")
            print("-" * 40)

        except Exception as e:
            print(f"❌ صار خطأ: {e}\n")


if __name__ == "__main__":
    main()
