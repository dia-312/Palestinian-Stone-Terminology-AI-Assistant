import json
import os
import chromadb
from google import genai
from google.genai.types import EmbedContentConfig
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# احذف الـ collection القديمة إذا موجودة وابني من جديد
try:
    chroma_client.delete_collection("stone_terms")
except:
    pass

collection = chroma_client.create_collection(
    name="stone_terms",
    metadata={"hnsw:space": "cosine"}
)


def get_embedding(text: str) -> list:
    """تحويل النص لـ vector باستخدام Gemini embeddings"""
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values


def build_searchable_text(term: dict) -> str:
    """دمج كل معلومات المصطلح في نص واحد قابل للبحث"""
    parts = []
    parts.append(f"المصطلح المحلي: {term.get('term_local', '')}")
    parts.append(f"المصطلح العربي: {term.get('term_arabic', '')}")
    parts.append(f"المصطلح الإنجليزي: {term.get('term_standard', '')}")
    parts.append(f"المعنى: {term.get('meaning_simple', '')}")
    parts.append(f"الوصف العلمي: {term.get('scientific_description', '')}")

    if term.get('synonyms'):
        parts.append(f"المرادفات: {', '.join(term['synonyms'])}")

    if term.get('common_usage'):
        parts.append(f"الاستخدامات: {', '.join(term['common_usage'])}")

    if term.get('notes'):
        parts.append(f"ملاحظات: {term['notes']}")

    return "\n".join(parts)


def ingest_json(json_path: str):
    """قراءة الـ JSON وإدخال كل مصطلح للـ ChromaDB"""
    with open(json_path, "r", encoding="utf-8") as f:
        terms = json.load(f)

    print(f"📂 تم تحميل {len(terms)} مصطلح من الملف")

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for i, term in enumerate(terms):
        searchable_text = build_searchable_text(term)
        embedding = get_embedding(searchable_text)

        documents.append(searchable_text)
        embeddings.append(embedding)
        metadatas.append({
            "term_local": term.get("term_local", ""),
            "term_arabic": term.get("term_arabic", ""),
            "term_standard": term.get("term_standard", ""),
            "term_category": term.get("term_category", ""),
            "meaning_simple": term.get("meaning_simple", ""),
            "notes": term.get("notes", ""),
            "synonyms": ", ".join(term.get("synonyms", [])),
            "common_usage": ", ".join(term.get("common_usage", []))
        })
        ids.append(f"term_{i}")

        print(f"  ✅ ({i+1}/{len(terms)}) تم معالجة: {term.get('term_local', '')}")

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"\n🎉 تم بناء قاعدة البيانات بنجاح! {len(terms)} مصطلح محفوظ في ./chroma_db")


if __name__ == "__main__":
    ingest_json("data/terms.json")
