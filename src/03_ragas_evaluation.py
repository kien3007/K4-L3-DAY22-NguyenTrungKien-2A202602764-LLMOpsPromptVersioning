"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.llm_factory import get_llm, get_embeddings
from qa_pairs import QA_PAIRS

# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# TODO: Copy SYSTEM_V1 và SYSTEM_V2 mà bạn đã viết ở file 02_prompt_hub_ab_routing.py
# ⚠️ Cả 2 phải chứa {context}, ví dụ kết thúc bằng "...\n\nContext:\n{context}"
#    Thiếu {context} → LLM không thấy tài liệu, không báo lỗi, faithfulness/context_* rất thấp.
SYSTEM_V1 = (
    "You are a friendly and helpful AI assistant. Answer the question concisely (2-4 sentences) "
    "based strictly on the provided context. If the information is not present in the context, "
    "state that you do not know.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI knowledge analyst. Thoroughly examine the provided context, identify "
    "the relevant facts, and provide a clear, well-structured, and comprehensive answer (3-5 sentences). "
    "Do not speculate or extrapolate beyond the provided context.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str, max_retries: int = 5) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi với cơ chế retry tự động phòng ngừa rớt mạng.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    import time
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)

    last_err = None
    for attempt in range(max_retries):
        try:
            answer = (prompt | llm | StrOutputParser()).invoke({
                "context":  ctx_str,
                "question": question,
            })
            return {"answer": str(answer), "contexts": contexts}
        except Exception as e:
            last_err = e
            wait_time = 2 ** attempt
            print(f"    [Retry {attempt+1}/{max_retries}] Lỗi kết nối: {e}. Thử lại sau {wait_time}s...")
            time.sleep(wait_time)

    # Fallback an toàn nếu rớt mạng hoàn toàn sau 5 lần thử
    print(f"    [Fallback] Dùng câu trả lời an toàn sau khi thử {max_retries} lần: {last_err}")
    fallback_ans = "Based on the provided context, the information is not sufficiently clear."
    return {"answer": fallback_ans, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định với checkpointing từng câu.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    import time
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    cache_path = Path(__file__).parent.parent / "data" / f"cache_rag_outputs_{prompt_version}.json"
    results = []
    if cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached_data, list):
                results = cached_data
                print(f"📦 Đã tìm thấy cache: {len(results)} câu hỏi đã xử lý cho prompt {prompt_version}")
        except Exception:
            results = []

    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        if i <= len(results):
            continue

        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

        # Lưu checkpoint ngay lập tức
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        if config.PROVIDER == "gemini" and i < len(QA_PAIRS):
            time.sleep(4)

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # TODO: Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # TODO: Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    # TODO: Tạo EvaluationDataset từ rag_results
    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    # TODO: Gọi evaluate() với đầy đủ 4 metrics
    from ragas.run_config import RunConfig
    run_cfg = RunConfig(max_workers=2, max_retries=10, timeout=180)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_cfg,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        valid_vals = [float(v) for v in raw if v is not None and not np.isnan(v)]
        scores[key] = float(np.mean(valid_vals)) if valid_vals else 0.0

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # TODO: Tạo vectorstore
    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    evidence_report_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"

    v1_scores = None
    v2_scores = None
    if report_path.exists():
        try:
            old = json.loads(report_path.read_text(encoding="utf-8"))
            if "prompt_v1_scores" in old and len(old["prompt_v1_scores"]) == 4:
                v1_scores = old["prompt_v1_scores"]
                print("📦 Đã tải v1_scores từ báo cáo trước đó.")
            if "prompt_v2_scores" in old and len(old["prompt_v2_scores"]) == 4:
                v2_scores = old["prompt_v2_scores"]
                print("📦 Đã tải v2_scores từ báo cáo trước đó.")
        except Exception:
            pass

    if not v1_scores:
        v1_scores = run_ragas_eval(v1_results, "v1")
        # Lưu checkpoint V1
        interim = {"prompt_v1_scores": v1_scores, "prompt_v2_scores": {}, "target_met": False}
        report_path.write_text(json.dumps(interim, indent=2), encoding="utf-8")

    if not v2_scores:
        v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # TODO: Lưu báo cáo vào data/ragas_report.json và evidence/03_ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")

    evidence_report_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    evidence_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {evidence_report_path}")


if __name__ == "__main__":
    main()
