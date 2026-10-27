# Evaluation

The evaluation framework measures the pipeline against a **labelled benchmark**. Every number shown in the app or in this repository comes from an actual run stored in SQLite (`experiments`, `experiment_runs`, `run_results`) and can be exported to CSV and JSON. Nothing is hard-coded.

## Benchmark format

```json
{
  "qid": "Q09",
  "question": "What is the first response time and resolution target for a P1 critical incident?",
  "expected_answer": "P1 incidents get a first response within 15 minutes ... within 4 hours.",
  "answerable": true,
  "relevant_documents": ["it_support_incident_escalation_policy.pdf"],
  "evidence": ["First response within 15 minutes, 24 hours a day. Target resolution within 4 hours."],
  "relevant_chunk_ids": [],
  "key_facts": ["15 minutes", "4 hours"],
  "category": "it-ops"
}
```

- `evidence`: verbatim passages that answer the question (the primary relevance label).
- `relevant_chunk_ids`: optional exact chunk ids; they take precedence, but only make sense for the chunking config they were labelled under.
- `relevant_documents`: used for document-level recall, and as a fallback label.
- `key_facts`: short strings a correct answer must contain (numbers, names, dates).
- `answerable: false` marks questions the corpus can't answer, to test abstention.

Benchmarks can be created and edited in the Evaluation lab (inline editor), imported from JSON or CSV (`|` separates list values, `||` separates evidence passages) and exported. The bundled **sample benchmark** has 22 questions about the fictional demo documents: 18 answerable (one needs two documents) and 4 unanswerable. A test verifies that every evidence passage occurs verbatim in the sample documents.

## Relevance definition

A retrieved chunk is **relevant** to a question if:

1. the question has `relevant_chunk_ids` and the chunk id is one of them; otherwise
2. the question has `evidence` and the chunk contains one of the passages. After lower-casing and whitespace normalisation, the passage is either a substring of the chunk or shares a contiguous run of at least **60 %** of its characters with it (`difflib.SequenceMatcher.find_longest_match`). The partial rule credits a chunk when a chunk boundary cuts a passage; otherwise
3. the question has `relevant_documents` and the chunk belongs to one of them.

Semantic similarity is **never** used as ground truth.

## Retrieval metrics (k = top-k)

| Metric | Definition |
|---|---|
| Precision@k | relevant chunks in the top k ÷ k. The denominator is k even if fewer chunks are returned. Its maximum is below 1 when a question has fewer relevant chunks than k, so compare it across configs with the same k |
| Recall@k | share of the question's evidence passages found in at least one top-k chunk (or labelled chunk ids / documents found) |
| Hit rate@k | 1 if any top-k chunk is relevant |
| MRR | mean over questions of 1 / rank of the first relevant chunk (0 if none) |
| nDCG@k | binary-relevance DCG ÷ ideal DCG |
| Doc recall@k | share of `relevant_documents` represented in the top k |
| Gate pass (answerable) | share of answerable questions whose best cosine ≥ the evidence floor |
| Gate block (unanswerable) | share of unanswerable questions whose best cosine < the floor |
| Latency | query embedding + search time per question (mean, p95), on the machine that ran it |

Retrieval metrics are averaged over **answerable** questions that have labels. Unanswerable questions only contribute to the gate metrics. Retrieval evaluation doesn't need an LLM.

## Answer metrics

| Metric | Definition | Type |
|---|---|---|
| Answer rate (answerable) | share of answerable questions that got a non-abstaining answer | deterministic |
| Correct abstention (unanswerable) | share of unanswerable questions where the gate blocked or the model abstained | deterministic |
| Citation validity | valid labels ÷ all `[Sn]` labels in an answer, averaged over answers that cite anything. Valid = maps to a chunk in the prompt that still exists in the DB | deterministic |
| Citation coverage | share of produced (non-abstaining) answers with ≥ 1 valid citation | deterministic |
| Cited-evidence precision | share of cited chunks that are relevant per the labels, i.e. whether the model cited the *right* evidence | deterministic |
| Key-fact recall | share of `key_facts` present in the answer after normalisation (case, punctuation, thousands separators) | deterministic |
| Token F1 | SQuAD-style token overlap with `expected_answer`. Penalises verbose but correct answers | deterministic |
| Lexical faithfulness | share of answer sentences (≥ 3 content words) whose words are ≥ 60 % present in their cited chunks (or all retrieved chunks if the sentence cites nothing) | heuristic proxy |
| Context relevance | mean query/chunk cosine over the retrieved context | embedding proxy |
| Judge faithfulness / correctness | optional LLM rating 1–5 rescaled to 0–1; judge model and prompt version (`judge-v1`) are stored with the experiment | **model estimate** |

**About model-graded scores:** an LLM judge is another model's opinion. With the same small local model judging its own answers, scores are biased and noisy. They are reported separately, never mixed into the deterministic metrics, and are left empty when the judge's output can't be parsed (never guessed).

**About lexical faithfulness:** it catches answers that introduce content not present in the sources (e.g. invented numbers), but it can't detect a contradiction phrased with the same words, and it can penalise valid paraphrases.

## Operational metrics

Generation time, end-to-end time (retrieval + gate + prompt + generation + citation validation), first-token latency when streaming, prompt and completion tokens as reported by the provider, and estimated cost **only** if the provider reports tokens and `RAG_PRICE_PER_1K_INPUT/OUTPUT` are set. For local Ollama no cost is reported.

## Experiments

- **Retrieval grid:** strategies × top-k × chunk size/overlap × embedding model. Each configuration is one *run*; per-question rows keep the retrieved chunk ids, relevance flags and latency.
- **Answer evaluation:** one run per prompt variant, with the current retrieval settings and LLM.
- Every experiment stores its environment: Python, OS, device, package versions, embedding models, LLM model and Ollama digest, judge model, temperature and evidence floor.
- Exports: `outputs/experiments/experiment_<id>_results.csv` (per-question rows) and `experiment_<id>_metadata.json` (config grid, environment, aggregate metrics per run).

## Results

See the README section *Measured results* for the numbers from the runs made while building this project, and the Experiment history page for anything you run yourself.
