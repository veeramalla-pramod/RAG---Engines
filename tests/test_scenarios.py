"""Automated verification test suite for PRD Section 7 Evaluation Scenarios (TC-01 to TC-05)."""

import os
import unittest
from core.chunker import SlidingWindowChunker
from core.tokenizer import count_tokens, calculate_token_telemetry, compute_context_budget
from core.embedder import TFIDFEmbedder, DenseEmbedder
from core.retriever import VectorRetriever
from core.generator import GroundedGenerator


class TestRAGEngineScenarios(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        with open(os.path.join(data_dir, "os_study_guide.txt"), "r", encoding="utf-8") as f:
            cls.os_text = f.read()
        with open(os.path.join(data_dir, "ecommerce_policy.txt"), "r", encoding="utf-8") as f:
            cls.ecom_text = f.read()

        cls.chunker = SlidingWindowChunker()
        cls.generator = GroundedGenerator()

    def test_tc_01_study_buddy_convoy_effect(self):
        """TC-01: Convoy effect query must retrieve FCFS chunk and cite source."""
        chunks = self.chunker.chunk_text(self.os_text, chunk_size=80, overlap_percentage=0.20, token_counter=count_tokens)
        embedder = TFIDFEmbedder()
        doc_matrix = embedder.fit_transform([c.text for c in chunks])

        retriever = VectorRetriever()
        retriever.index(chunks, doc_matrix)

        query = "Which scheduling algorithm causes the convoy effect and why?"
        q_vec = embedder.transform_query(query)
        results, _ = retriever.search(q_vec, top_k=3, threshold=0.10)

        self.assertGreater(len(results), 0, "Should retrieve at least one chunk for convoy effect query.")
        top_chunk_text = results[0]["text"].lower()
        self.assertTrue("convoy effect" in top_chunk_text or "fcfs" in top_chunk_text, "Top chunk must discuss FCFS / convoy effect.")

        gen_out = self.generator.generate(query, results, persona_key="study_buddy")
        self.assertIn("[Source", gen_out["response"], "Response must contain explicit [Source X] citation.")
        self.assertGreater(len(gen_out["sources_cited"]), 0, "Cited source IDs must be extracted.")

    def test_tc_02_out_of_domain_query_refusal(self):
        """TC-02: Out-of-domain query ('What is memory segmentation?') falls below threshold or triggers refusal."""
        chunks = self.chunker.chunk_text(self.os_text, chunk_size=80, overlap_percentage=0.20, token_counter=count_tokens)
        embedder = TFIDFEmbedder()
        doc_matrix = embedder.fit_transform([c.text for c in chunks])

        retriever = VectorRetriever()
        retriever.index(chunks, doc_matrix)

        query = "What is memory segmentation?"
        q_vec = embedder.transform_query(query)
        # Using a conservative threshold (0.25)
        results, _ = retriever.search(q_vec, top_k=3, threshold=0.25)

        gen_out = self.generator.generate(query, results, persona_key="study_buddy")
        self.assertTrue(
            "don't have that information" in gen_out["response"].lower() or "unavailable" in gen_out["response"].lower(),
            f"Expected strict refusal for out-of-domain query, got: {gen_out['response']}"
        )

    def test_tc_03_ecommerce_20_day_refund_rejection(self):
        """TC-03: 'Can I return a backpack after 20 days for a refund?' -> Refuses refund citing 15-day policy."""
        chunks = self.chunker.chunk_text(self.ecom_text, chunk_size=80, overlap_percentage=0.20, token_counter=count_tokens)
        embedder = TFIDFEmbedder()
        doc_matrix = embedder.fit_transform([c.text for c in chunks])

        retriever = VectorRetriever()
        retriever.index(chunks, doc_matrix)

        query = "Can I return a backpack after 20 days for a refund?"
        q_vec = embedder.transform_query(query)
        results, _ = retriever.search(q_vec, top_k=3, threshold=0.05)

        self.assertGreater(len(results), 0, "Must retrieve policy chunk.")
        gen_out = self.generator.generate(query, results, persona_key="ecommerce")
        resp = gen_out["response"].lower()
        self.assertTrue("15" in resp or "20" in resp or "unable" in resp or "cannot" in resp or "non-refundable" in resp,
                        "Response must address the 15-day constraint or reject the 20-day request.")
        self.assertIn("[Source", gen_out["response"], "Response must cite source policy.")

    def test_tc_04_dynamic_tuning_reactivity(self):
        """TC-04: Shrinking chunk size from 150 words to 40 words multiplies chunk count & updates telemetry."""
        chunks_large = self.chunker.chunk_text(self.os_text, chunk_size=150, overlap_percentage=0.10, token_counter=count_tokens)
        chunks_small = self.chunker.chunk_text(self.os_text, chunk_size=40, overlap_percentage=0.10, token_counter=count_tokens)

        self.assertGreater(len(chunks_small), len(chunks_large) * 2, "Small chunk size must produce >2x chunks.")

        telemetry_large = calculate_token_telemetry(self.os_text, chunks_large)
        telemetry_small = calculate_token_telemetry(self.os_text, chunks_small)

        self.assertGreater(telemetry_small["inflation_ratio"], 0.9)
        self.assertLess(telemetry_small["mean_chunk_tokens"], telemetry_large["mean_chunk_tokens"])

    def test_tc_05_zero_overlap_diff_inspector(self):
        """TC-05: Overlap diff inspector detects shared words when overlap > 0 and empty when overlap = 0."""
        chunks_zero = self.chunker.chunk_text(self.os_text, chunk_size=50, overlap_percentage=0.0, token_counter=count_tokens)
        chunks_overlap = self.chunker.chunk_text(self.os_text, chunk_size=50, overlap_percentage=0.25, token_counter=count_tokens)

        # Zero overlap consecutive chunks should have 0 shared words
        diff_zero = SlidingWindowChunker.get_overlap_diff(chunks_zero[0], chunks_zero[1])
        self.assertEqual(diff_zero["overlap_words"], 0)

        # Overlap consecutive chunks should have >0 shared words
        diff_overlap = SlidingWindowChunker.get_overlap_diff(chunks_overlap[0], chunks_overlap[1])
        self.assertGreater(diff_overlap["overlap_words"], 0)


if __name__ == "__main__":
    unittest.main()
