"""
Verification tests for the FAQ chatbot.

Run with: python test_chatbot.py
Covers exact matches, paraphrased/similar questions, off-topic fallback,
and edge cases (empty input, gibberish).
"""
from chatbot import FAQChatbot, FALLBACK_MESSAGE

# (user query, expected FAQ question it should match to, should_be_confident)
TEST_CASES = [
    # Exact / near-exact matches
    ("How do I reset my password?", "How do I reset my password?", True),
    ("What is CloudSync?", "What is CloudSync?", True),

    # Paraphrased matches (the real test of the similarity matching)
    ("I forgot my password, how can I reset it?", "How do I reset my password?", True),
    ("How much does it cost per month?", "How much does CloudSync cost?", True),
    ("How can I cancel my plan?", "How do I cancel my subscription?", True),
    ("Do you have a mobile app?", "Which devices are supported?", True),
    ("Is my data safe and encrypted?", "Is my data encrypted?", True),
    ("Can I get my money back?", "Can I get a refund?", True),
    ("How do I get in touch with support?", "How do I contact customer support?", True),
    ("I deleted a file by accident, can I get it back?", "How do I recover a deleted file?", True),

    # Off-topic / should fall back
    ("What's the weather like today?", None, False),
    ("Can you write me a poem?", None, False),
    ("What is the capital of France?", None, False),
]


def run_tests():
    bot = FAQChatbot()
    passed, failed = 0, 0

    print(f"Running {len(TEST_CASES)} test cases against FAQChatbot...\n")
    print(f"{'Query':<50} {'Score':<8} {'Confident':<10} {'Result'}")
    print("-" * 90)

    for query, expected_question, should_be_confident in TEST_CASES:
        result = bot.respond_verbose(query)
        is_confident = result["is_confident"]
        matched_q = result["matched_question"]

        ok = is_confident == should_be_confident
        if ok and should_be_confident:
            # also check it matched the *right* FAQ, not just *a* confident one
            ok = matched_q == expected_question

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"{query[:48]:<50} {result['score']:<8} {str(is_confident):<10} {status}")
        if not ok:
            print(f"   -> matched: {matched_q!r}, expected: {expected_question!r}")

    # Edge cases
    print("\nEdge cases:")
    empty_response = bot.respond("")
    print(f"Empty input handled: {'PASS' if empty_response else 'FAIL'}")

    gibberish_response = bot.respond("asdkfj qpwoe zzxcv")
    gibberish_ok = FALLBACK_MESSAGE in gibberish_response
    print(f"Gibberish falls back correctly: {'PASS' if gibberish_ok else 'FAIL'}")
    if gibberish_ok:
        passed += 1
    else:
        failed += 1

    print(f"\n{passed}/{passed + failed} test cases passed.")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
