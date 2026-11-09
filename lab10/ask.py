"""Ask micro-rag a question from the command line. Provided complete.

    python3 ask.py "why do databases read whole blocks?"

Echo mode (no LLM configured) prints the assembled prompt — which is the
point: you see exactly what retrieval chose before any model touches it.
Set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL to generate a real answer.
"""

import sys

from micro_rag import Retriever, answer, chunk_corpus

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    retriever = Retriever(chunk_corpus())
    print(answer(" ".join(sys.argv[1:]), retriever))
