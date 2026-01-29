# RAG Pipeline for Claims Denial QA

---

While replacing the retrieval layer with embeddings and adding a relevance-aware
answer synthesizer, my primary points of consideration were:

- Which model should we use for embeddings?
- What is an appropriate similarity threshold?
- How many documents should we provide the answer synthesizer?
- What model should we use for answer synthesis?

## Embedding Model Selection

Since this is running locally and we're building the embeddings at runtime,
using a model with fewer dimensions and lower latency is preferable. However,
if we were running this in a production environment, using a larger, slower
model would be acceptable, as we would be able to build the embeddings in an
asynchronous pipeline that looks something like:

```
| Claim data comes in | ---> | Calculate embedding | ---> | Write to vector DB |
```

In the current setup, we calculate the embeddings for all the claim data at once,
plus the embedding for the question. In the case where we have this async data
pipeline, we would only need to calculate the embedding of the question at the
time of inference.

For lower latency, I chose the `all-MiniLM-L6-v2`, but for a production setup
I would choose `all-mpnet-base-v2` (double the size).

If we had the time and resources to build our own embedding model that's been
tuned specifically for medical data, then that would also be something for us to
explore, but only if the open source offerings are not performing well enough.

## Similarity Threshold & Document Selection

I observed in the baseline agent that only the top 3 documents (as measured by
TF-IDF and cosine similarity) were included in the answer. However, there may
very well be more than 3 related documents.

Instead of choosing an arbitrary number of documents to include, I instead opted
to use a similarity threshold so that all sufficiently relevant documents are
included in the answer synthesis. With some manual experimentation, I found that
a similarity threshold of `0.5` resulted in good performance on the evaluation.

If we have too low of a threshold, we'd end up with a lot of documents requiring
a model with a larger context window to hold all the relevant documents. This
would both increase latency and costs.

## Answer Synthesis Model Selection

My goal in selecting a model for answer synthesis was to identify the cheapest
and smallest model that would still produce useful answers to the end-user.
After experimenting with `gpt-5.2`, `gpt-5-mini`, and `gpt-5-nano`, I found
that `gpt-5-nano` was both the fastest and still provided coherent responses to
the user's queries.
