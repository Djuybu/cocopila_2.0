# Graph status

TODO: GraphRetriever not implemented yet.

The only existing graph direction is notes/22_09_26.txt (convert chunks to a graph
database and investigate Neo4j). There is no graph schema, graph dataset, Neo4j
connection or working graph retrieval code to move.

artifacts/indexes/graph is reserved for future graph artifacts. No invented graph
implementation, graph CLI, model config or training placeholders have been added.

A future implementation must implement BaseRetriever.retrieve(query, top_k),
return the same candidate contract, resolve nodes back to authoritative official
chunk/document IDs, and join CandidateGenerator as an independent source.
It must not move reranking or selection into graph retrieval. Define entity and
relation schemas against the actual corpus/competition rules before implementing.
Existing multilingual/medical_ner.py remains an unfinished interface, not a graph
entity extraction implementation.
