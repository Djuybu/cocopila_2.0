"""Document-disjoint split checks; no random split is silently introduced."""


def validate_split_leakage(splits):
    owners = {}
    for split, rows in splits.items():
        for row in rows:
            doc_id = row["doc_id"]
            if doc_id in owners and owners[doc_id] != split:
                raise ValueError(f"Document leakage: {doc_id} in {owners[doc_id]} and {split}")
            owners[doc_id] = split
