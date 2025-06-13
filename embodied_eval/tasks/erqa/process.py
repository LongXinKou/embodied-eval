ERQA_METRICS = {"acc"}

def erqa_doc_to_visual(doc, dataset_kwargs=None):
    return doc[dataset_kwargs["images"]]

def erqa_doc_to_text(doc, dataset_kwargs=None):
    return doc[dataset_kwargs["question"]]

def erqa_process_results(doc, results, dataset_kwargs=None):
    target = doc[dataset_kwargs["answer"]]
    acc = results[0].strip().upper() == target
    result_dict = {"target": target, "where2place_acc": acc}
    return {key: value for key, value in result_dict.items()}