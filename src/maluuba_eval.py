from nlgeval import NLGEval
nlgeval = NLGEval()  # loads the models
references = [["The boy is good", "The boy is good yeah"], ["They came to play", "They came to dance"]]
hypothesis = ["The boy is not good", "They are dancing"]
metrics_dict = nlgeval.compute_metrics(references, hypothesis)
metric_ind_dict = nlgeval.compute_individual_metrics(references[0], hypothesis[0])

print(metrics_dict)
print(metric_ind_dict)