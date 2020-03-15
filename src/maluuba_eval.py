from nlgeval import NLGEval
nlgeval = NLGEval()  # loads the models
references = [["The boy is good", "The boy is good yeah"], ["They came to play", "They came to dance"]]
hypothesis = ["The boy is not good", "They are dancing"]
metrics_dict = nlgeval.compute_individual_metrics(references, hypothesis)