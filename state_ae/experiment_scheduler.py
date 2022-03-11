from parameters import parameters, dotdict
from train import train
def run_experiments():
    modifications = [
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "root",
            "beta": 0.03
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "root",
            "beta": 0.1
        },
        {
            "loss_beta_plan": "increase",
            "zero_supp_version": "root",
            "beta": 0.05
        },
        {
            "loss_beta_plan": "increase",
            "zero_supp_version": "root",
            "beta": 0.1
        },
        {
            "loss_beta_plan": "increase",
            "zero_supp_version": "root",
            "beta": 0.3
        },
        {
            "loss_beta_plan": "increase",
            "zero_supp_version": "root",
            "beta": 0.7
        },
    ]

    for mod in modifications:
        params = parameters.copy()
        params["name"] = f'SAE_{mod["zero_supp_version"]}_{mod["loss_beta_plan"]}_{mod["beta"]}'
        for key, val in mod.items():
            params[key] = val
        params = dotdict(params)
        train(parameters=params)
    
if __name__ == "__main__":
    run_experiments()