from parameters import parameters, dotdict
from train import train
def run_experiments():
    modifications = [
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.01,
            "suffix": "_run2"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.03,
            "suffix": "_run2"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.05,
            "suffix": "_run2"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.07,
            "suffix": "_run2"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.1,
            "suffix": "_run2"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.2,
            "suffix": "_run2"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.3,
            "suffix": "_run2"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.5,
            "suffix": "_run2"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.01,
            "suffix": "_run3"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.03,
            "suffix": "_run3"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.05,
            "suffix": "_run3"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.07,
            "suffix": "_run3"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.1,
            "suffix": "_run3"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.2,
            "suffix": "_run3"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.3,
            "suffix": "_run3"
        },
        {
            "loss_beta_plan": "paper",
            "zero_supp_version": "weighted_root",
            "beta": 0.5,
            "suffix": "_run3"
        }
    ]

    for mod in modifications[1::2]:
        params = parameters.copy()
        name = f'SAE_{mod["zero_supp_version"]}_{mod["loss_beta_plan"]}_{mod["beta"]}'
        if "dropout" in mod.keys():
            name += "_dropout"
        params["name"] = name
        for key, val in mod.items():
            params[key] = val
        params = dotdict(params)
        print(params)
        train(parameters=params)
    
if __name__ == "__main__":
    run_experiments()