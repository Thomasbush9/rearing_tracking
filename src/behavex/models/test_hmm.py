from behavex.models.extract_latents import extract_session_latents_for_hmm
from behavex.models.hmm_trainer import HiddenMarkovModelTrainer
from argparse import ArgumentParser
import numpy as np





if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--session_path", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str)

    args = parser.parse_args()

    print("Extracting the latents space:")
    latents = extract_session_latents_for_hmm(
        args.session_path,
        args.model_path
    )

    print("training the HMM")
    # sticky_prior: initial self-transition probability fed to EM.
    # 0.95 → each state expects to stay put for ~1/0.05 = 20 patches = ~1.3s at 62.4fps
    trainer = HiddenMarkovModelTrainer(n_states=8, covariance_type="full")
    trainer.fit([latents], sticky_prior=0.95)
    states, log_probs = trainer.decode_states([latents])

    print("Saving the model and the latents")
    # save them for later: 
    np.save("session_latents.npy", latents)
    trainer.save("hhm_session_1.pkl")





    
