import banana_model
import gauss_model

experiments = {
    "banana": banana_model.get_problem(
        dim=2, a=20, n0=None, n=100000, tau1=0.0005, tau2=0.0004
    ),
    "gauss": gauss_model.get_problem(dim=10, n=100000, gamma_shape=0.5)
}
experiments["clip-banana"] = experiments["banana"]
experiments["clip-gauss"] = experiments["gauss"]
