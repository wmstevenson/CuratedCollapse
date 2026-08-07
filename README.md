# Model collapse dynamics in Tabular Data
This repository holds the codebase and results for our experiments with model collapse in Tabular Data. The results are presented in our paper;
```
@inproceedings{Stevenson2026,
  author = {William Martin Stevenson, Anne Gerdes, Lukas Galke Poech, Anton Danholt Lautrup},
  title = {Curating the Collapse: How Sample-Level Auditing Accelerates Model Collapse in Synthesis of Tabular Data},
  booktitle = {Proceedings of the ninth AAAI/ACM Conference on AI, Ethics, and Society},
  series = {{AIES} 2026},
  address = {Malmö, Sweeden},
  year = {2026},
  note = {[Accepted: 17-07-2026]},
}
```
Our results show how unchecked synthetic-data reuse can result in decreased output diversity in subsequent models and reduced downstream utility. We also observe and that aggressive post-hoc auditing (1) at best decreases diversity, and at worst does so while also accelerating the model collapse.

> (1) Ahmed M. Alaa, Boris van Breugel, Evgeny S. Saveliev, Mihaela van der Schaar. How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models. ICML 2022: 290-306. [OpenReveiw](https://openreview.net/forum?id=8qWazUd8Jm)
> 
## Reproducing the results
The code is accessed from the `main.py` script through a runner, which takes parameters (experiment configuration and seed) for every experiment. 

However, nothing is special about the codebase, which is essentially just a loop over several calls to Synthcity. The essence of the experimental pipeline can be summarised in the pseudocode 
```python
original_data = df_train
training_data = df_train
for i in range(NUM_GENS):
    syn_model = Plugins().get("ctgan")
    syn_model.fit(training_data)

    temp_syn_data = []
    while len(temp_syn_data) < original_data:
        queries = syn_model.generate(count = NUM_GENERATE)

        accepted_mask = audit_alpha_precision(original_data, query, audit_level)
        temp_syn_data.append(queries[accepted_mask])

    # Update training data for next generation
    training_data = temp_syn_data[:len(original_data)]
    evaluate(original_data, training_data)    
```

We did make one adjustment to the way alpha precision is calculated in Synthcity and incorporated the audit function. 

```python
class AlphaPrecision(StatisticalEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="authenticity", **kwargs)

    @staticmethod
    def name() -> str:
        return "alpha_precision"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def metrics(
        self,
        X: np.ndarray,
        X_syn: np.ndarray,
        emb_center: Optional[np.ndarray] = None,
    ) -> Tuple:
        if emb_center is None:
            emb_center = np.mean(X, axis=0)

        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)

        Radii = np.quantile(
            np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas
        )

        synth_center = np.mean(X_syn, axis=0)

        alpha_precision_curve = []
        beta_coverage_curve = []

        synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
        real_to_real, _ = nbrs_real.kneighbors(X)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)

        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        real_synth_closest = X_syn[real_to_synth_args]

        real_synth_closest_d = np.sqrt(
            np.sum((real_synth_closest - synth_center) ** 2, axis=1)
        )

        closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

        for k in range(len(Radii)):
            precision_mask = synth_to_center <= Radii[k]
            alpha_precision = np.mean(precision_mask)

            beta_coverage = np.mean(
                (
                    (real_to_synth <= real_to_real)
                    * (real_synth_closest_d <= closest_synth_Radii[k])
                )
            )

            alpha_precision_curve.append(alpha_precision)
            beta_coverage_curve.append(beta_coverage)

        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen)

        Delta_precision_alpha = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(alpha_precision_curve))
        ) / np.sum(alphas)

        Delta_coverage_beta = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(beta_coverage_curve))
        ) / np.sum(alphas)

        return (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        )

    def audit_alpha_precision(
        self,
        X: DataLoader,
        X_syn: DataLoader,
        alpha: float,
    ) -> np.ndarray:

        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        oneclass_model = self._get_oneclass_model(X_)

        X_ = self._oneclass_predict(oneclass_model, X_)
        X_syn_ = self._oneclass_predict(oneclass_model, X_syn_)

        emb_center = oneclass_model.c.detach().cpu().numpy()

        real_dists = np.sqrt(np.sum((X_ - emb_center) ** 2, axis=1))
        radius_alpha = np.quantile(real_dists, alpha)

        synth_to_center = np.sqrt(np.sum((X_syn_ - emb_center) ** 2, axis=1))
        precision_audit_mask = synth_to_center <= radius_alpha

        return precision_audit_mask

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:

        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        oneclass_model = self._get_oneclass_model(X_)
        X_ = self._oneclass_predict(oneclass_model, X_)
        X_syn_ = self._oneclass_predict(oneclass_model, X_syn_)
        emb_center = oneclass_model.c.detach().cpu().numpy()

        (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            delta_precision_alpha,
            delta_coverage_beta,
            authenticity,
        ) = self.metrics(X_, X_syn_, emb_center=emb_center)

        return {
            "alphas": np.asarray(alphas).tolist(),
            "alpha_precision_curve": np.asarray(alpha_precision_curve).tolist(),
            "beta_coverage_curve": np.asarray(beta_coverage_curve).tolist(),
            "delta_precision_alpha": float(delta_precision_alpha),
            "delta_coverage_beta": float(delta_coverage_beta),
            "authenticity": float(authenticity),
        }
