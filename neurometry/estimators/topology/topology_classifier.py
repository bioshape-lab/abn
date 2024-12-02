import os

import numpy as np
import torch
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import neurometry.datasets.synthetic as synthetic

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs


class TopologyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        num_samples,
        fano_factor,
        homology_dimensions=(0, 1, 2),
        reduce_dim=False,
    ):
        self.num_samples = num_samples
        self.fano_factor = fano_factor
        self.homology_dimensions = homology_dimensions
        self.reduce_dim = reduce_dim
        self.classifier = RandomForestClassifier()

    def _generate_ref_data(self, input_data):
        """Generate reference synthetic point clouds to train the classifier.

        Parameters
        ----------
        input_data : array-like, shape=[num_points, encoding_dim]
            Input data to generate the reference point clouds.

        Returns
        -------
        ref_point_clouds : list of array-like, shape=[num_points, encoding_dim]
            Reference point clouds.
        ref_labels : array-like, shape=[num_samples]
            Reference labels. 0 for null point clouds, 1 for circle, 2 for sphere, 3 for torus.
        """
        num_points = input_data.shape[0]
        encoding_dim = input_data.shape[1]

        rng = np.random.default_rng(seed=0)
        null_point_clouds = [
            np.array([rng.permutation(row) for row in input_data])
            for _ in range(self.num_samples)
        ]

        circle_task_points, _ = synthetic.hypersphere(1, num_points)
        circle_point_clouds = []
        for _ in range(self.num_samples):
            circle_noisy_points, _ = synthetic.synthetic_neural_manifold(
                points=circle_task_points,
                encoding_dim=encoding_dim,
                nonlinearity="sigmoid",
                scales=5 * gs.random.rand(encoding_dim),
                fano_factor=self.fano_factor,
            )
            circle_point_clouds.append(circle_noisy_points)

        sphere_task_points, _ = synthetic.hypersphere(2, num_points)
        sphere_point_clouds = []
        for _ in range(self.num_samples):
            sphere_noisy_points, _ = synthetic.synthetic_neural_manifold(
                points=sphere_task_points,
                encoding_dim=encoding_dim,
                nonlinearity="sigmoid",
                scales=5 * gs.random.rand(encoding_dim),
                fano_factor=self.fano_factor,
            )
            sphere_point_clouds.append(sphere_noisy_points)

        torus_task_points, _ = synthetic.hypertorus(2, num_points)
        torus_point_clouds = []
        for _ in range(self.num_samples):
            torus_noisy_points, _ = synthetic.synthetic_neural_manifold(
                points=torus_task_points,
                encoding_dim=encoding_dim,
                nonlinearity="sigmoid",
                scales=5 * gs.random.rand(encoding_dim),
                fano_factor=self.fano_factor,
            )
            torus_point_clouds.append(torus_noisy_points)
        null_point_labels = np.zeros(self.num_samples)
        circle_labels = np.ones(self.num_samples)
        sphere_labels = 2 * np.ones(self.num_samples)
        torus_labels = 3 * np.ones(self.num_samples)
        ref_labels = np.concatenate(
            [
                null_point_labels,
                circle_labels,
                sphere_labels,
                torus_labels,
            ]
        )

        ref_point_clouds = [
            *null_point_clouds,
            *circle_point_clouds,
            *sphere_point_clouds,
            *torus_point_clouds,
        ]

        return ref_point_clouds, ref_labels

    def _compute_topo_features(self, diagrams):
        """Compute topological features from persistence diagrams.

        Parameters
        ----------
        diagrams : list of array-like, shape=[num_diagrams, num_points, 3]
            Persistence diagrams.

        Returns
        -------
        topo_features : array-like, shape=[num_diagrams, self.homology_dimensions]
            Topological features, e.g. persistence entropy.
        """
        PE = PersistenceEntropy()
        return PE.fit_transform(diagrams)

    def fit(self, X, y=None):
        """Fit the classifier topological features extracted from persistence diagrams of synthetic data.

        Parameters
        ----------
        X : array-like, shape=[num_points, encoding_dim]
            Input data.
        y : array-like, shape=[num_samples]
            Labels. Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if not isinstance(X, np.ndarray | torch.Tensor):
            raise ValueError(
                f"Expected array-like input for X, but got {type(X).__name__}."
            )

        ref_point_clouds, ref_labels = self._generate_ref_data(X)
        self.ref_labels = ref_labels
        if self.reduce_dim:
            pca = PCA(n_components=10)
            ref_point_clouds = [
                pca.fit_transform(point_cloud) for point_cloud in ref_point_clouds
            ]
        ref_diagrams = compute_persistence_diagrams(
            ref_point_clouds, homology_dimensions=self.homology_dimensions
        )
        ref_features = self._compute_topo_features(ref_diagrams)
        self.ref_features = ref_features
        X_ref_train, X_ref_valid, y_ref_train, y_ref_valid = train_test_split(
            ref_features, ref_labels
        )
        self.classifier.fit(X_ref_train, y_ref_train)
        print(f"Classifier score: {self.classifier.score(X_ref_valid, y_ref_valid)}")
        return self

    def predict(self, X):
        """Predict the topology class of the input data.

        Parameters
        ----------
        X : array-like, shape=[num_points, encoding_dim]
            Input data.

        Returns
        -------
        y_pred : array-like, shape=[num_points]
            Predicted topology class.

        """
        if self.reduce_dim:
            pca = PCA(n_components=10)
            X = pca.fit_transform(X)
        diagram = compute_persistence_diagrams(
            [X], homology_dimensions=self.homology_dimensions
        )
        features = self._compute_topo_features(diagram)
        self.features = features
        prediction = self.classifier.predict(features)
        label_map = {0: "null", 1: "circle", 2: "sphere", 3: "torus"}
        prediction_label = label_map.get(prediction[0], "unknown")
        print(f"Predicted topology: {prediction_label}")

        return prediction

    def plot_topo_feature_space(self):
        """Plot the topological feature space of the reference data."""
        import plotly.graph_objects as go

        color_map = {
            0: "black",
            1: "red",
            2: "blue",
            3: "green",
        }
        names = {0: "null", 1: "circle", 2: "sphere", 3: "torus"}

        fig = go.Figure()

        for label in np.unique(self.ref_labels):
            mask = self.ref_labels == label
            fig.add_trace(
                go.Scatter3d(
                    x=self.ref_features[mask, 0],
                    y=self.ref_features[mask, 1],
                    z=self.ref_features[mask, 2],
                    mode="markers",
                    name=names[label],
                    marker=dict(size=3, color=color_map[label]),
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=self.features[:, 0],
                y=self.features[:, 1],
                z=self.features[:, 2],
                mode="markers",
                name="Input data",
                marker=dict(size=5, color="orange"),
            )
        )
        fig.show()


def compute_persistence_diagrams(
    representations,
    homology_dimensions=(0, 1, 2),
    coeff=2,
    metric="euclidean",
    weighted=False,
    n_jobs=-1,
):
    if weighted:
        WR = WeightedRipsPersistence(
            metric=metric,
            homology_dimensions=homology_dimensions,
            coeff=coeff,
        )
        diagrams = WR.fit_transform(representations)
    else:
        VR = VietorisRipsPersistence(
            metric=metric,
            homology_dimensions=homology_dimensions,
            coeff=coeff,
            reduced_homology=False,
            n_jobs=n_jobs,
        )
        diagrams = VR.fit_transform(representations)
    return diagrams


def _shuffle_entries(data, rng):
    return np.array([rng.permutation(row) for row in data])


def compute_diagrams_shuffle(X, num_shuffles, seed=0, homology_dimensions=(0, 1)):
    rng = np.random.default_rng(seed)
    shuffled_Xs = [_shuffle_entries(X, rng) for _ in range(num_shuffles)]
    return compute_persistence_diagrams(
        [X, *shuffled_Xs], homology_dimensions=homology_dimensions
    )
