import os

import pytest
from sklearn.exceptions import NotFittedError

import neurometry.datasets.synthetic as synthetic
from neurometry.estimators.topology.topology_classifier import TopologyClassifier

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs


class BaseTopologyTest:
    """Base class for topology classifier tests."""

    num_points = 700
    encoding_dim = 10
    fano_factor = 0.1
    num_samples = 100
    homology_dimensions = (0, 1)

    @classmethod
    def setup_class(cls):
        """Set up common data for all tests."""
        cls.classifier = TopologyClassifier(
            num_samples=cls.num_samples,
            fano_factor=cls.fano_factor,
            homology_dimensions=cls.homology_dimensions,
            reduce_dim=True,
        )

        cls.circle_data = cls.generate_circle_data()
        cls.sphere_data = cls.generate_sphere_data()
        cls.torus_data = cls.generate_torus_data()

    @classmethod
    def generate_circle_data(cls):
        task_points, _ = synthetic.hypersphere(1, cls.num_points)
        noisy_points, _ = synthetic.synthetic_neural_manifold(
            points=task_points,
            encoding_dim=cls.encoding_dim,
            nonlinearity="sigmoid",
            scales=5 * gs.random.rand(cls.encoding_dim),
            fano_factor=cls.fano_factor,
        )
        return noisy_points

    @classmethod
    def generate_sphere_data(cls):
        task_points, _ = synthetic.hypersphere(2, cls.num_points)
        noisy_points, _ = synthetic.synthetic_neural_manifold(
            points=task_points,
            encoding_dim=cls.encoding_dim,
            nonlinearity="sigmoid",
            scales=5 * gs.random.rand(cls.encoding_dim),
            fano_factor=cls.fano_factor,
        )
        return noisy_points

    @classmethod
    def generate_torus_data(cls):
        task_points, _ = synthetic.hypertorus(2, cls.num_points)
        noisy_points, _ = synthetic.synthetic_neural_manifold(
            points=task_points,
            encoding_dim=cls.encoding_dim,
            nonlinearity="sigmoid",
            scales=5 * gs.random.rand(cls.encoding_dim),
            fano_factor=cls.fano_factor,
        )
        return noisy_points


class TestTopologyClassifier(BaseTopologyTest):
    """Unit tests for the topology classifier."""

    def test_invalid_input(self):
        """Test classifier with invalid input data."""
        invalid_data = "invalid_input"
        with pytest.raises(ValueError):
            self.classifier.fit(invalid_data)

    def test_fit_and_predict_circle(self):
        """Test that the fit method runs without errors and predict on circle data."""
        try:
            self.classifier.fit(self.circle_data)
        except Exception as e:
            pytest.fail(f"Fit method raised an exception: {e}")

        # If fit is successful, test the prediction
        prediction = self.classifier.predict(self.circle_data)
        assert prediction[0] == 1, "Prediction for circle data should be 1 (circle)"

    # def test_predict_sphere(self):
    #     """Test prediction on sphere data."""
    #     self.classifier.fit(self.sphere_data)
    #     prediction = self.classifier.predict(self.sphere_data)
    #     assert prediction[0] == 2, "Prediction for sphere data should be 2 (sphere)"

    def test_predict_torus(self):
        """Test prediction on torus data."""
        self.classifier.fit(self.torus_data)
        prediction = self.classifier.predict(self.torus_data)
        assert prediction[0] == 3, "Prediction for torus data should be 3 (torus)"

    def test_not_fitted_error(self):
        """Test that predicting before fitting raises an error."""
        classifier = TopologyClassifier(
            num_samples=self.num_samples,
            fano_factor=self.fano_factor,
            homology_dimensions=self.homology_dimensions,
            reduce_dim=True,
        )
        with pytest.raises(NotFittedError):
            classifier.predict(self.circle_data)
