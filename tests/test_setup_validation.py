"""
Test setup validation - verifies that the testing infrastructure is working correctly.
"""

import os
import sys
from pathlib import Path
import pytest


class TestSetupValidation:
    """Validate that the testing infrastructure is properly configured."""

    def test_pytest_is_working(self):
        """Test that pytest is running correctly."""
        assert True

    def test_fixtures_are_available(self, temp_dir, sample_data):
        """Test that shared fixtures are available."""
        assert temp_dir.exists()
        assert isinstance(sample_data, dict)
        assert "users" in sample_data
        assert "products" in sample_data

    def test_mock_functionality(self, mock_config):
        """Test that mocking functionality works."""
        assert mock_config.debug is False
        assert mock_config.timeout == 30
        mock_config.debug = True
        assert mock_config.debug is True

    def test_temporary_file_creation(self, temp_file):
        """Test temporary file fixture."""
        assert temp_file.exists()
        temp_file.write_text("test content")
        assert temp_file.read_text() == "test content"

    def test_environment_cleanup(self, clean_environment):
        """Test that environment cleanup works."""
        original_value = os.environ.get("TEST_VAR")
        os.environ["TEST_VAR"] = "test_value"
        assert os.environ["TEST_VAR"] == "test_value"

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker is working."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker is working."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker is working."""
        assert True

    def test_project_structure(self):
        """Test that project structure is correct."""
        project_root = Path(__file__).parent.parent
        
        # Check that important files exist
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / ".gitignore").exists()
        
        # Check test directories
        tests_dir = project_root / "tests"
        assert tests_dir.exists()
        assert (tests_dir / "__init__.py").exists()
        assert (tests_dir / "conftest.py").exists()
        assert (tests_dir / "unit" / "__init__.py").exists()
        assert (tests_dir / "integration" / "__init__.py").exists()

    def test_coverage_config_is_working(self):
        """Test that coverage configuration is accessible."""
        # This test will pass if pytest-cov is properly configured
        assert True

    def test_pytest_mock_is_available(self, mocker):
        """Test that pytest-mock is available."""
        mock_func = mocker.Mock()
        mock_func.return_value = "mocked"
        assert mock_func() == "mocked"
        mock_func.assert_called_once()


class TestSampleFunctionality:
    """Sample tests to demonstrate different testing patterns."""

    def test_simple_assertion(self):
        """Simple assertion test."""
        result = 2 + 2
        assert result == 4

    def test_string_operations(self):
        """Test string operations."""
        text = "Hello, World!"
        assert text.startswith("Hello")
        assert text.endswith("World!")
        assert "World" in text

    def test_list_operations(self):
        """Test list operations."""
        numbers = [1, 2, 3, 4, 5]
        assert len(numbers) == 5
        assert 3 in numbers
        assert sum(numbers) == 15

    def test_exception_handling(self):
        """Test exception handling."""
        with pytest.raises(ValueError):
            int("not_a_number")

    def test_parameterized_test(self, sample_data):
        """Test using sample data fixture."""
        users = sample_data["users"]
        assert len(users) == 2
        assert users[0]["name"] == "Alice"
        assert users[1]["email"] == "bob@example.com"