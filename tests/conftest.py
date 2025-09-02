"""
Shared pytest fixtures for the test suite.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_directory = tempfile.mkdtemp()
    yield Path(temp_directory)
    shutil.rmtree(temp_directory)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp()
    yield Path(path)
    os.close(fd)
    os.unlink(path)


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = Mock()
    config.debug = False
    config.timeout = 30
    config.max_retries = 3
    return config


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ],
        "products": [
            {"id": 101, "name": "Widget A", "price": 9.99},
            {"id": 102, "name": "Widget B", "price": 19.99},
        ]
    }


@pytest.fixture
def mock_database():
    """Mock database connection."""
    db = Mock()
    db.connect.return_value = True
    db.execute.return_value = Mock(fetchall=Mock(return_value=[]))
    db.close.return_value = None
    return db


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables after each test."""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_requests():
    """Mock requests module for HTTP testing."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post, \
         patch('requests.put') as mock_put, \
         patch('requests.delete') as mock_delete:
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_response.text = '{"success": true}'
        
        mock_get.return_value = mock_response
        mock_post.return_value = mock_response
        mock_put.return_value = mock_response
        mock_delete.return_value = mock_response
        
        yield {
            'get': mock_get,
            'post': mock_post,
            'put': mock_put,
            'delete': mock_delete,
            'response': mock_response
        }


@pytest.fixture
def captured_output(capsys):
    """Capture stdout and stderr output."""
    def _capture():
        captured = capsys.readouterr()
        return captured.out, captured.err
    return _capture


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def change_test_dir(temp_dir):
    """Change to a temporary directory for the test."""
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    yield temp_dir
    os.chdir(original_cwd)