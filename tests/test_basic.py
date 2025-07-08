"""Simple test to verify pytest setup."""


def test_basic_functionality():
    """Test basic Python functionality."""
    assert 1 + 1 == 2
    assert "hello" + " world" == "hello world"


def test_imports():
    """Test that we can import our modules."""
    from locness_datamanager.config import get_config
    config = get_config()
    assert isinstance(config, dict)


class TestBasic:
    """Basic test class."""
    
    def test_class_method(self):
        """Test class method."""
        assert True
