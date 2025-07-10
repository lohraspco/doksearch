# Tests Directory

This directory contains all test files for the RAG Document Search System.

## Test Files

- **`test_logging.py`** - Tests logging functionality and Unicode support
- **`test_search.py`** - Tests vector store and search functionality
- **`test_search_fixed.py`** - Tests search with fixed settings (lower threshold, device fixes)
- **`test_system.py`** - Tests the complete RAG system
- **`clear_database.py`** - Utility to clear the vector database
- **`run_tests.py`** - Test runner to execute all tests

## Running Tests

### Run All Tests
```bash
# From the project root
python tests/run_tests.py

# Or from the tests directory
cd tests
python run_tests.py
```

### Run Specific Test
```bash
# Run a specific test
python tests/run_tests.py test_logging
python tests/run_tests.py test_search
python tests/run_tests.py test_system
```

### Run Individual Tests
```bash
# From the project root
python tests/test_logging.py
python tests/test_search.py
python tests/test_system.py

# Clear database
python tests/clear_database.py
```

## Test Structure

Each test file follows this pattern:
1. **Setup logging** - Initialize logging for the test
2. **Import modules** - Import the modules to test
3. **Run tests** - Execute test functions
4. **Report results** - Log success/failure

## Log Files

Test logs are written to:
- `logs/test_runner.log` - Test runner logs
- `logs/test_*.log` - Individual test logs

## Adding New Tests

1. Create a new test file in this directory
2. Follow the naming convention: `test_*.py`
3. Add the test to the list in `run_tests.py`
4. Include proper logging and error handling

## Example Test Structure

```python
#!/usr/bin/env python3
"""
Test description
"""

import sys
import os
from logging_config import setup_logging, get_logger
import logging

# Set up logging
loggers = setup_logging(log_level=logging.INFO, log_file="test_name.log")
logger = get_logger('test_name')

def test_function():
    """Test specific functionality."""
    try:
        # Test code here
        logger.info("✅ Test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting test...")
    success = test_function()
    
    if success:
        logger.info("✅ All tests completed successfully")
    else:
        logger.error("❌ Tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
``` 